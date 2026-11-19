"""
app/feasibility.py
==================
Short capacity and noise probe that runs before the full policy-latency study.

It answers three planning questions for the machine it runs on:

1. Throughput: how long does one put+get round-trip take under the
   Category-3 profile (ML-KEM-768 + ML-DSA-65) and the Category-5 profile
   (ML-KEM-1024 + ML-DSA-87), and how does that grow with record size?
2. Noise: how much do repeated measurements of the same record vary, and
   does the machine drift over time?  From this it estimates how many
   iterations and records are needed for tight confidence intervals.
3. Projection: how long would the full study take for N records x K
   iterations, both as 7 separate policy runs (old design) and as the
   paired design, which measures each record once under each of the two
   profiles and derives all policies from those measurements.

Run inside the benchmark container:
    python -m app.feasibility --records 400 --iterations 20 \
        --output /app/results/feasibility/probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from . import db, stage2
from .benchmark import (
    DATA_DIR,
    _clear_benchmark_data,
    _resolve_policy_decision,
    _run_single,
)
from .policy import BASELINE_KEM_ID, BASELINE_SIG_ID, STRONG_KEM_ID, STRONG_SIG_ID

PROFILES = {
    "baseline": "uniform_baseline",  # Cat-3
    "strong": "uniform_strong",      # Cat-5
}
STAGES = ["encrypt_ms", "sign_ms", "db_insert_ms", "db_fetch_ms", "verify_ms", "decrypt_ms", "end_to_end_ms"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pct(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _ols(x: List[float], y: List[float]):
    """Return (intercept, slope, r2) for y = a + b x."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((yi - my) ** 2 for yi in y)
    ss_res = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return a, b, r2


def _fmt_dur(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 5400:
        return f"{sec / 60:.0f} min"
    if sec < 172800:
        return f"{sec / 3600:.1f} h"
    return f"{sec / 86400:.1f} d"


def _stratified(paths: List[Path], n: int, tail: int, rng: random.Random) -> List[Path]:
    ordered = sorted(paths, key=lambda p: (p.stat().st_size, p.name))
    if n >= len(ordered):
        return ordered
    picks = {round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)}
    picks |= set(range(len(ordered) - tail, len(ordered)))  # always include the largest records
    sel = [ordered[i] for i in sorted(picks)]
    rng.shuffle(sel)
    return sel


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def run_probe(conn, *, records: int, iterations: int, warmup: int, tail: int,
              sentinel_every: int, seed: int) -> Dict:
    rng = random.Random(seed)
    all_paths = sorted((DATA_DIR / "faers_records").glob("*.json"))
    if not all_paths:
        raise SystemExit("No records in Data/faers_records — run scripts/build_faers_records.py first")
    population_sizes = [p.stat().st_size for p in all_paths]
    chosen = _stratified(all_paths, records, tail, rng)
    sentinel = sorted(all_paths, key=lambda p: p.stat().st_size)[len(all_paths) // 2]
    sentinel_pt = sentinel.read_bytes()

    print("  generating keys ...", flush=True)
    hybrid_keys = {
        BASELINE_KEM_ID: stage2.generate_hybrid_recipient_keys(BASELINE_KEM_ID),
        STRONG_KEM_ID: stage2.generate_hybrid_recipient_keys(STRONG_KEM_ID),
    }
    sig_keys = {
        BASELINE_SIG_ID: stage2.generate_sig_keys(BASELINE_SIG_ID, key_id="bench-sig-key-v1"),
        STRONG_SIG_ID: stage2.generate_sig_keys(STRONG_SIG_ID, key_id="bench-sig-key-v1"),
    }

    def one(pt: bytes, profile: str, name: str):
        mode = PROFILES[profile]
        decision = _resolve_policy_decision(mode, len(pt), 0)
        return _run_single(
            conn, scheme="hybrid", plaintext=pt,
            aad=f"benchmark|scheme=hybrid|size={len(pt)}".encode(),
            classical_keys=None, hybrid_keys_by_kem=hybrid_keys, sig_keys_by_sig=sig_keys,
            decision=decision, effective_policy_mode=mode, payload_name=name,
        )

    per_record: List[Dict] = []
    sentinel_log: List[Dict] = []
    t_start = time.perf_counter()
    round_trips = 0

    for idx, path in enumerate(chosen, 1):
        pt = path.read_bytes()
        _clear_benchmark_data(conn)
        order = ["baseline", "strong"]
        rng.shuffle(order)
        for _ in range(warmup):
            for prof in order:
                one(pt, prof, path.name)
                round_trips += 1
        samples: Dict[str, List[Dict[str, float]]] = {"baseline": [], "strong": []}
        for i in range(iterations):
            seq = order if i % 2 == 0 else order[::-1]  # ABBA counterbalancing
            for prof in seq:
                s = one(pt, prof, path.name)
                round_trips += 1
                samples[prof].append({k: getattr(s, k) for k in STAGES})
        per_record.append({"name": path.name, "size_bytes": len(pt), "samples": samples})

        if sentinel_every and idx % sentinel_every == 0:
            _clear_benchmark_data(conn)
            vals = {p: [one(sentinel_pt, p, sentinel.name).end_to_end_ms for _ in range(5)] for p in PROFILES}
            round_trips += 10
            sentinel_log.append({
                "t_sec": time.perf_counter() - t_start,
                "baseline_ms": statistics.median(vals["baseline"]),
                "strong_ms": statistics.median(vals["strong"]),
            })
        if idx % 25 == 0 or idx == len(chosen):
            el = time.perf_counter() - t_start
            print(f"  [{idx:>4}/{len(chosen)}] {el:6.1f}s elapsed, {round_trips / el:7.1f} round-trips/s", flush=True)

    wall = time.perf_counter() - t_start
    return {
        "population_sizes": population_sizes,
        "per_record": per_record,
        "sentinel": {"name": sentinel.name, "size_bytes": len(sentinel_pt), "log": sentinel_log},
        "wall_sec": wall,
        "round_trips": round_trips,
    }


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyze(raw: Dict, *, iterations: int, warmup: int) -> Dict:
    recs = raw["per_record"]
    pop = raw["population_sizes"]
    out: Dict = {"profiles": {}}

    # Per-record medians and noise.
    for prof in PROFILES:
        sizes, med_e2e, cvs = [], [], []
        stage_meds = {k: [] for k in STAGES}
        for r in recs:
            vals = [s["end_to_end_ms"] for s in r["samples"][prof]]
            sizes.append(r["size_bytes"])
            med_e2e.append(statistics.median(vals))
            if len(vals) > 1 and statistics.mean(vals) > 0:
                cvs.append(statistics.stdev(vals) / statistics.mean(vals))
            for k in STAGES:
                stage_meds[k].append(statistics.median(s[k] for s in r["samples"][prof]))
        a, b, r2 = _ols([float(s) for s in sizes], med_e2e)
        pop_mean_rt = a + b * (sum(pop) / len(pop))
        cvs.sort()
        out["profiles"][prof] = {
            "fit_intercept_ms": a,
            "fit_ms_per_kib": b * 1024,
            "fit_r2": r2,
            "predicted_population_mean_ms": pop_mean_rt,
            "median_record_e2e_ms": statistics.median(med_e2e),
            "per_record_cv_median": _pct(cvs, 50),
            "per_record_cv_p90": _pct(cvs, 90),
            "stage_median_ms": {k: statistics.median(v) for k, v in stage_meds.items()},
        }

    # Paired strong - baseline deltas (per record, per iteration pair).
    deltas, rel = [], []
    pair_cv = []
    for r in recs:
        bl = [s["end_to_end_ms"] for s in r["samples"]["baseline"]]
        st = [s["end_to_end_ms"] for s in r["samples"]["strong"]]
        d = [x - y for x, y in zip(st, bl)]
        md = statistics.median(d)
        deltas.append(md)
        rel.append(md / statistics.median(bl) * 100 if statistics.median(bl) else 0.0)
        if len(d) > 1:
            pair_cv.append(statistics.stdev(d))
    sd_rec = statistics.stdev(deltas) if len(deltas) > 1 else float("nan")
    mean_d = statistics.mean(deltas)
    out["paired_delta"] = {
        "median_ms": statistics.median(deltas),
        "mean_ms": mean_d,
        "sd_across_records_ms": sd_rec,
        "median_within_record_sd_ms": statistics.median(pair_cv) if pair_cv else float("nan"),
        "median_relative_pct": statistics.median(rel),
        "records_with_strong_slower_pct": sum(1 for d in deltas if d > 0) / len(deltas) * 100,
    }

    # Sample-size guidance.
    cv_med = max(out["profiles"]["baseline"]["per_record_cv_median"], out["profiles"]["strong"]["per_record_cv_median"])
    out["guidance"] = {
        # iterations so a per-record mean has 95% CI half-width <= 5% / 2%
        "iterations_for_5pct_ci_per_record": math.ceil((1.96 * cv_med / 0.05) ** 2),
        "iterations_for_2pct_ci_per_record": math.ceil((1.96 * cv_med / 0.02) ** 2),
        # records so the workload-mean paired delta has 95% CI half-width <= 10% / 5% of itself
        "records_for_10pct_ci_on_delta": math.ceil((1.96 * sd_rec / (0.10 * abs(mean_d))) ** 2) if mean_d else None,
        "records_for_5pct_ci_on_delta": math.ceil((1.96 * sd_rec / (0.05 * abs(mean_d))) ** 2) if mean_d else None,
    }

    # Drift.
    log = raw["sentinel"]["log"]
    if len(log) >= 3:
        for prof in PROFILES:
            ys = [e[f"{prof}_ms"] for e in log]
            third = max(1, len(ys) // 3)
            first, last = statistics.median(ys[:third]), statistics.median(ys[-third:])
            out.setdefault("drift", {})[prof] = {
                "sentinel_cv": statistics.stdev(ys) / statistics.mean(ys),
                "last_vs_first_third_pct": (last - first) / first * 100,
                "min_ms": min(ys), "max_ms": max(ys),
            }

    # Projection.
    obs_rt_per_sec = raw["round_trips"] / raw["wall_sec"]
    bl_rt = out["profiles"]["baseline"]["predicted_population_mean_ms"]
    st_rt = out["profiles"]["strong"]["predicted_population_mean_ms"]
    # per-record fixed overhead (table clear etc.) estimated from observed throughput vs modelled
    probe_mean_rt = statistics.mean(
        statistics.mean(s["end_to_end_ms"] for s in r["samples"][p]) for r in recs for p in PROFILES
    )
    harness_overhead = max(0.0, (1000.0 / obs_rt_per_sec) / probe_mean_rt)  # wall / measured ratio
    proj = []
    n_pop = len(pop)
    for n in sorted({1000, 5000, 10000, 25000, n_pop}):
        if n > n_pop:
            continue
        for k in (1, 3, 5, 10, 20):
            per_iter = (k + warmup)
            paired = n * per_iter * (bl_rt + st_rt) / 1000 * harness_overhead
            # old design: 7 policies run separately; adaptive runs are mostly baseline records
            separate = n * per_iter * (5 * bl_rt + bl_rt + st_rt) / 1000 * harness_overhead
            proj.append({"records": n, "iterations": k, "paired_2profile_sec": paired, "separate_7policy_sec": separate})
    out["projection"] = proj
    out["observed"] = {
        "wall_sec": raw["wall_sec"],
        "round_trips": raw["round_trips"],
        "round_trips_per_sec": obs_rt_per_sec,
        "harness_overhead_factor": harness_overhead,
        "population_records": n_pop,
        "population_mean_size_bytes": sum(pop) / n_pop,
    }
    return out


def print_report(a: Dict, meta: Dict) -> None:
    print()
    print("=" * 78)
    print(" FEASIBILITY PROBE")
    print("=" * 78)
    print(f" host cpus={meta['cpu_count']}  python={meta['python']}  probe records={meta['records_measured']}"
          f"  iterations={meta['iterations']}  warmup={meta['warmup']}")
    o = a["observed"]
    print(f" wall {o['wall_sec']:.0f}s, {o['round_trips']:,} round-trips, {o['round_trips_per_sec']:.0f}/s "
          f"(harness overhead x{o['harness_overhead_factor']:.2f})")
    print()
    print(f" {'profile':<9} {'median e2e':>11} {'intercept':>10} {'ms/KiB':>8} {'R2':>5} "
          f"{'pop mean':>9} {'CV p50':>7} {'CV p90':>7}")
    for prof, p in a["profiles"].items():
        print(f" {prof:<9} {p['median_record_e2e_ms']:>9.3f}ms {p['fit_intercept_ms']:>8.3f}ms "
              f"{p['fit_ms_per_kib']:>8.4f} {p['fit_r2']:>5.2f} {p['predicted_population_mean_ms']:>7.3f}ms "
              f"{p['per_record_cv_median'] * 100:>6.1f}% {p['per_record_cv_p90'] * 100:>6.1f}%")
    print()
    print(" stage medians (ms):  " + "  ".join(s.replace("_ms", "") for s in STAGES))
    for prof, p in a["profiles"].items():
        print(f"   {prof:<9} " + "  ".join(f"{p['stage_median_ms'][s]:.3f}" for s in STAGES))
    d = a["paired_delta"]
    print()
    print(f" paired Cat5 - Cat3: median {d['median_ms']:+.3f} ms ({d['median_relative_pct']:+.1f}%), "
          f"strong slower on {d['records_with_strong_slower_pct']:.0f}% of records, "
          f"sd across records {d['sd_across_records_ms']:.3f} ms")
    g = a["guidance"]
    print(f" iterations/record for +/-5% CI: {g['iterations_for_5pct_ci_per_record']}, "
          f"+/-2%: {g['iterations_for_2pct_ci_per_record']}")
    print(f" records for +/-10% CI on mean delta: {g['records_for_10pct_ci_on_delta']}, "
          f"+/-5%: {g['records_for_5pct_ci_on_delta']}")
    if "drift" in a:
        for prof, dr in a["drift"].items():
            print(f" drift {prof:<9}: sentinel CV {dr['sentinel_cv'] * 100:.1f}%, "
                  f"last vs first third {dr['last_vs_first_third_pct']:+.1f}%, "
                  f"range {dr['min_ms']:.3f}-{dr['max_ms']:.3f} ms")
    print()
    print(f" {'records':>8} {'iters':>5}  {'paired (2 profiles)':>20}  {'7 separate runs':>16}")
    for r in a["projection"]:
        print(f" {r['records']:>8,} {r['iterations']:>5}  {_fmt_dur(r['paired_2profile_sec']):>20}  "
              f"{_fmt_dur(r['separate_7policy_sec']):>16}")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=int, default=400, help="records to probe, stratified by size")
    ap.add_argument("--tail", type=int, default=10, help="always include this many of the largest records")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sentinel-every", type=int, default=20, help="re-measure a fixed record every N records (drift)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="/app/results/feasibility/probe.json")
    args = ap.parse_args()

    cfg = db.config_from_env()
    with db.connect(cfg) as conn:
        db.init_schema(conn)
        raw = run_probe(conn, records=args.records, iterations=args.iterations, warmup=args.warmup,
                        tail=args.tail, sentinel_every=args.sentinel_every, seed=args.seed)

    analysis = analyze(raw, iterations=args.iterations, warmup=args.warmup)
    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "records_measured": len(raw["per_record"]),
        "iterations": args.iterations,
        "warmup": args.warmup,
    }
    print_report(analysis, meta)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_small = {k: v for k, v in raw.items() if k != "population_sizes"}
    out.write_text(json.dumps({"meta": meta, "analysis": analysis, "raw": raw_small}, indent=1))
    print(f"\n wrote {out}")


if __name__ == "__main__":
    main()
