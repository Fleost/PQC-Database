#!/usr/bin/env python3
"""
analyze_paired.py
=================
Statistics, tables and figures for the paired policy-latency suite
(results/paired/*, produced by app/paired_study.py via run_suite.ps1).

Run on the host after the suite finishes:
    python analyze_paired.py --results results/paired --out results/analysis

Per-record values are the median of the 20 measured iterations for that
(record, profile).  Policies are derived from the two measured tiers: a
record's latency under a policy is its latency under the tier the policy
assigns it.  All comparisons are paired on records.

Statistics per policy, relative to uniform Cat-3 (baseline):
  mean / median / p95 / p99 of per-record latency
  mean paired delta (ms, %) with 95% bootstrap CI (records resampled)
  Wilcoxon signed-rank test on per-record deltas, Holm-corrected
  rank-biserial effect size; Hodges-Lehmann shift (Walsh-average subsample)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RNG = np.random.default_rng(20261105)
B = 10_000  # bootstrap resamples

# Main size-threshold sweep: elevate the largest 50 / 25 / 10 / 5 / 1 % of records.  Each T is the
# record-size quantile of the 50,000-record sample (faers_q25k), so the share elevated is the
# definition and T is what an operator would configure.  (The earlier sweep, 1458/2228/2990/4196/8091,
# only reproduced the original paper's shares on an undocumented dataset.)
SWEEP = {1332: 50, 2067: 25, 3558: 10, 5083: 5, 10901: 1}   # T (bytes) -> % of records elevated
MATCHED = list(SWEEP)
# Equal-share size thresholds, one per content flag (filled in main() from the manifest): the size
# policy that elevates exactly as many records as the flag, for the equal-cost coverage comparison.
EQ_THRESHOLDS: dict = {}
PAPER = [10240, 25600, 51200, 102400, 256000]     # paper's 10/25/50/100/250 KB (KiB)
STAGES = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt"]
CRYPTO = ["encrypt", "sign", "verify", "decrypt"]
COL_BASE, COL_STRONG = "#3b6ea8", "#c8553d"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_run(d: Path):
    files = sorted(glob.glob(str(d / "chunk_*.csv.gz")))
    if not files:
        return None, None, None
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    env = pd.concat((pd.read_csv(f) for f in sorted(glob.glob(str(d / "env_*.csv.gz")))), ignore_index=True)
    meta = json.loads((d / "meta.json").read_text())
    df["crypto_ms"] = df[[f"{s}_ms" for s in CRYPTO]].sum(axis=1)
    df["crypto_cpu_ms"] = df[[f"{s}_cpu_ms" for s in CRYPTO]].sum(axis=1)
    df["db_ms"] = df["db_insert_ms"] + df["db_fetch_ms"]
    return df, env, meta


METRICS = (["crypto_ms", "crypto_cpu_ms", "e2e_ms", "e2e_cpu_ms", "db_ms"]
           + [f"{s}_ms" for s in STAGES] + [f"{s}_cpu_ms" for s in STAGES]
           + ["enc_t_pq_encaps_ms", "dec_t_pq_decaps_ms", "sig_sign_ms", "sig_verify_ms",
              "enc_t_aes_gcm_ms", "dec_t_aes_gcm_ms", "enc_t_ecdh_ms", "dec_t_ecdh_ms",
              "enc_t_hkdf_ms", "dec_t_hkdf_ms", "enc_t_wrap_ms", "dec_t_unwrap_ms", "enc_t_eph_keygen_ms",
              "enc_t_dek_gen_ms"])


def per_record(df: pd.DataFrame) -> pd.DataFrame:
    """Median over iterations -> one row per (record, profile)."""
    m = df[df.kind == "main"]
    cols = [c for c in METRICS + ["policy_ms", "policy_cpu_ms"] if c in m.columns]
    g = m.groupby(["record", "size_bytes", "profile"])[cols].median().reset_index()
    return g


# Content flags per record (index = record file name, columns flag_<rule>),
# loaded from the dataset manifest when it has them (v2 dataset).
FLAGS: "pd.DataFrame | None" = None
FLAG_LABELS = {
    "death": "Flag: death (DE)", "death_pt": "Flag: death incl. death PT",
    "fatal_lt": "Flag: fatal or life-threatening", "severe": "Flag: severe outcome",
    "serious": "Flag: any serious outcome", "pediatric": "Flag: pediatric", "pregnancy": "Flag: pregnancy",
    "suicide": "Flag: suicide / self-harm", "substance": "Flag: substance use", "hiv": "Flag: HIV",
    "special": "Flag: special category", "sensitive": "Flag: death or special category",
    "severe_special": "Flag: severe or special category",
}
MAIN_FLAGS = ["death", "fatal_lt", "special", "sensitive", "severe", "severe_special", "serious"]
COMBO_THRESHOLDS = [1332, 5083]   # death flag OR one of the sweep's thresholds (largest 50 % / 5 %)


def _flag(records, rule: str) -> np.ndarray:
    return FLAGS[f"flag_{rule}"].reindex(records).fillna(0).to_numpy().astype(bool)


def spec_mask(spec: str, records, sizes: np.ndarray) -> np.ndarray:
    """Which records a policy spec (as used by paired_study) elevates."""
    if spec.startswith("adaptive:"):
        return sizes > int(spec.split(":")[1])
    if spec.startswith("flag:"):
        return _flag(records, spec.split(":")[1])
    if spec.startswith("flagsize:"):
        _, rule, t = spec.split(":")
        return _flag(records, rule) | (sizes > int(t))
    raise ValueError(spec)


def wide(pr: pd.DataFrame, metric: str, profiles=("baseline", "strong")) -> pd.DataFrame:
    w = pr.pivot_table(index=["record", "size_bytes"], columns="profile", values=metric).reset_index()
    return w.dropna(subset=list(profiles))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def bootstrap_mean_ratio(policy: np.ndarray, base: np.ndarray):
    """95% CI for mean(policy-base) and mean(policy)/mean(base)-1, records resampled."""
    n = len(base)
    d_means, ratios = np.empty(B), np.empty(B)
    bs = 250
    for i in range(0, B, bs):
        idx = RNG.integers(0, n, size=(min(bs, B - i), n))
        p, b = policy[idx].mean(axis=1), base[idx].mean(axis=1)
        d_means[i:i + len(p)] = p - b
        ratios[i:i + len(p)] = p / b - 1
    return np.percentile(d_means, [2.5, 97.5]), np.percentile(ratios, [2.5, 97.5]) * 100


def hodges_lehmann(d: np.ndarray, max_pairs: int = 2_000_000) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    n = len(d)
    if n * (n + 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n)
        return float(np.median((d[i] + d[j]) / 2))
    i, j = RNG.integers(0, n, max_pairs), RNG.integers(0, n, max_pairs)
    return float(np.median((d[i] + d[j]) / 2))


def wilcoxon(d: np.ndarray):
    nz = d[d != 0]
    if len(nz) < 10:
        return float("nan"), float("nan"), len(nz)
    res = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method="approx")
    ranks = stats.rankdata(np.abs(nz))
    w_plus, w_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    rbc = (w_plus - w_minus) / (len(nz) * (len(nz) + 1) / 2)
    return float(res.pvalue), float(rbc), len(nz)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = np.isfinite(p).sum()
    for rank, k in enumerate(order):
        if not np.isfinite(p[k]):
            adj[k] = np.nan
            continue
        running = max(running, min(1.0, (m - rank) * p[k]))
        adj[k] = running
    return adj


def policies(sizes: np.ndarray, records=None):
    """Return list of (name, label, threshold, strong_mask).

    Flag (selective-elevation) policies are added when FLAGS is loaded and the
    caller passes the record names aligned with ``sizes``."""
    out = [("uniform_baseline", "Uniform Cat. 3", None, np.zeros(len(sizes), bool))]
    for t in MATCHED:
        out.append((f"adaptive_{t}", f"Largest {SWEEP[t]}%, T = {t:,} B", t, sizes > t))
    for t in PAPER:
        out.append((f"adaptive_paper_{t}", f"Adaptive, T = {t // 1024} KiB", t, sizes > t))
    if FLAGS is not None and records is not None:
        for rule in FLAG_LABELS:
            if f"flag_{rule}" in FLAGS:
                out.append((f"flag_{rule}", FLAG_LABELS[rule], None, _flag(records, rule)))
        for rule, t in EQ_THRESHOLDS.items():
            out.append((f"sizeeq_{rule}", f"Size, same share as {rule}, T = {t:,} B", t, sizes > t))
        for t in COMBO_THRESHOLDS:
            out.append((f"flagsize_death_{t}", f"Death or size > {t:,} B", t, _flag(records, "death") | (sizes > t)))
    out.append(("uniform_strong", "Uniform Cat. 5", None, np.ones(len(sizes), bool)))
    return out


def policy_table(pr: pd.DataFrame, metrics=("crypto_ms", "e2e_ms", "crypto_cpu_ms", "db_ms")) -> pd.DataFrame:
    rows = []
    base_w = wide(pr, "crypto_ms")
    sizes = base_w["size_bytes"].to_numpy()
    total_bytes = sizes.sum()
    for name, label, thr, mask in policies(sizes, base_w["record"].to_numpy()):
        row = {"policy": name, "label": label, "threshold_bytes": thr,
               "pct_records_elevated": mask.mean() * 100,
               "pct_plaintext_elevated": sizes[mask].sum() / total_bytes * 100}
        for metric in metrics:
            w = wide(pr, metric).set_index("record").loc[base_w["record"]]
            b, s = w["baseline"].to_numpy(), w["strong"].to_numpy()
            pol = np.where(mask, s, b)
            d = pol - b
            key = metric.replace("_ms", "")
            row[f"{key}_mean"] = pol.mean()
            row[f"{key}_median"] = np.median(pol)
            row[f"{key}_p95"] = np.percentile(pol, 95)
            row[f"{key}_p99"] = np.percentile(pol, 99)
            row[f"{key}_delta_mean_ms"] = d.mean()
            row[f"{key}_delta_pct"] = (pol.mean() / b.mean() - 1) * 100
            if mask.any():
                ci_d, ci_pct = bootstrap_mean_ratio(pol, b)
                p, rbc, n_nz = wilcoxon(d)
                row.update({f"{key}_delta_ci_lo": ci_d[0], f"{key}_delta_ci_hi": ci_d[1],
                            f"{key}_pct_ci_lo": ci_pct[0], f"{key}_pct_ci_hi": ci_pct[1],
                            f"{key}_wilcoxon_p": p, f"{key}_rank_biserial": rbc,
                            f"{key}_hl_shift_elevated_ms": hodges_lehmann(d[mask]),
                            f"{key}_pct_elevated_slower": (d[mask] > 0).mean() * 100,
                            f"{key}_workload_total_s": pol.sum() / 1000})
            else:
                row[f"{key}_workload_total_s"] = pol.sum() / 1000
        rows.append(row)
    t = pd.DataFrame(rows)
    for metric in metrics:
        key = metric.replace("_ms", "")
        col = f"{key}_wilcoxon_p"
        if col in t:
            t[f"{key}_p_holm"] = holm(t[col].to_numpy())
    return t


def storage_table(env: pd.DataFrame) -> pd.DataFrame:
    w = env.pivot_table(index=["record", "size_bytes"], columns="profile",
                        values=["total_stored_bytes", "overhead_bytes"]).reset_index()
    sizes = w["size_bytes"].to_numpy()
    rows = []
    tb, ts = w[("total_stored_bytes", "baseline")].to_numpy(), w[("total_stored_bytes", "strong")].to_numpy()
    ob, os_ = w[("overhead_bytes", "baseline")].to_numpy(), w[("overhead_bytes", "strong")].to_numpy()
    for name, label, thr, mask in policies(sizes, w["record"].to_numpy()):
        tot = np.where(mask, ts, tb).sum()
        ovh = np.where(mask, os_, ob).sum()
        rows.append({"policy": name, "label": label,
                     "pct_records_elevated": mask.mean() * 100,
                     "pct_plaintext_elevated": sizes[mask].sum() / sizes.sum() * 100,
                     "total_stored_bytes": int(tot), "overhead_bytes": int(ovh),
                     "delta_footprint_pct": (tot / tb.sum() - 1) * 100,
                     "delta_overhead_pct": (ovh / ob.sum() - 1) * 100})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# auxiliary experiments
# ---------------------------------------------------------------------------

def coverage_table(pr: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """For every policy: how many records it elevates, and what share of the
    important reports (by each flag) it actually protects at Cat. 5."""
    w = wide(pr, "crypto_ms")
    recs, sizes = w["record"].to_numpy(), w["size_bytes"].to_numpy()
    sto = st.set_index("policy")
    rows = []
    for name, label, thr, mask in policies(sizes, recs):
        row = {"policy": name, "label": label, "pct_records_elevated": mask.mean() * 100,
               "delta_overhead_pct": sto.loc[name, "delta_overhead_pct"] if name in sto.index else np.nan}
        for rule in ("death", "fatal_lt", "special", "sensitive", "severe"):
            f = _flag(recs, rule)
            row[f"pct_{rule}_protected"] = mask[f].mean() * 100 if f.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ladder_table(pr: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    """Classical -> Cat. 3 -> policies -> Cat. 5, each relative to the
    classical baseline (X25519 + ECDSA P-256), paired on records."""
    keep = {"uniform_baseline", "uniform_strong", *(f"adaptive_{t}" for t in MATCHED),
            *(f"flag_{r}" for r in MAIN_FLAGS), *(f"flagsize_death_{t}" for t in COMBO_THRESHOLDS)}
    ov = env.pivot_table(index="record", columns="profile", values="overhead_bytes")
    rows = []
    base = {}
    for metric in ("crypto_ms", "e2e_ms"):
        w = wide(pr, metric, profiles=("classical", "baseline", "strong"))
        base[metric] = w
    recs = base["crypto_ms"]["record"].to_numpy()
    sizes = base["crypto_ms"]["size_bytes"].to_numpy()
    o = ov.reindex(recs)
    oc, ob, os_ = o["classical"].to_numpy(), o["baseline"].to_numpy(), o["strong"].to_numpy()
    entries = [("classical", "Classical (X25519 + ECDSA P-256)", None, None)]
    entries += [(n, l, t, m) for n, l, t, m in policies(sizes, recs) if n in keep]
    for name, label, thr, mask in entries:
        row = {"policy": name, "label": label,
               "pct_records_elevated": np.nan if mask is None else mask.mean() * 100}
        for metric in ("crypto_ms", "e2e_ms"):
            w = base[metric].set_index("record").loc[recs]
            c, b, s = w["classical"].to_numpy(), w["baseline"].to_numpy(), w["strong"].to_numpy()
            pol = c if mask is None else np.where(mask, s, b)
            key = metric.replace("_ms", "")
            row[f"{key}_mean"] = pol.mean()
            row[f"{key}_vs_classical_ms"] = (pol - c).mean()
            row[f"{key}_vs_classical_pct"] = (pol.mean() / c.mean() - 1) * 100
            if mask is not None:
                _, ci = bootstrap_mean_ratio(pol, c)
                row[f"{key}_vs_classical_ci_lo"], row[f"{key}_vs_classical_ci_hi"] = ci
        ovh = oc if mask is None else np.where(mask, os_, ob)
        row["overhead_mean_bytes"] = ovh.mean()
        row["overhead_vs_classical_x"] = ovh.mean() / oc.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def mean_delta_ci(pr: pd.DataFrame, metric: str, a: str = "baseline", b_: str = "strong"):
    """Mean paired delta (b_ - a) with bootstrap CI."""
    w = wide(pr, metric, profiles=(a, b_))
    b, s = w[a].to_numpy(), w[b_].to_numpy()
    ci_d, ci_pct = bootstrap_mean_ratio(s, b)
    return {"n": len(b), "base_mean": b.mean(), "strong_mean": s.mean(), "delta_mean": (s - b).mean(),
            "delta_ci": list(ci_d), "pct": (s.mean() / b.mean() - 1) * 100, "pct_ci": list(ci_pct)}


def quarter_tables(pr: pd.DataFrame, env: pd.DataFrame, quarters: pd.DataFrame):
    """Per-quarter robustness: the E3 policy and storage tables recomputed
    within each source quarter, plus a test that the Cat. 5 penalty does not
    differ between quarters (unpaired: different records)."""
    rows, tests = [], {}
    pr = pr.merge(quarters, on="record")
    env = env.merge(quarters, on="record")
    qs = sorted(pr.source_quarter.unique())
    for q in qs:
        pq = pr[pr.source_quarter == q]
        lat = policy_table(pq, metrics=("crypto_ms", "e2e_ms")).set_index("policy")
        sto = storage_table(env[env.source_quarter == q]).set_index("policy")
        for pol in lat.index:
            r, s = lat.loc[pol], sto.loc[pol]
            rows.append({"quarter": q, "policy": pol, "label": r.label,
                         "n_records": int(pq.record.nunique()),
                         "pct_records_elevated": r.pct_records_elevated,
                         "pct_plaintext_elevated": r.pct_plaintext_elevated,
                         "crypto_mean_ms": r.crypto_mean, "crypto_delta_pct": r.crypto_delta_pct,
                         "crypto_pct_ci_lo": r.get("crypto_pct_ci_lo"), "crypto_pct_ci_hi": r.get("crypto_pct_ci_hi"),
                         "e2e_mean_ms": r.e2e_mean, "e2e_delta_pct": r.e2e_delta_pct,
                         "e2e_pct_ci_lo": r.get("e2e_pct_ci_lo"), "e2e_pct_ci_hi": r.get("e2e_pct_ci_hi"),
                         "e2e_p_holm": r.get("e2e_p_holm"),
                         "delta_footprint_pct": s.delta_footprint_pct, "delta_overhead_pct": s.delta_overhead_pct})
    # Between-quarter comparison of the per-record Cat. 5 - Cat. 3 delta.
    if len(qs) == 2:
        for metric in ("crypto_ms", "e2e_ms"):
            w = wide(pr, metric).merge(quarters, on="record")
            d = w["strong"] - w["baseline"]
            a, b = d[w.source_quarter == qs[0]].to_numpy(), d[w.source_quarter == qs[1]].to_numpy()
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            diff = a.mean() - b.mean()
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            tests[metric] = {"quarters": qs, "n": [len(a), len(b)],
                             "delta_mean_ms": [float(a.mean()), float(b.mean())],
                             "diff_of_delta_ms": float(diff), "diff_ci95_ms": [float(diff - 1.96 * se), float(diff + 1.96 * se)],
                             "mannwhitney_p": float(u.pvalue)}
    return pd.DataFrame(rows), tests


def to_latex_quarters(qt: pd.DataFrame, flags: bool = False) -> str:
    qs = sorted(qt.quarter.unique())
    if flags:
        want = [f"flag_{r}" for r in MAIN_FLAGS] + [f"flagsize_death_{t}" for t in COMBO_THRESHOLDS] + ["uniform_strong"]
        keep = qt[qt.policy.isin(want)]
    else:
        keep = qt[~qt.policy.str.startswith(("adaptive_paper", "flag", "sizeeq")) & (qt.policy != "uniform_baseline")]
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{20{q[:2]} {q[2:]} ($n$={int(qt[qt.quarter == q].n_records.iloc[0]):,})}}"
                      for q in qs)
    lines = [r"\begin{tabular}{l" + "rrr" * len(qs) + "}", r"\toprule",
             f"Configuration & {head} \\\\",
             " & " + " & ".join([r"\% Elev. & $\Delta$E2E & $\Delta$Ovh."] * len(qs)) + r" \\", r"\midrule"]
    for pol in keep.policy.drop_duplicates():
        cells = []
        for q in qs:
            r = keep[(keep.quarter == q) & (keep.policy == pol)].iloc[0]
            cells += [f"{r.pct_records_elevated:.1f}\\%", f"{r.e2e_delta_pct:+.2f}\\%", f"{r.delta_overhead_pct:+.2f}\\%"]
        lines.append(f"{keep[keep.policy == pol].label.iloc[0]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def e5_validation(df: pd.DataFrame, margin_pct: float = 2.0) -> list:
    """Direct adaptive vs tier measured in the same run, per record (TOST)."""
    pr = per_record(df)
    out = []
    specs = sorted(p for p in pr.profile.unique() if p.split(":")[0] in ("adaptive", "flag", "flagsize"))
    if FLAGS is None:
        specs = [s for s in specs if s.startswith("adaptive:")]
    for spec in specs:
        for metric in ("crypto_ms", "e2e_ms"):
            w = pr.pivot_table(index=["record", "size_bytes"], columns="profile", values=metric).reset_index()
            mask = spec_mask(spec, w["record"].to_numpy(), w["size_bytes"].to_numpy())
            derived = np.where(mask, w["strong"], w["baseline"])
            direct = w[spec].to_numpy()
            d = direct - derived
            n, md, sd = len(d), d.mean(), d.std(ddof=1)
            se = sd / np.sqrt(n)
            margin = margin_pct / 100 * derived.mean()
            t_lo, t_hi = (md + margin) / se, (md - margin) / se
            p_tost = max(1 - stats.t.cdf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
            ci90 = (md - stats.t.ppf(0.95, n - 1) * se, md + stats.t.ppf(0.95, n - 1) * se)
            out.append({"profile": spec, "metric": metric, "n": n, "pct_elevated": float(mask.mean() * 100),
                        "derived_mean": derived.mean(),
                        "direct_mean": direct.mean(), "mean_diff_ms": md, "ci90_ms": list(ci90),
                        "margin_ms": margin, "tost_p": p_tost, "equivalent": bool(p_tost < 0.05)})
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def fig_cost_coverage(pt: pd.DataFrame, st: pd.DataFrame, out: Path):
    # Both threshold sets lie on one threshold continuum, so sorting by coverage
    # gives a single monotone curve.
    sel = pt[~pt.policy.str.startswith(("flag", "sizeeq"))].sort_values("pct_plaintext_elevated").copy()
    sst = st.set_index("policy").loc[sel.policy]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = sel.pct_plaintext_elevated.to_numpy()
    for key, lab, mk in (("crypto", "Crypto stages", "o"), ("e2e", "End-to-end", "s")):
        y = sel[f"{key}_delta_pct"].to_numpy()
        lo = (y - sel[f"{key}_pct_ci_lo"].fillna(0)).to_numpy()
        hi = (sel[f"{key}_pct_ci_hi"].fillna(0) - y).to_numpy()
        axes[0].errorbar(x, y, yerr=[lo, hi], marker=mk, capsize=3, label=lab)
    axes[0].set_xlabel("Plaintext bytes elevated to Cat. 5 (%)")
    axes[0].set_ylabel("Mean latency change vs Cat. 3 (%)")
    axes[0].legend(frameon=False)
    axes[0].set_title("Latency")
    axes[1].plot(x, sst.delta_footprint_pct, marker="o", label="Footprint")
    axes[1].plot(x, sst.delta_overhead_pct, marker="s", label="Overhead")
    axes[1].set_xlabel("Plaintext bytes elevated to Cat. 5 (%)")
    axes[1].set_ylabel("Storage change vs Cat. 3 (%)")
    axes[1].legend(frameon=False)
    axes[1].set_title("Storage")
    for ax in axes:
        ax.grid(alpha=.3)
    _save(fig, out, "fig_cost_coverage")


def fig_flag_policies(pt: pd.DataFrame, st: pd.DataFrame, cov: pd.DataFrame, out: Path):
    """Size-threshold vs content-flag policies on a common x-axis (% records
    elevated): latency cost, storage cost, and share of death reports protected."""
    sto, cv = st.set_index("policy"), cov.set_index("policy")
    size = pt[pt.policy.str.startswith(("adaptive", "uniform"))].sort_values("pct_records_elevated")
    flag = pt[pt.policy.isin([f"flag_{r}" for r in MAIN_FLAGS] + [f"flagsize_death_{t}" for t in COMBO_THRESHOLDS])]
    short = {"death": "death", "fatal_lt": "fatal/LT", "severe": "severe", "serious": "serious",
             "special": "special", "sensitive": "death+special", "severe_special": "severe+special"}
    short = {f"flag_{r}": short.get(r, r) for r in MAIN_FLAGS}
    short.update({f"flagsize_death_{t}": f"death or >{t:,} B" for t in COMBO_THRESHOLDS})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [(lambda d: d.e2e_delta_pct, "End-to-end latency vs Cat. 3 (%)"),
              (lambda d: sto.loc[d.policy, "delta_overhead_pct"].to_numpy(), "Storage overhead vs Cat. 3 (%)"),
              (lambda d: cv.loc[d.policy, "pct_death_protected"].to_numpy(), "Death reports at Cat. 5 (%)")]
    for ax, (f, ylab) in zip(axes, panels):
        ax.plot(size.pct_records_elevated, f(size), marker="o", color=COL_BASE, label="Size threshold")
        ax.scatter(flag.pct_records_elevated, f(flag), marker="^", s=45, color=COL_STRONG, zorder=3,
                   label="Content flag")
        ax.set_xlabel("Records elevated to Cat. 5 (%)"); ax.set_ylabel(ylab); ax.grid(alpha=.3)
    # Cost panels: flags fall on the size line (cost depends only on share
    # elevated), so they are left unlabelled.  Coverage panel: stagger labels
    # with leader lines, since every death-based flag sits at 100 %.
    ax = axes[2]
    fl = flag.assign(y=np.asarray(panels[2][0](flag))).sort_values("pct_records_elevated")
    heights, k = [88, 78, 68, 58], 0
    for x, y, p in zip(fl.pct_records_elevated, fl.y, fl.policy):
        if y >= 99.5:
            ty, k = heights[k % len(heights)], k + 1
            ax.annotate(short[p], (x, y), xytext=(x + 1.5, ty), fontsize=7, va="center",
                        arrowprops=dict(arrowstyle="-", lw=.5, color="0.5"))
        else:
            ax.annotate(short[p], (x, y), textcoords="offset points", xytext=(5, -3), fontsize=7)
    ax.set_ylim(-5, 108)
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, out, "fig_flag_policies")


def fig_stage_breakdown(pr: pd.DataFrame, out: Path):
    profs = [p for p in ("classical", "baseline", "strong") if p in set(pr.profile)]
    names = {"classical": "Classical\n(X25519/P-256)", "baseline": "Cat. 3\n(768/65)", "strong": "Cat. 5\n(1024/87)"}
    fig, ax = plt.subplots(figsize=(7, 1.2 + 1.0 * len(profs)))
    labels = [names[p] for p in profs]
    left = np.zeros(len(profs))
    colors = plt.cm.tab10(np.arange(len(STAGES)))
    for k, s in enumerate(STAGES):
        vals = np.array([pr[pr.profile == p][f"{s}_ms"].mean() for p in profs])
        ax.barh(labels, vals, left=left, color=colors[k], label=s.replace("_", " "))
        left += vals
    ax.set_xlabel("Mean per-record latency (ms)")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.25))
    _save(fig, out, "fig_stage_breakdown")


def fig_paired_delta(pr: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, metric, title in ((axes[0], "crypto_ms", "Crypto stages"), (axes[1], "e2e_ms", "End-to-end")):
        w = wide(pr, metric)
        d = (w["strong"] - w["baseline"]).to_numpy()
        lo, hi = np.percentile(d, [0.5, 99.5])
        ax.hist(np.clip(d, lo, hi), bins=120, color=COL_STRONG, alpha=.8)
        ax.axvline(0, color="k", lw=.8)
        ax.axvline(np.median(d), color="k", ls="--", lw=.8, label=f"median {np.median(d):+.3f} ms")
        ax.set_title(f"{title}: Cat. 5 − Cat. 3 per record")
        ax.set_xlabel("Paired difference (ms)")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Records")
    _save(fig, out, "fig_paired_delta")


def fig_size(pr: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    w = wide(pr, "crypto_ms")
    bins = np.unique(np.percentile(w.size_bytes, np.linspace(0, 100, 11)))
    w["bin"] = pd.cut(w.size_bytes, bins, include_lowest=True)
    g = w.groupby("bin", observed=True).agg(size=("size_bytes", "median"), b=("baseline", "median"), s=("strong", "median"))
    ax.plot(g["size"] / 1024, g.b, marker="o", color=COL_BASE, label="Cat. 3")
    ax.plot(g["size"] / 1024, g.s, marker="s", color=COL_STRONG, label="Cat. 5")
    ax.set_xscale("log")
    ax.set_xlabel("Record size (KiB, decile median)")
    ax.set_ylabel("Crypto latency (ms)")
    ax.legend(frameon=False); ax.grid(alpha=.3)
    _save(fig, out, "fig_latency_vs_size")


def fig_drift(runs: dict, stability: list, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    df = runs["e3_main"][0]
    s = df[df.kind == "sentinel"]
    for prof, col in (("baseline", COL_BASE), ("strong", COL_STRONG)):
        g = s[s.profile == prof].groupby("seq").agg(t=("t_rel_s", "min"), e2e=("e2e_ms", "median"), c=("crypto_ms", "median"))
        axes[0].plot(g.t / 60, g.c, color=col, marker=".", label=f"{prof} crypto")
        axes[0].plot(g.t / 60, g.e2e, color=col, ls=":", label=f"{prof} e2e")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Minutes into E3"); axes[0].set_ylabel("Sentinel latency (ms, log)")
    axes[0].legend(frameon=False, fontsize=7); axes[0].set_title("Drift (fixed sentinel record)")
    if stability:
        pretty = {"e3_subset": "E3 (subset)", "e6_stability_r1": "E6 run 1\n(start)",
                  "e6_stability_r2": "E6 run 2\n(middle)", "e6_stability_r3": "E6 run 3\n(end)"}
        names = [pretty.get(r["run"], r["run"]) for r in stability]
        for k, (key, mk) in enumerate((("crypto", "o"), ("e2e", "s"))):
            y = [r[key]["delta_mean"] for r in stability]
            lo = [r[key]["delta_mean"] - r[key]["delta_ci"][0] for r in stability]
            hi = [r[key]["delta_ci"][1] - r[key]["delta_mean"] for r in stability]
            axes[1].errorbar(np.arange(len(names)) + k * .1, y, yerr=[lo, hi], marker=mk, ls="none", capsize=3,
                             label=key)
        axes[1].set_xticks(range(len(names)), names, fontsize=8)
        axes[1].set_ylabel("Mean Cat. 5 − Cat. 3 (ms)"); axes[1].legend(frameon=False)
        axes[1].set_title("Between-run stability (5,000-record subset)")
    _save(fig, out, "fig_drift_stability")


def fig_primitives(pr: pd.DataFrame, out: Path, title: str):
    prims = [("enc_t_pq_encaps_ms", "ML-KEM encaps"), ("dec_t_pq_decaps_ms", "ML-KEM decaps"),
             ("sig_sign_ms", "Sign"), ("sig_verify_ms", "Verify"),
             ("enc_t_ecdh_ms", "X25519 (enc)"), ("enc_t_aes_gcm_ms", "AES-GCM enc")]
    profs = [(p, c, l) for p, c, l in (("classical", "#7a7a7a", "Classical (ECDSA P-256)"),
                                       ("baseline", COL_BASE, "Cat. 3 (ML-DSA-65)"),
                                       ("strong", COL_STRONG, "Cat. 5 (ML-DSA-87)")) if p in set(pr.profile)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(prims))
    bw = .8 / len(profs)
    for k, (prof, col, lab) in enumerate(profs):
        v = [pr[pr.profile == prof][c].median() * 1000 for c, _ in prims]
        ax.bar(x + (k - (len(profs) - 1) / 2) * bw, v, bw, color=col, label=lab)
    ax.set_xticks(x, [l for _, l in prims], rotation=20, fontsize=8)
    ax.set_ylabel("Median per call (µs)"); ax.legend(frameon=False); ax.set_title(title)
    _save(fig, out, "fig_primitives")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def to_latex_latency(pt: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Configuration & \% Rec. Elev. & Crypto (ms) & $\Delta$Crypto & End-to-end (ms) & $\Delta$E2E & $p_{\text{Holm}}$ \\",
             r"\midrule"]
    for _, r in pt.iterrows():
        if r.policy == "uniform_baseline":
            dc = de = p = "—"
        else:
            dc = f"{r.crypto_delta_pct:+.2f}\\% [{r.crypto_pct_ci_lo:+.2f}, {r.crypto_pct_ci_hi:+.2f}]"
            de = f"{r.e2e_delta_pct:+.2f}\\% [{r.e2e_pct_ci_lo:+.2f}, {r.e2e_pct_ci_hi:+.2f}]"
            p = "$<10^{-300}$" if r.crypto_p_holm == 0 else f"{r.crypto_p_holm:.1e}"
        lines.append(f"{r.label} & {r.pct_records_elevated:.1f}\\% & {r.crypto_mean:.3f} & {dc} & "
                     f"{r.e2e_mean:.3f} & {de} & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def to_latex_ladder(lad: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Configuration & \% Elev. & Crypto (ms) & vs.\ classical & End-to-end (ms) & vs.\ classical & Overhead (B) \\",
             r"\midrule"]
    for _, r in lad.iterrows():
        if r.policy == "classical":
            dc = de = "—"
            elev = "—"
        else:
            dc = f"{r.crypto_vs_classical_pct:+.1f}\\%"
            de = f"{r.e2e_vs_classical_pct:+.1f}\\%"
            elev = f"{r.pct_records_elevated:.1f}\\%"
        lines.append(f"{r.label} & {elev} & {r.crypto_mean:.3f} & {dc} & {r.e2e_mean:.3f} & {de} & "
                     f"{r.overhead_mean_bytes:,.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/paired")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--manifest", default="Data/faers_records_manifest.csv",
                    help="Record manifest with source_quarter (for the per-quarter table)")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for d in sorted(p for p in res.iterdir() if p.is_dir()):
        df, env, meta = load_run(d)
        if df is not None:
            runs[d.name] = (df, env, meta)
            print(f"loaded {d.name}: {len(df):,} rows, {df.record.nunique():,} records")
    summary = {"runs": {k: {"rows": len(v[0]), "records": int(v[0].record.nunique()), "meta": v[2]}
                        for k, v in runs.items()}}

    # ── manifest: quarters and (v2) content flags ────────────────────────────
    global FLAGS
    manifest = pd.read_csv(args.manifest) if Path(args.manifest).exists() else None
    if manifest is not None:
        fcols = [c for c in manifest.columns if c.startswith("flag_")]
        if fcols:
            FLAGS = manifest.set_index("file")[fcols]
            summary["flag_prevalence_pct"] = {c[5:]: float(manifest[c].mean() * 100) for c in fcols}
            # equal-share size threshold per main flag: smallest T with share(size > T) <= flag share
            s_sorted = np.sort(manifest.size_bytes.to_numpy())
            n = len(s_sorted)
            for rule in MAIN_FLAGS:
                k = int(np.ceil(n * (1 - manifest[f"flag_{rule}"].mean())))
                EQ_THRESHOLDS[rule] = int(s_sorted[k - 1])
            summary["size_eq_thresholds"] = dict(EQ_THRESHOLDS)
        # the sweep's thresholds are quantiles of faers_q25k; warn if this manifest is a different sample
        for t, p in SWEEP.items():
            got = (manifest.size_bytes > t).mean() * 100
            if abs(got - p) >= 0.1:
                print(f"WARNING: T = {t:,} B elevates {got:.2f}% of this manifest's records, not {p}%")

    # ── E3 main ──────────────────────────────────────────────────────────────
    df3, env3, _ = runs["e3_main"]
    pr3 = per_record(df3)
    has_classical = "classical" in set(pr3.profile)
    pr3.to_csv(out / "e3_per_record.csv.gz", index=False)
    pt = policy_table(pr3)
    pt.to_csv(out / "table_policy_latency.csv", index=False)
    st = storage_table(env3)
    st.to_csv(out / "table_policy_storage.csv", index=False)
    size_pols = pt.policy.str.startswith(("uniform", "adaptive_")) & ~pt.policy.str.startswith("adaptive_paper")
    (out / "table_policy_latency.tex").write_text(to_latex_latency(pt[size_pols]))
    (out / "table_policy_latency_paper_thresholds.tex").write_text(
        to_latex_latency(pt[pt.policy.str.startswith("adaptive_paper") | pt.policy.str.startswith("uniform")]))
    if FLAGS is not None:
        want = (["uniform_baseline"] + [f"flag_{r}" for r in MAIN_FLAGS]
                + [f"flagsize_death_{t}" for t in COMBO_THRESHOLDS] + ["uniform_strong"])
        (out / "table_policy_latency_flags.tex").write_text(
            to_latex_latency(pt.set_index("policy").loc[want].reset_index()))
        cov = coverage_table(pr3, st)
        cov.to_csv(out / "table_policy_coverage.csv", index=False)
        fig_flag_policies(pt, st, cov, out)
        # cost of classifying a report (JSON parse + flag check), measured per record in E3
        if "classify_death_ms" in env3:
            cl = env3.drop_duplicates("record").classify_death_ms * 1000
            summary["classify_death_us"] = {"median": float(cl.median()), "p95": float(cl.quantile(.95)),
                                            "p99": float(cl.quantile(.99)), "max": float(cl.max())}
    if has_classical:
        lad = ladder_table(pr3, env3)
        lad.to_csv(out / "table_tier_ladder.csv", index=False)
        (out / "table_tier_ladder.tex").write_text(to_latex_ladder(lad))

    # ── E3 by source quarter (robustness) ────────────────────────────────────
    if manifest is not None:
        quarters = manifest[["file", "source_quarter"]].rename(columns={"file": "record"})
        qt, qtests = quarter_tables(pr3, env3, quarters)
        qt.to_csv(out / "table_policy_by_quarter.csv", index=False)
        (out / "table_policy_by_quarter.tex").write_text(to_latex_quarters(qt))
        if FLAGS is not None:
            (out / "table_policy_by_quarter_flags.tex").write_text(to_latex_quarters(qt, flags=True))
        summary["quarter_comparison"] = qtests

    stage_rows = []
    for metric in [f"{s}_ms" for s in STAGES] + [f"{s}_cpu_ms" for s in STAGES] + [
            "crypto_ms", "crypto_cpu_ms", "db_ms", "e2e_ms", "e2e_cpu_ms",
            "enc_t_pq_encaps_ms", "dec_t_pq_decaps_ms", "sig_sign_ms", "sig_verify_ms"]:
        w = wide(pr3, metric)
        d = w["strong"] - w["baseline"]
        row = {"metric": metric, "baseline_mean": w.baseline.mean(), "strong_mean": w.strong.mean(),
               "baseline_median": w.baseline.median(), "strong_median": w.strong.median(),
               "delta_mean": d.mean(), "delta_median": d.median(),
               "delta_pct": (w.strong.mean() / w.baseline.mean() - 1) * 100,
               "pct_records_strong_slower": (d > 0).mean() * 100}
        if has_classical and "classical" in w:
            row.update({"classical_mean": w.classical.mean(), "classical_median": w.classical.median(),
                        "baseline_vs_classical_ms": (w.baseline - w.classical).mean(),
                        "strong_vs_classical_ms": (w.strong - w.classical).mean()})
        stage_rows.append(row)
    pd.DataFrame(stage_rows).to_csv(out / "table_stage_breakdown.csv", index=False)

    # within-record noise
    m = df3[df3.kind == "main"]
    noise = m.groupby(["record", "profile"])[["crypto_ms", "crypto_cpu_ms", "db_insert_ms", "e2e_ms"]].agg(
        lambda x: x.std() / x.mean() * 100).median()
    summary["within_record_cv_median_pct"] = noise.to_dict()
    summary["db_wait_ms_mean"] = float((m.db_insert_ms - m.db_insert_cpu_ms).mean()
                                       + (m.db_fetch_ms - m.db_fetch_cpu_ms).mean())
    summary["crypto_descheduled_ms_mean"] = float((m.crypto_ms - m.crypto_cpu_ms).mean())

    # ── E6 stability ─────────────────────────────────────────────────────────
    stability = []
    for name in sorted(k for k in runs if k.startswith("e6_stability")):
        prs = per_record(runs[name][0])
        entry = {"run": name, "crypto": mean_delta_ci(prs, "crypto_ms"), "e2e": mean_delta_ci(prs, "e2e_ms")}
        if "classical" in set(prs.profile):
            entry["classical_to_cat3"] = {"crypto": mean_delta_ci(prs, "crypto_ms", "classical", "baseline"),
                                          "e2e": mean_delta_ci(prs, "e2e_ms", "classical", "baseline")}
        stability.append(entry)
    # same subset measured inside E3 for comparison
    if stability:
        r1 = runs[sorted(k for k in runs if k.startswith("e6_stability"))[0]][0]
        subset = set(r1[r1.kind == "main"].record.unique())  # exclude the sentinel record
        pr3s = pr3[pr3.record.isin(subset)]
        stability.insert(0, {"run": "e3_subset", "crypto": mean_delta_ci(pr3s, "crypto_ms"),
                             "e2e": mean_delta_ci(pr3s, "e2e_ms")})
        for key in ("crypto", "e2e"):
            vals = np.array([r[key]["delta_mean"] for r in stability])
            summary[f"stability_{key}_between_run_cv_pct"] = float(vals.std(ddof=1) / vals.mean() * 100)
            pcts = np.array([r[key]["pct"] for r in stability])
            summary[f"stability_{key}_pct_range"] = [float(pcts.min()), float(pcts.max())]
    summary["stability"] = stability

    # ── E5 direct adaptive ───────────────────────────────────────────────────
    # e5_direct (2026-11-05: earlier thresholds + flags) and e5_direct_sweep (2026-11-11: the current
    # size sweep and its combined policies) are validated the same way; each entry names its run.
    e5_runs = [k for k in ("e5_direct", "e5_direct_sweep") if k in runs]
    if e5_runs:
        summary["e5_validation"] = [dict(v, run=k) for k in e5_runs for v in e5_validation(runs[k][0])]
    for k, key in (("e5_direct", "e5_policy_decision_us"), ("e5_direct_sweep", "e5_sweep_policy_decision_us")):
        d5 = runs[k][0] if k in runs else None
        if d5 is not None and "policy_ms" in d5:
            m5 = d5[d5.kind == "main"]
            summary[key] = {
                p: {"median": float(g.policy_ms.median() * 1000), "p99": float(g.policy_ms.quantile(.99) * 1000),
                    "cpu_median": float(g.policy_cpu_ms.median() * 1000)}
                for p, g in m5.groupby("profile")}

    # ── E7 crypto only / E8 async commit ─────────────────────────────────────
    for name in ("e7_nodb", "e8_async_commit"):
        if name in runs:
            prx = per_record(runs[name][0])
            profs = [p for p in ("classical", "baseline", "strong") if p in set(prx.profile)]
            summary[name] = {"crypto": mean_delta_ci(prx, "crypto_ms"), "e2e": mean_delta_ci(prx, "e2e_ms"),
                             "db_insert_mean_ms": {p: float(prx[prx.profile == p].db_insert_ms.mean()) for p in profs},
                             "within_record_cv_e2e": float(runs[name][0][runs[name][0].kind == "main"]
                                                           .groupby(["record", "profile"]).e2e_ms
                                                           .agg(lambda x: x.std() / x.mean()).median())}
            if "classical" in profs:
                summary[name]["classical_to_cat3"] = {
                    "crypto": mean_delta_ci(prx, "crypto_ms", "classical", "baseline"),
                    "e2e": mean_delta_ci(prx, "e2e_ms", "classical", "baseline")}
                summary[name]["classical_to_cat5"] = {
                    "crypto": mean_delta_ci(prx, "crypto_ms", "classical", "strong"),
                    "e2e": mean_delta_ci(prx, "e2e_ms", "classical", "strong")}
            if name == "e7_nodb":
                fig_primitives(prx, out, "Primitive latency, no database (E7)")
                prims = {}
                for c in ("enc_t_pq_encaps_ms", "dec_t_pq_decaps_ms", "sig_sign_ms", "sig_verify_ms",
                          "enc_t_ecdh_ms", "enc_t_aes_gcm_ms"):
                    prims[c] = {p: float(prx[prx.profile == p][c].median()) for p in profs}
                summary["e7_primitives_median_ms"] = prims

    # ── figures ──────────────────────────────────────────────────────────────
    fig_cost_coverage(pt, st, out)
    fig_stage_breakdown(pr3, out)
    fig_paired_delta(pr3, out)
    fig_size(pr3, out)
    fig_drift(runs, stability, out)

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(pt[["label", "pct_records_elevated", "pct_plaintext_elevated", "crypto_mean", "crypto_delta_pct",
              "crypto_pct_ci_lo", "crypto_pct_ci_hi", "e2e_mean", "e2e_delta_pct", "e2e_pct_ci_lo",
              "e2e_pct_ci_hi"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(st[["label", "delta_footprint_pct", "delta_overhead_pct"]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
