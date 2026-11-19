"""
app/paired_study.py
===================
Paired, interleaved latency study for policy-driven PQ parameter selection.

Design
------
* Every record is measured under every requested profile, back-to-back, so
  policies are compared on the same record under the same machine state
  (a within-record paired design).  The default profiles are the two NIST
  tiers used by the policy: ``baseline`` (ML-KEM-768 + ML-DSA-65, Cat. 3)
  and ``strong`` (ML-KEM-1024 + ML-DSA-87, Cat. 5).  Uniform and adaptive
  policies at any threshold are then derived offline, because a record's
  latency under a policy is its latency under the profile that policy
  assigns it.  ``adaptive:<bytes>`` profiles run the policy directly (E5).
* Profile order is counterbalanced: a random starting order per record,
  reversed on every other iteration (ABBA).
* Warmup round-trips per profile are run and discarded.
* Tables are TRUNCATEd at the start of every chunk (default 500 records);
  inserts only, so no dead-tuple build-up.
* Python's garbage collector is disabled during each record's measured
  iterations and run between records, outside any timer.
* Every stage is timed twice: wall clock (``time.perf_counter``) and
  thread CPU time (``time.thread_time``).  For crypto stages the gap
  between the two is time the thread was descheduled; for DB stages it is
  time spent waiting on PostgreSQL (network, execution, WAL flush).
* A fixed median-size sentinel record is re-measured every
  ``--sentinel-every`` records to track machine drift.
* Each round-trip's decrypted plaintext is checked against the input
  (outside the timers).

The round-trip mirrors ``service.put_record`` / ``service.get_record``
step-for-step (encrypt -> sign -> INSERT+COMMIT -> audit INSERT+COMMIT ->
SELECT -> verify -> decrypt), with an extra CPU-time read at each boundary.

Output (``--out-dir``), written in chunks so a long run can be resumed:
    meta.json                 run parameters, library and host versions
    chunk_00000.csv.gz ...    one row per measured round-trip
    env_00000.csv.gz ...      stored-envelope sizes per (record, profile)
    done.txt                  completed chunk ids

Examples
--------
    # E3 main study: all records, both tiers, 10 iterations
    python -m app.paired_study --out-dir /app/results/paired/e3_main

    # E5 direct adaptive validation on a 5,000-record subset
    python -m app.paired_study --records 5000 --profiles adaptive:2990,adaptive:8091 \
        --out-dir /app/results/paired/e5_direct

    # E7 crypto-only (no database)
    python -m app.paired_study --records 5000 --no-db --out-dir /app/results/paired/e7_nodb
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import os
import platform
import random
import statistics
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from . import db, elevation, stage2
from .benchmark import DATA_DIR, KEY_ID, TENANT_ID
from .policy import (
    BASELINE_KEM_ID, BASELINE_SIG_ID, STRONG_KEM_ID, STRONG_SIG_ID,
    ThresholdPolicyConfig, select_policy_for_record,
)
from .service import ServiceContext, _row_to_record

STAGES = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt", "e2e"]
ENC_OPS = ["t_dek_gen_ms", "t_aes_gcm_ms", "t_eph_keygen_ms", "t_ecdh_ms", "t_pq_encaps_ms", "t_hkdf_ms", "t_wrap_ms"]
DEC_OPS = ["t_ecdh_ms", "t_pq_decaps_ms", "t_hkdf_ms", "t_unwrap_ms", "t_aes_gcm_ms"]
COLUMNS = (
    ["kind", "seq", "record", "size_bytes", "profile", "kem_id", "sig_id", "tier", "iter", "pos", "t_rel_s"]
    + [f"{s}_ms" for s in STAGES]
    + [f"{s}_cpu_ms" for s in STAGES]
    + [f"enc_{o}" for o in ENC_OPS]
    + [f"dec_{o}" for o in DEC_OPS]
    + ["sig_sign_ms", "sig_verify_ms"]
    + ["policy_ms", "policy_cpu_ms"]
)
ENV_COLUMNS = ["seq", "record", "size_bytes", "profile", "kem_id", "sig_id", "tier",
               "pq_ct_bytes", "signature_bytes", "header_bytes", "overhead_bytes", "total_stored_bytes",
               "classify_death_ms"]

# Classical baseline: X25519-ECDH + HKDF + AES-KW (no ML-KEM) and ECDSA P-256.
CLASSICAL_KEM_ID = "X25519"
CLASSICAL_SIG_ID = stage2.SIG_ID_ECDSA_P256

PROFILE_HELP = ("classical, baseline, strong, adaptive:<bytes>, flag:<rule>, flagsize:<rule>:<bytes> "
                f"(rules: {', '.join(elevation.RULES)})")


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

def _tier(name: str) -> Tuple[str, str, str]:
    if name == "strong":
        return STRONG_KEM_ID, STRONG_SIG_ID, "strong"
    return BASELINE_KEM_ID, BASELINE_SIG_ID, "baseline"


def resolve_profile(spec: str, size: int, data: bytes = b"{}") -> Tuple[str, str, str]:
    """Return (kem_id, sig_id, tier) for a profile spec and record.

    ``data`` is the serialized record; only content-based (flag) policies read
    it.  The harness times this call per round-trip (``policy_ms``)."""
    if spec == "classical":
        return CLASSICAL_KEM_ID, CLASSICAL_SIG_ID, "classical"
    if spec in ("baseline", "strong"):
        return _tier(spec)
    if spec.startswith("adaptive:"):
        d = select_policy_for_record(size, ThresholdPolicyConfig(threshold_bytes=int(spec.split(":", 1)[1])))
        return d.kem_id, d.sig_id, d.tier_label
    if spec.startswith("flag:"):
        return _tier("strong" if elevation.rule_matches(spec.split(":", 1)[1], data) else "baseline")
    if spec.startswith("flagsize:"):
        _, rule, thr = spec.split(":")
        # size check first: it is free, and skips parsing for large records
        hit = size > int(thr) or elevation.rule_matches(rule, data)
        return _tier("strong" if hit else "baseline")
    raise ValueError(f"unknown profile {spec!r} (use {PROFILE_HELP})")


# ---------------------------------------------------------------------------
# one round-trip
# ---------------------------------------------------------------------------

class Harness:
    def __init__(self, conn, *, no_db: bool):
        self.conn = conn
        self.no_db = no_db
        self.ctx = ServiceContext(tenant_id=TENANT_ID, key_id=KEY_ID, version=1)
        self.kem_keys = {k: stage2.generate_hybrid_recipient_keys(k) for k in (BASELINE_KEM_ID, STRONG_KEM_ID)}
        self.classical_keys = stage2.generate_classical_recipient_keys()
        self.sig_keys = {s: stage2.generate_sig_keys(s, key_id="bench-sig-key-v1")
                         for s in (BASELINE_SIG_ID, STRONG_SIG_ID, CLASSICAL_SIG_ID)}

    def truncate(self) -> None:
        if self.no_db:
            return
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE encrypted_records, audit_events")
        self.conn.commit()

    def round_trip(self, plaintext: bytes, kem_id: str, sig_id: str) -> Dict:
        pc, tt = time.perf_counter, time.thread_time
        classical = kem_id == CLASSICAL_KEM_ID
        skeys = self.sig_keys[sig_id]
        if classical:
            ckeys, pq_pk, pq_sk = self.classical_keys, None, None
        else:
            hk = self.kem_keys[kem_id]
            ckeys, pq_pk, pq_sk = hk.classical, hk.pq_public_key, hk.pq_sk
        scheme = "classical" if classical else "hybrid"
        aad = f"benchmark|scheme={scheme}|size={len(plaintext)}".encode()
        w, c = {}, {}

        w0, c0 = pc(), tt()
        # ── write path ───────────────────────────────────────────────────────
        if classical:
            rec = stage2.encrypt(plaintext, scheme="classical", recipient_classical_pk=ckeys.pk, aad=aad)
        else:
            rec = stage2.encrypt(plaintext, scheme="hybrid", recipient_classical_pk=ckeys.pk,
                                 recipient_pq_pk=pq_pk, pq_kem_id=kem_id, aad=aad)
        w1, c1 = pc(), tt()
        rec, sign_ops = stage2.sign_envelope(rec, skeys)
        w2, c2 = pc(), tt()
        env = stage2.envelope_sizes(rec)
        w2b, c2b = pc(), tt()
        if not self.no_db:
            record_id, _ = db.insert_record(
                self.conn, version=rec.version, tenant_id=self.ctx.tenant_id, key_id=self.ctx.key_id,
                scheme=rec.scheme, payload_cipher=rec.payload_cipher, kem_id=rec.kem_id, kdf_id=rec.kdf_id,
                wrap_id=rec.wrap_id, pq_kem_id=rec.pq_kem_id, aad=rec.aad, nonce=rec.nonce, tag=rec.tag,
                ciphertext=rec.ciphertext, wrapped_dek=rec.wrapped_dek, eph_pubkey=rec.ephemeral_pubkey,
                salt=rec.salt, hkdf_info=rec.hkdf_info, pq_ct=rec.pq_ciphertext, sig_alg_id=rec.sig_alg_id,
                signature=rec.signature, sig_key_id=rec.sig_key_id,
            )
            db.audit_event(self.conn, tenant_id=self.ctx.tenant_id, record_id=record_id, event_type="put",
                           key_id=self.ctx.key_id, scheme=scheme, status="ok")
        w3, c3 = pc(), tt()
        # ── read path ────────────────────────────────────────────────────────
        if not self.no_db:
            row, _ = db.fetch_record(self.conn, tenant_id=TENANT_ID, record_id=record_id)
            got = _row_to_record(row)
        else:
            got = rec
        w4, c4 = pc(), tt()
        verify_ops = stage2.verify_envelope(got, skeys.verify_key)
        w5, c5 = pc(), tt()
        pt, dec_ops = stage2.decrypt(got, recipient_classical_sk=ckeys.sk, recipient_pq_sk=pq_sk)
        w6, c6 = pc(), tt()

        if pt != plaintext:
            raise AssertionError("round-trip plaintext mismatch")

        ms = lambda a, b: (b - a) * 1000.0
        w = {"encrypt": ms(w0, w1), "sign": ms(w1, w2), "db_insert": ms(w2b, w3), "db_fetch": ms(w3, w4),
             "verify": ms(w4, w5), "decrypt": ms(w5, w6), "e2e": ms(w0, w6)}
        c = {"encrypt": ms(c0, c1), "sign": ms(c1, c2), "db_insert": ms(c2b, c3), "db_fetch": ms(c3, c4),
             "verify": ms(c4, c5), "decrypt": ms(c5, c6), "e2e": ms(c0, c6)}
        return {"wall": w, "cpu": c, "enc_ops": rec.metrics or {}, "dec_ops": dec_ops or {},
                "sign_ops": sign_ops, "verify_ops": verify_ops, "env": env}


def _row(kind, seq, name, size, spec, kem, sig, tier, it, pos, t_rel, r, pol=(0.0, 0.0)) -> List:
    f = lambda v: f"{v:.5f}"
    return (
        [kind, seq, name, size, spec, kem, sig, tier, it, pos, f"{t_rel:.3f}"]
        + [f(r["wall"][s]) for s in STAGES]
        + [f(r["cpu"][s]) for s in STAGES]
        + [f(r["enc_ops"].get(o, 0.0)) for o in ENC_OPS]
        + [f(r["dec_ops"].get(o, 0.0)) for o in DEC_OPS]
        + [f(r["sign_ops"].get("t_sig_sign_ms", 0.0)), f(r["verify_ops"].get("t_sig_verify_ms", 0.0))]
        + [f(pol[0]), f(pol[1])]
    )


def _time_policy(spec: str, size: int, data: bytes):
    """Time one policy decision (wall, CPU) in ms; returns (resolved, (w, c))."""
    w0, c0 = time.perf_counter(), time.thread_time()
    res = resolve_profile(spec, size, data)
    return res, ((time.perf_counter() - w0) * 1000.0, (time.thread_time() - c0) * 1000.0)


def _classify_cost_ms(data: bytes, reps: int = 5) -> float:
    """Median cost of classifying one serialized report with the death rule."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        elevation.rule_matches("death", data)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

class Rec:
    __slots__ = ("name", "data")

    def __init__(self, name: str, data: bytes):
        self.name, self.data = name, data


def load_records(dataset: str = "faers_records") -> List[Rec]:
    """Load every record of ``Data/<dataset>`` into memory, sorted by name.

    Prefers Data/<dataset>.tar (one sequential read; per-file reads over a
    Docker Desktop bind mount cost ~20-30 ms each).  Falls back to the folder.
    """
    bundle = DATA_DIR / f"{dataset}.tar"
    recs: List[Rec] = []
    if bundle.exists():
        with tarfile.open(bundle) as tf:
            for m in tf:
                if m.isfile() and m.name.endswith(".json"):
                    recs.append(Rec(Path(m.name).name, tf.extractfile(m).read()))
    else:
        for p in (DATA_DIR / dataset).glob("*.json"):
            recs.append(Rec(p.name, p.read_bytes()))
    if not recs:
        raise SystemExit("No records found — run scripts/build_faers_records.py first")
    recs.sort(key=lambda r: r.name)
    print(f"  loaded {len(recs):,} records ({sum(len(r.data) for r in recs) / 1e6:.1f} MB)"
          f" from {'bundle' if bundle.exists() else 'folder'}", flush=True)
    return recs


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def run(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for p in profiles:
        resolve_profile(p, 1)  # validate

    all_recs = load_records(args.dataset)
    rng = random.Random(args.seed)
    paths = list(all_recs)
    if args.records and args.records < len(paths):
        if args.stratify_quarter:
            # equal share from each source quarter (read from the record itself)
            by_q: Dict[str, List[Rec]] = {}
            for r in paths:
                by_q.setdefault(json.loads(r.data)["source_quarter"], []).append(r)
            per = args.records // len(by_q)
            paths = []
            for q in sorted(by_q):
                paths += rng.sample(by_q[q], per)
            print(f"  stratified subset: {per:,} per quarter from {sorted(by_q)}", flush=True)
        else:
            paths = rng.sample(paths, args.records)
        paths = sorted(paths, key=lambda r: r.name)
    order_rng = random.Random(args.seed + 1)
    order_rng.shuffle(paths)  # measurement order is randomised, fixed by seed
    sentinel = sorted(all_recs, key=lambda r: (len(r.data), r.name))[len(all_recs) // 2]
    sentinel_pt = sentinel.data

    chunks = [paths[i:i + args.chunk_size] for i in range(0, len(paths), args.chunk_size)]
    done_file = out / "done.txt"
    done = set(int(x) for x in done_file.read_text().split()) if done_file.exists() else set()

    conn = None if args.no_db else db.connect(db.config_from_env())
    if conn is not None:
        db.init_schema(conn)
        if args.sync_commit == "off":
            with conn.cursor() as cur:
                cur.execute("SET synchronous_commit TO off")
            conn.commit()
    harness = Harness(conn, no_db=args.no_db)
    # Move long-lived objects (loaded records, modules, keys) into the permanent
    # generation so the per-record gc.collect() only scans fresh garbage.
    gc.collect()
    gc.freeze()

    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "profiles": profiles,
        "records_total": len(paths),
        "chunks_total": len(chunks),
        "chunks_already_done": sorted(done),
        "sentinel": {"record": sentinel.name, "size_bytes": len(sentinel_pt)},
        "host": platform.node(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import oqs  # noqa: WPS433
        meta["liboqs_version"] = oqs.oqs_version()
        meta["liboqs_python_version"] = oqs.oqs_python_version()
    except Exception as exc:  # pragma: no cover
        meta["liboqs_version"] = f"unavailable: {exc}"
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute("SELECT version()"); meta["postgres_version"] = cur.fetchone()[0]
            cur.execute("SHOW synchronous_commit"); meta["synchronous_commit"] = cur.fetchone()[0]
        conn.commit()
    meta_path = out / "meta.json"
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        prev.setdefault("resumes", []).append(meta)
        meta_path.write_text(json.dumps(prev, indent=1))
    else:
        meta_path.write_text(json.dumps(meta, indent=1))

    print(f"  profiles={profiles}  records={len(paths):,}  iterations={args.iterations}  warmup={args.warmup}"
          f"  chunks={len(chunks)} (done {len(done)})  no_db={args.no_db}  sync_commit={args.sync_commit}", flush=True)

    t_start = time.perf_counter()
    rt_count = 0
    seq_base = 0
    for ci, chunk in enumerate(chunks):
        if ci in done:
            seq_base += len(chunk)
            continue
        rows: List[List] = []
        env_rows: List[List] = []
        # Clear tables once per chunk: TRUNCATE costs ~50 ms (lock + file swap
        # + flush), too slow per record.  Tables hold at most one chunk's rows
        # (~23k at 20 iterations) and are insert-only, so no dead tuples build up.
        harness.truncate()
        for j, path in enumerate(chunk):
            seq = seq_base + j
            pt = path.data
            size = len(pt)
            resolved = {p: resolve_profile(p, size, pt) for p in profiles}
            order = list(profiles)
            order_rng.shuffle(order)

            gc.collect()
            if args.gc_off:
                gc.disable()
            try:
                classify_ms = _classify_cost_ms(pt)
                for _ in range(args.warmup):
                    for p in order:
                        harness.round_trip(pt, resolved[p][0], resolved[p][1])
                for it in range(args.iterations):
                    seq_order = order if it % 2 == 0 else order[::-1]
                    for pos, p in enumerate(seq_order):
                        # the policy decision is re-made (and timed) every round-trip
                        (kem, sig, tier), pol = _time_policy(p, size, pt)
                        t_rel = time.perf_counter() - t_start
                        r = harness.round_trip(pt, kem, sig)
                        rows.append(_row("main", seq, path.name, size, p, kem, sig, tier, it, pos, t_rel, r, pol))
                        rt_count += 1
                        if it == 0:
                            e = r["env"]
                            env_rows.append([seq, path.name, size, p, kem, sig, tier, e["pq_ct_bytes"],
                                             e.get("signature_bytes", 0), e["header_bytes"], e["overhead_bytes"],
                                             e["total_stored_bytes"], f"{classify_ms:.5f}"])
            finally:
                if args.gc_off:
                    gc.enable()

            if args.sentinel_every and (seq + 1) % args.sentinel_every == 0:
                gc.collect()
                if args.gc_off:
                    gc.disable()
                try:
                    sp = ["baseline", "strong"]
                    for it in range(args.sentinel_reps):
                        for pos, p in enumerate(sp if it % 2 == 0 else sp[::-1]):
                            kem, sig, tier = resolve_profile(p, len(sentinel_pt))
                            t_rel = time.perf_counter() - t_start
                            r = harness.round_trip(sentinel_pt, kem, sig)
                            rows.append(_row("sentinel", seq, sentinel.name, len(sentinel_pt), p, kem, sig, tier,
                                             it, pos, t_rel, r))
                finally:
                    if args.gc_off:
                        gc.enable()

        # Write the chunk atomically, then mark it done.
        tmp = out / f"chunk_{ci:05d}.csv.gz.tmp"
        with gzip.open(tmp, "wt", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(COLUMNS); wr.writerows(rows)
        tmp.replace(out / f"chunk_{ci:05d}.csv.gz")
        tmp = out / f"env_{ci:05d}.csv.gz.tmp"
        with gzip.open(tmp, "wt", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(ENV_COLUMNS); wr.writerows(env_rows)
        tmp.replace(out / f"env_{ci:05d}.csv.gz")
        with done_file.open("a") as fh:
            fh.write(f"{ci}\n")
        done.add(ci)
        seq_base += len(chunk)

        el = time.perf_counter() - t_start
        remaining = len(chunks) - len(done)
        per_chunk = el / max(1, ci + 1 - len(meta["chunks_already_done"]))
        e2e = [float(r_[COLUMNS.index("e2e_ms")]) for r_ in rows if r_[0] == "main"]
        print(f"  chunk {ci + 1:>4}/{len(chunks)}  {el / 60:6.1f} min  {rt_count / el:6.0f} rt/s  "
              f"median e2e {statistics.median(e2e):.3f} ms  ETA {remaining * per_chunk / 60:5.1f} min", flush=True)

    meta_final = json.loads(meta_path.read_text())
    meta_final["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_final["wall_sec_this_session"] = time.perf_counter() - t_start
    meta_path.write_text(json.dumps(meta_final, indent=1))
    if conn is not None:
        conn.close()
    print(f"  done: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--profiles", default="baseline,strong", help="comma list of " + PROFILE_HELP)
    ap.add_argument("--dataset", default="faers_records",
                    help="record set under Data/ (reads Data/<dataset>.tar, else the folder)")
    ap.add_argument("--records", type=int, default=0, help="0 = all records; N = seeded random subset")
    ap.add_argument("--stratify-quarter", action="store_true",
                    help="with --records N: take N/quarters records from each source quarter")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--sentinel-every", type=int, default=500)
    ap.add_argument("--sentinel-reps", type=int, default=6)
    ap.add_argument("--gc-off", action=argparse.BooleanOptionalAction, default=True,
                    help="disable Python GC during measured iterations (default on)")
    ap.add_argument("--no-db", action="store_true", help="crypto only: skip PostgreSQL (E7)")
    ap.add_argument("--sync-commit", choices=["on", "off"], default="on", help="PostgreSQL synchronous_commit (E8)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
