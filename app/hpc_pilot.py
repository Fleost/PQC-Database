"""Quarter-separated diagnostic pilot using the original service and crypto code.

This runner supersedes the legacy benchmark CLI for the new FAERS pilot scope.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time
import uuid

import psycopg
from psycopg import sql
from . import db, service, stage2
from .policy import ThresholdPolicyConfig, select_policy_for_record

THRESHOLDS = (10, 25, 50, 100, 250)
STAGES = ("encrypt", "sign", "db_insert", "put_total", "db_fetch", "verify", "decrypt", "get_total", "roundtrip")

def check(condition, message):
    if not condition:
        raise ValueError(message)

def percentile(values, percent):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * percent / 100
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)

def summarize(values):
    check(all(math.isfinite(x) and x >= 0 for x in values), "Invalid latency observation")
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                stddev=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values) if values else None, max=max(values) if values else None,
                median=percentile(values, 50), p95=percentile(values, 95), p99=percentile(values, 99))

def load_dataset(directory):
    manifest = directory / "metadata" / "manifest.jsonl"
    entries, seen, cases = [], set(), set()
    quarter = directory.name
    for line in manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        name = item["filename"]
        check(Path(name).name == name and name.endswith(".json"), "Unsafe manifest filename")
        check(name not in seen, "Duplicate manifest filename")
        check(item["quarter"] == quarter, "Quarter mismatch")
        check(item["caseid"] not in cases, "Duplicate case in quarter")
        seen.add(name)
        cases.add(item["caseid"])
        check(all(type(item[k]) is bool for k in ("reported_death", "reported_life_threatening", "elevation_flag")), "Flags must be booleans")
        check(item["elevation_flag"] == (item["reported_death"] or item["reported_life_threatening"]), "Inconsistent DE/LT flag")
        check(item["elevation_rule"] == "faers-de-lt-v1", "Unknown elevation rule")
        payload = (directory / "faers_records" / name).read_bytes()
        check(len(payload) == item["payload_bytes"] and hashlib.sha256(payload).hexdigest() == item["sha256"], "Payload checksum or size mismatch")
        doc = json.loads(payload)
        check(set(doc) == {"demographics", "drugs", "reactions", "indications", "outcomes", "report_sources", "therapies"}, "Expected seven source-faithful sections")
        demo = {key.lower(): value for key, value in doc["demographics"].items()}
        check(all(demo[key] == item[key] for key in ("primaryid", "caseid", "caseversion")), "Manifest identifiers disagree with demographics")
        for section, values in doc.items():
            if section == "demographics":
                continue
            check(isinstance(values, list), "Child section must be an array")
            for row in values:
                identifiers = {key.lower(): value for key, value in row.items()}
                check(identifiers["primaryid"] == item["primaryid"] and identifiers["caseid"] == item["caseid"], "Child identifiers disagree with demographics")
        codes = {next(v for k,v in row.items() if k.lower() == "outc_cod").upper() for row in doc["outcomes"]}
        check(item["reported_death"] == ("DE" in codes) and item["reported_life_threatening"] == ("LT" in codes), "Flags disagree with source outcomes")
        entries.append(item)
    check(entries, "Empty dataset")
    check({p.name for p in (directory / "faers_records").iterdir()} == seen, "Manifest membership mismatch")
    return entries, hashlib.sha256(manifest.read_bytes()).hexdigest()

def diagnostic_sample(entries, count, seed):
    check(0 < count <= len(entries), "Pilot sample count exceeds available documents")
    ordered = sorted(entries, key=lambda item: (item["payload_bytes"], item["filename"]))
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in ordered:
        bin_index = sum(item["payload_bytes"] > threshold * 1024 for threshold in THRESHOLDS)
        groups[(item["elevation_flag"], bin_index)].append(item)
    selected = {ordered[0]["filename"]: ordered[0], ordered[-1]["filename"]: ordered[-1]}
    for group in groups.values():
        item = rng.choice(group)
        selected[item["filename"]] = item
    check(len(selected) <= count, f"Sample must be at least {len(selected)} to cover the available strata")
    remainder = [item for item in ordered if item["filename"] not in selected]
    for item in rng.sample(remainder, count - len(selected)):
        selected[item["filename"]] = item
    return sorted(selected.values(), key=lambda item: item["filename"])

def configurations():
    result = [dict(mode="uniform_baseline", threshold=None), dict(mode="uniform_strong", threshold=None)]
    result += [dict(mode=mode, threshold=threshold * 1024) for mode in ("adaptive_size", "adaptive_size_or_importance") for threshold in THRESHOLDS]
    return result

def assignment(item, config):
    if config["mode"].startswith("uniform_"):
        strong = config["mode"] == "uniform_strong"
        return dict(tier="strong" if strong else "baseline", reason="uniform", kem="ML-KEM-1024" if strong else "ML-KEM-768", sig="ML-DSA-87" if strong else "ML-DSA-65")
    decision = select_policy_for_record(item["payload_bytes"], ThresholdPolicyConfig(threshold_bytes=config["threshold"]),
                                        elevation_flag=item["elevation_flag"] if config["mode"] == "adaptive_size_or_importance" else False)
    return dict(tier=decision.tier_label, reason=decision.elevation_reason, kem=decision.kem_id, sig=decision.sig_id)

def security_selftest(conn):
    """Assert rejection outside the try block so a failed check cannot pass."""
    db.init_schema(conn)
    for kem, sig in zip(("ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"), ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87")):
        recipient, signer = stage2.generate_hybrid_recipient_keys(kem), stage2.generate_sig_keys(sig)
        ctx = service.ServiceContext("selftest", "selftest")
        rid, _ = service.put_record(conn, ctx=ctx, plaintext=b"pilot-selftest", aad=b"test", scheme="hybrid", recipient_keys=recipient, pq_kem_id=kem, sig_keys=signer)
        kwargs = dict(tenant_id=ctx.tenant_id, record_id=rid, recipient_keys=recipient, pq_sk=recipient.pq_sk,
                      sig_verify_key=signer.verify_key, require_signature=True, expected_sig_id=sig)
        plaintext, _ = service.get_record(conn, **kwargs)
        check(plaintext == b"pilot-selftest", "Selftest plaintext mismatch")
        for change, expected_exception in (("aad = aad || decode('ff','hex')", stage2.SigVerificationError),
                                           ("sig_alg_id = NULL, signature = NULL, sig_key_id = NULL", ValueError)):
            conn.execute("UPDATE encrypted_records SET " + change + " WHERE id = %s", (rid,))
            conn.commit()
            rejected = False
            try:
                service.get_record(conn, **kwargs)
            except expected_exception:
                rejected = True
            check(rejected, "Security selftest accepted tampered or stripped signature")
    conn.execute("TRUNCATE encrypted_records, audit_events RESTART IDENTITY")
    conn.commit()

def run_configuration(conn, directory, entries, config, output, run_id, manifest_hash, iterations, warmups, order_seed):
    db.init_schema(conn)
    check(conn.execute("SELECT COUNT(*) FROM encrypted_records").fetchone()[0] == 0, "Expected a fresh database")
    keys = {kem: stage2.generate_hybrid_recipient_keys(kem) for kem in ("ML-KEM-768", "ML-KEM-1024")}
    signers = {sig: stage2.generate_sig_keys(sig) for sig in ("ML-DSA-65", "ML-DSA-87")}
    ctx = service.ServiceContext("pilot", "pilot-key")
    payloads = {item["filename"]: (directory / "faers_records" / item["filename"]).read_bytes() for item in entries}
    groups = defaultdict(lambda: defaultdict(list))
    storage, order = {}, []
    started = time.perf_counter()
    with (output / "samples.jsonl").open("x", encoding="utf-8") as raw:
        # The seed depends on quarter, not configuration: paired payload order.
        rng = random.Random(order_seed)
        for iteration in range(-warmups, iterations):
            shuffled = list(entries)
            rng.shuffle(shuffled)
            for item in shuffled:
                choice = assignment(item, config)
                recipient, signer = keys[choice["kem"]], signers[choice["sig"]]
                start = time.perf_counter()
                rid, put = service.put_record(conn, ctx=ctx, plaintext=payloads[item["filename"]], aad=b"faers-pilot", scheme="hybrid", recipient_keys=recipient, pq_kem_id=choice["kem"], sig_keys=signer)
                get_start = time.perf_counter()
                plaintext, get = service.get_record(conn, tenant_id=ctx.tenant_id, record_id=rid, recipient_keys=recipient, pq_sk=recipient.pq_sk,
                                                     sig_verify_key=signer.verify_key, require_signature=True, expected_sig_id=choice["sig"])
                end = time.perf_counter()
                check(plaintext == payloads[item["filename"]], "Roundtrip plaintext mismatch")
                if iteration < 0:
                    continue
                values = {"encrypt": put["encrypt"], "sign": put["sign"], "db_insert": put["db_insert"], "put_total": put["total"],
                          "db_fetch": get["db_fetch"], "verify": get["verify"], "decrypt": get["decrypt"],
                          "get_total": (end - get_start) * 1000, "roundtrip": (end - start) * 1000}
                check(all(math.isfinite(v) and v >= 0 for v in values.values()), "Invalid timing")
                size_bin = sum(item["payload_bytes"] > t * 1024 for t in THRESHOLDS)
                for label in ("overall", "flagged" if item["elevation_flag"] else "unflagged", choice["tier"], "size_bin_" + str(size_bin)):
                    for stage, value in values.items():
                        groups[label][stage].append(value)
                record = {**item, **config, **choice, "run_id": run_id, "manifest_sha256": manifest_hash,
                          "measurement_index": len(order), "iteration": iteration, "latency_ms": values,
                          "envelope": put["envelope"]}
                raw.write(json.dumps(record, sort_keys=True) + "\n")
                order.append((iteration, item["filename"]))
                if item["filename"] in storage:
                    check(storage[item["filename"]] == put["envelope"], "Envelope size changed across repeated writes")
                storage[item["filename"]] = put["envelope"]
    check(len(order) == len(entries) * iterations, "Incomplete measured operation count")
    strong = [item for item in entries if assignment(item, config)["tier"] == "strong"]
    flagged = [item for item in entries if item["elevation_flag"]]
    plaintext_bytes = sum(item["payload_bytes"] for item in entries)
    stored_bytes = sum(env["total_stored_bytes"] for env in storage.values())
    flagged_strong = sum(item["elevation_flag"] for item in strong)
    coverage = dict(unique_documents=len(entries), strong_documents=len(strong), strong_record_fraction=len(strong)/len(entries),
                    strong_plaintext_bytes=sum(item["payload_bytes"] for item in strong), flagged_documents=len(flagged),
                    flagged_strong_documents=flagged_strong, flagged_strong_fraction=flagged_strong / len(flagged) if flagged else None,
                    elevation_reasons={reason: sum(assignment(item, config)["reason"] == reason for item in entries) for reason in ("uniform", "size_only", "flag_only", "both", "neither")})
    coverage["strong_byte_fraction"] = coverage["strong_plaintext_bytes"] / plaintext_bytes
    logical = dict(plaintext_bytes=plaintext_bytes, total_stored_bytes=stored_bytes,
                   overhead_bytes=stored_bytes-plaintext_bytes, amplification=stored_bytes/plaintext_bytes,
                   definition="sum of envelope byte fields once per unique document; excludes DB pages, indexes, audit, labels and key material")
    report = dict(schema_version=1, run_id=run_id, quarter=directory.name, config=config, manifest_sha256=manifest_hash,
                  measured_rounds=len(order), warmup_rounds=len(entries)*warmups, iterations=iterations, warmups=warmups,
                  latency_ms={group: {stage: summarize(values) for stage,values in stages.items()} for group,stages in groups.items()},
                  coverage=coverage, logical_storage=logical, elapsed_seconds=time.perf_counter()-started,
                  payload_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    check(args.iterations > 0 and args.warmups >= 0, "Invalid repetition counts")
    args.output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    status = dict(status="running", run_id=run_id, reports=[])
    environment = dict(python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
                       dependencies={name: importlib.metadata.version(name) for name in ("cryptography", "psycopg", "liboqs-python")},
                       scheduler={key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_NODELIST")},
                       source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    admin = psycopg.connect(dbname="postgres", autocommit=True)
    owned_database = None
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier("pilot_test_" + run_id)))
        owned_database = "pilot_test_" + run_id
        with psycopg.connect(dbname=owned_database) as conn:
            security_selftest(conn)
        admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(owned_database)))
        owned_database = None
        status["security_selftest"] = "passed"
        if not args.self_test_only:
            datasets = {}
            for index, quarter in enumerate(("2025Q4", "2026Q1")):
                directory = args.datasets / quarter
                entries, manifest_hash = load_dataset(directory)
                selected = diagnostic_sample(entries, args.sample_size, args.seed + index)
                datasets[quarter] = (directory, selected, manifest_hash, args.seed + 1000 * index)
                (args.output / (quarter + ".selected.jsonl")).write_text("".join(json.dumps(item, sort_keys=True)+"\n" for item in selected))
            schedule = [(quarter, config) for quarter in datasets for config in configurations()]
            random.Random(args.seed).shuffle(schedule)
            (args.output / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
            order_hashes = {}
            for index, (quarter, config) in enumerate(schedule):
                directory, entries, manifest_hash, order_seed = datasets[quarter]
                name = f"{index:02d}-{quarter}-{config['mode']}-{config['threshold']}"
                destination = args.output / name
                destination.mkdir()
                owned_database = f"pilot_{run_id}_{index}"
                admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(owned_database)))
                with psycopg.connect(dbname=owned_database) as conn:
                    report = run_configuration(conn, directory, entries, config, destination, run_id, manifest_hash, args.iterations, args.warmups, order_seed)
                if quarter in order_hashes:
                    check(report["payload_order_sha256"] == order_hashes[quarter], "Unpaired payload order")
                order_hashes[quarter] = report["payload_order_sha256"]
                status["reports"].append(dict(path=name, quarter=quarter, config=config, measured_rounds=report["measured_rounds"], elapsed_seconds=report["elapsed_seconds"]))
                (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
                admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(owned_database)))
                owned_database = None
                print(f"PASS {name}", flush=True)
            check(len(status["reports"]) == 24, "Incomplete schedule")
        status["status"] = "passed"
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        if owned_database is not None:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(owned_database)))
        admin.close()

if __name__ == "__main__":
    main()
