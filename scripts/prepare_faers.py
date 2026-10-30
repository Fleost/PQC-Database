#!/usr/bin/env python3
"""Reconstruct source-faithful JSON from the seven FAERS ASCII tables.

Run once per quarter; never combine quarters implicitly. No third-party modules.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

SECTIONS = dict(DEMO="demographics", DRUG="drugs", REAC="reactions",
                INDI="indications", OUTC="outcomes", RPSR="report_sources", THER="therapies")

def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def source_files(root, table, quarter):
    suffix = quarter[2:]
    stem = table + suffix
    matches = [p for p in root.rglob("*.txt") if p.stem.upper() == stem]
    parts = sorted(p for p in root.rglob("*.txt") if re.fullmatch(stem + r"_PART\d+", p.stem.upper()))
    if matches and parts:
        raise ValueError(f"Ambiguous original and split files for {stem}; provide only one representation")
    if len(matches) > 1:
        raise ValueError(f"Duplicate source table {stem}")
    if parts:
        numbers = [int(re.search(r"(\d+)$", p.stem).group(1)) for p in parts]
        if numbers != list(range(1, len(parts) + 1)):
            raise ValueError(f"Missing or misnumbered chunks for {stem}")
    if not matches and not parts:
        raise ValueError(f"Missing {stem}.txt")
    return matches or parts

def rows(files):
    expected = None
    for path in files:
        with path.open(encoding="utf-8-sig", errors="strict", newline="") as handle:
            reader = csv.reader(handle, delimiter="$", quoting=csv.QUOTE_NONE)
            header = next(reader)
            if len(set(header)) != len(header):
                raise ValueError(f"Duplicate column names in {path}")
            if expected is not None and expected != header:
                raise ValueError(f"Inconsistent chunk headers in {path}")
            expected = header
            for number, values in enumerate(reader, 2):
                if not values:
                    continue
                if len(values) != len(header):
                    raise ValueError(f"{path}:{number}: expected {len(header)} columns; got {len(values)}")
                yield dict(zip(header, values))

def field(row, key):
    matches = [value for name, value in row.items() if name.lower() == key]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous {key} field")
    return matches[0]

def prepare(root, output, quarter, sample_size=50000, seed=42):
    if not re.fullmatch(r"20\d{2}Q[1-4]", quarter):
        raise ValueError("quarter must look like 2025Q4")
    if output.exists():
        raise FileExistsError(output)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    inputs = {table: source_files(root, table, quarter) for table in SECTIONS}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".prepare-", dir=output.parent))
    try:
        staging = temp / "staging.sqlite"
        con = sqlite3.connect(staging)
        con.execute("CREATE TABLE demo (caseid TEXT PRIMARY KEY, version INTEGER, primaryid TEXT UNIQUE, data TEXT)")
        demo_source_rows = 0
        for row in rows(inputs["DEMO"]):
            demo_source_rows += 1
            pid, case, version = field(row, "primaryid"), field(row, "caseid"), field(row, "caseversion")
            if not pid.isdigit() or not case.isdigit() or not version.isdigit():
                raise ValueError("Non-numeric or blank DEMO identifier/version")
            old = con.execute("SELECT version,primaryid FROM demo WHERE caseid=?", (case,)).fetchone()
            if old and old[0] == int(version) and old[1] != pid:
                raise ValueError(f"Ambiguous latest case version for {case}")
            if old is None or int(version) > old[0]:
                con.execute("INSERT INTO demo VALUES (?,?,?,?) ON CONFLICT(caseid) DO UPDATE SET version=excluded.version,primaryid=excluded.primaryid,data=excluded.data", (case, int(version), pid, compact(row)))
        con.commit()
        eligible = con.execute("SELECT COUNT(*) FROM demo").fetchone()[0]
        if eligible < sample_size:
            raise ValueError(f"Only {eligible} eligible cases; requested {sample_size}")
        # Freeze the exact input order to the seeded sample, independently per quarter.
        ids = [row[0] for row in con.execute("SELECT primaryid FROM demo ORDER BY CAST(primaryid AS INTEGER)")]
        selected = sorted(random.Random(seed).sample(ids, sample_size), key=int)
        documents = {}
        metadata = {}
        for pid in selected:
            case, version, data = con.execute("SELECT caseid,version,data FROM demo WHERE primaryid=?", (pid,)).fetchone()
            documents[pid] = {section: ([] if table != "DEMO" else json.loads(data)) for table, section in SECTIONS.items()}
            metadata[pid] = dict(quarter=quarter, primaryid=pid, caseid=case, caseversion=str(version))
        counts = {"DEMO": demo_source_rows}
        for table, section in SECTIONS.items():
            if table == "DEMO":
                continue
            counts[table] = 0
            for row in rows(inputs[table]):
                counts[table] += 1
                pid = field(row, "primaryid")
                if pid in documents:
                    if field(row, "caseid") != metadata[pid]["caseid"]:
                        raise ValueError(f"Mismatched child caseid: {table} {pid}")
                    documents[pid][section].append(row)
        records = temp / "faers_records"
        meta = temp / "metadata"
        records.mkdir()
        meta.mkdir()
        with (meta / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for pid in selected:
                doc = documents.pop(pid)
                for section, value in doc.items():
                    if isinstance(value, list):
                        value.sort(key=compact)
                codes = {field(row, "outc_cod").upper() for row in doc["outcomes"]}
                payload = compact(doc).encode("utf-8")
                name = f"{pid}.json"
                (records / name).write_bytes(payload)
                entry = dict(metadata[pid], filename=name, payload_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                             reported_death="DE" in codes, reported_life_threatening="LT" in codes,
                             elevation_flag=bool(codes & {"DE", "LT"}), elevation_rule="faers-de-lt-v1")
                handle.write(compact(entry) + "\n")
        provenance = dict(quarter=quarter, sample_size=sample_size, seed=seed, eligible_latest_cases=eligible,
                          selection="seeded random without replacement; latest numeric caseversion within quarter",
                          source_rows=counts, source_files=[dict(name=p.name, sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest()) for files in inputs.values() for p in files],
                          warning="Reported DE/LT outcomes are prioritization metadata, not evidence of drug causation")
        (meta / "preparation.json").write_text(json.dumps(provenance, indent=2) + "\n")
        con.close()
        staging.unlink()
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarter", required=True)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.input, args.output, args.quarter, args.sample_size, args.seed)

if __name__ == "__main__":
    main()
