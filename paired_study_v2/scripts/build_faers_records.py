#!/usr/bin/env python3
"""
scripts/build_faers_records.py
==============================
Build one JSON document per FAERS safety report from the FDA's quarterly
ASCII extracts, then write a reproducible random sample of them to
``Data/faers_records/`` for the benchmark (``--payload-source faers``).

Pipeline
--------
1. Read DEMO for every quarter given.  Drop any case listed in a quarter's
   DELETE file.  Deduplicate by ``caseid``, keeping the most recent version
   (largest ``fda_dt``, then largest ``primaryid``), as the FDA's FAERS
   README recommends.
2. Draw a seeded random sample of ``--count`` reports from the deduplicated
   set.
3. Stream DRUG, THER, INDI, REAC, OUTC and RPSR and attach each row to its
   sampled report.  Therapy dates (THER) and indications (INDI) are nested
   under the drug they refer to (matched on drug sequence number).
4. Write ``<caseid>_<primaryid>.json`` per report plus ``manifest.csv``.

The ASCII files are '$'-delimited with a header row.  They are decoded as
latin-1, so every byte maps to a character and no row is dropped for
encoding reasons.

Usage
-----
    python scripts/build_faers_records.py \
        --quarter ../Q42025 --quarter ../Q12026 \
        --out Data/faers_records --count 50000 --seed 20260101
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.elevation import RULES, flags_from_doc  # noqa: E402  (dependency-free module)

TABLES = ["DEMO", "DRUG", "THER", "INDI", "REAC", "OUTC", "RPSR"]


def _find_table(quarter_dir: Path, table: str) -> Path:
    hits = sorted((quarter_dir / "ASCII").glob(f"{table}[0-9][0-9]Q[1-4].txt"))
    if not hits:
        raise FileNotFoundError(f"No {table}yyQn.txt under {quarter_dir / 'ASCII'}")
    return hits[0]


def _quarter_label(quarter_dir: Path) -> str:
    demo = _find_table(quarter_dir, "DEMO")
    m = re.match(r"DEMO(\d\dQ\d)", demo.name)
    return m.group(1) if m else quarter_dir.name


def _read_rows(path: Path) -> Iterator[Dict[str, str]]:
    """Yield each data row as a dict keyed by lower-case header name."""
    with path.open("rb") as fh:
        header = fh.readline().decode("latin-1").rstrip("\r\n").split("$")
        header = [h.strip().lower() for h in header]
        n = len(header)
        for raw in fh:
            line = raw.decode("latin-1").rstrip("\r\n")
            if not line:
                continue
            parts = line.split("$")
            if len(parts) < n:
                parts += [""] * (n - len(parts))
            elif len(parts) > n:
                # Extra '$' inside the last free-text column: keep it together.
                parts = parts[: n - 1] + ["$".join(parts[n - 1 :])]
            yield dict(zip(header, parts))


def _read_deleted(quarter_dir: Path) -> set:
    out = set()
    for p in (quarter_dir / "Deleted").glob("*.txt"):
        for line in p.read_text(encoding="latin-1").splitlines():
            line = line.strip()
            if line.isdigit():
                out.add(line)
    return out


def _clean(row: Dict[str, str], drop=("primaryid", "caseid")) -> Dict[str, str]:
    return {k: v for k, v in row.items() if k not in drop and v != ""}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quarter", action="append", required=True, type=Path,
                    help="Quarter directory containing ASCII/ and Deleted/ (repeatable, oldest first)")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for JSON records")
    ap.add_argument("--count", type=int, default=50000, help="Reports to sample (default 50000; 0 = all)")
    ap.add_argument("--per-quarter", type=int, default=0,
                    help="Stratified sample: N reports from each quarter (overrides --count). A case is "
                         "assigned to the quarter of its most recent version, so no case appears twice.")
    ap.add_argument("--seed", type=int, default=20260101, help="Sampling seed")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent (default 2; -1 = compact)")
    args = ap.parse_args(argv)

    quarters = [(q, _quarter_label(q)) for q in args.quarter]

    # ── 1. DEMO + deletions + dedup ──────────────────────────────────────────
    deleted: set = set()
    for q, _ in quarters:
        deleted |= _read_deleted(q)
    print(f"deleted caseids listed: {len(deleted):,}")

    best: Dict[str, Tuple[Tuple[str, int], str, Dict[str, str]]] = {}
    demo_rows = 0
    for q, label in quarters:
        for row in _read_rows(_find_table(q, "DEMO")):
            demo_rows += 1
            cid = row["caseid"]
            if cid in deleted:
                continue
            key = (row.get("fda_dt", ""), int(row["primaryid"] or 0))
            cur = best.get(cid)
            if cur is None or key > cur[0]:
                best[cid] = (key, label, row)
        print(f"  {label}: DEMO read (cumulative rows {demo_rows:,})")
    print(f"unique cases after dedup/deletes: {len(best):,}")

    # ── 2. Sample ────────────────────────────────────────────────────────────
    caseids = sorted(best)
    rng = random.Random(args.seed)
    if args.per_quarter:
        picked = []
        for _, label in quarters:
            pool = [c for c in caseids if best[c][1] == label]
            if len(pool) < args.per_quarter:
                raise SystemExit(f"{label}: only {len(pool):,} cases, cannot sample {args.per_quarter:,}")
            print(f"  {label}: pool {len(pool):,} cases -> sampling {args.per_quarter:,}")
            picked += rng.sample(pool, args.per_quarter)
        caseids = sorted(picked)
    elif args.count and args.count < len(caseids):
        caseids = sorted(rng.sample(caseids, args.count))
    selected = {best[c][2]["primaryid"]: c for c in caseids}
    print(f"sampled reports: {len(selected):,} (seed {args.seed})")

    # primaryid → (quarter label) so we only attach rows from the chosen version's quarter
    pid_quarter = {best[c][2]["primaryid"]: best[c][1] for c in caseids}

    # ── 3. Child tables ──────────────────────────────────────────────────────
    drugs: Dict[str, Dict[str, dict]] = defaultdict(dict)
    ther: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    indi: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    reac: Dict[str, list] = defaultdict(list)
    outc: Dict[str, list] = defaultdict(list)
    rpsr: Dict[str, list] = defaultdict(list)

    for q, label in quarters:
        for table in TABLES[1:]:
            path = _find_table(q, table)
            n = 0
            for row in _read_rows(path):
                pid = row["primaryid"]
                if pid not in selected or pid_quarter[pid] != label:
                    continue
                n += 1
                if table == "DRUG":
                    drugs[pid][row["drug_seq"]] = _clean(row)
                elif table == "THER":
                    ther[pid][row["dsg_drug_seq"]].append(_clean(row, ("primaryid", "caseid", "dsg_drug_seq")))
                elif table == "INDI":
                    indi[pid][row["indi_drug_seq"]].append(row["indi_pt"])
                elif table == "REAC":
                    reac[pid].append(_clean(row))
                elif table == "OUTC":
                    outc[pid].append(row["outc_cod"])
                elif table == "RPSR":
                    rpsr[pid].append(row["rpsr_cod"])
            print(f"  {label}: {table:<4} rows attached {n:,}")

    # ── 4. Write ─────────────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.json"):
        old.unlink()
    indent = None if args.indent < 0 else args.indent
    seps = (",", ":") if indent is None else (",", ": ")
    manifest_rows: List[List] = []
    for pid, cid in sorted(selected.items(), key=lambda kv: kv[1]):
        _, label, demo = best[cid]
        drug_list = []
        for seq in sorted(drugs[pid], key=lambda s: int(s) if s.isdigit() else 0):
            d = dict(drugs[pid][seq])
            if indi[pid].get(seq):
                d["indications"] = indi[pid][seq]
            if ther[pid].get(seq):
                d["therapy"] = ther[pid][seq]
            drug_list.append(d)
        doc = {
            "primaryid": pid,
            "caseid": cid,
            "source_quarter": label,
            "demographics": _clean(demo),
            "drugs": drug_list,
            "reactions": reac[pid],
            "outcomes": outc[pid],
            "report_sources": rpsr[pid],
        }
        name = f"{cid}_{pid}.json"
        data = json.dumps(doc, indent=indent, separators=seps, ensure_ascii=False).encode("utf-8")
        (args.out / name).write_bytes(data)
        fl = flags_from_doc(doc)
        manifest_rows.append([name, cid, pid, label, len(data), len(drug_list), len(reac[pid]),
                              "|".join(sorted(set(outc[pid]))), *(int(fl[r]) for r in RULES)])

    with (args.out.parent / f"{args.out.name}_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "caseid", "primaryid", "source_quarter", "size_bytes", "n_drugs", "n_reactions",
                    "outcomes", *(f"flag_{r}" for r in RULES)])
        w.writerows(manifest_rows)

    # Single-file bundle: the benchmark container reads this in one pass instead
    # of 50k small files over the Docker Desktop bind mount.
    import tarfile
    bundle = args.out.parent / f"{args.out.name}.tar"
    with tarfile.open(bundle, "w") as tf:
        for r in manifest_rows:
            tf.add(args.out / r[0], arcname=f"{args.out.name}/{r[0]}")

    sizes = sorted(r[4] for r in manifest_rows)
    tot = sum(sizes)
    pct = lambda p: sizes[min(len(sizes) - 1, int(p / 100 * len(sizes)))]
    print(f"\nwrote {len(sizes):,} records to {args.out}")
    print(f"  total {tot/1e6:,.1f} MB  min {sizes[0]:,}  p50 {pct(50):,}  p90 {pct(90):,}  "
          f"p99 {pct(99):,}  max {sizes[-1]:,} bytes")
    for t in (1458, 4196, 10240, 25600, 51200, 102400, 256000):
        above = [s for s in sizes if s > t]
        print(f"  > {t:>7,} B: {len(above)/len(sizes)*100:5.1f}% of records, "
              f"{sum(above)/tot*100:5.1f}% of bytes")
    for k, r in enumerate(RULES):
        col = 8 + k
        by_q = {}
        for row in manifest_rows:
            by_q.setdefault(row[3], []).append(row[col])
        print(f"  flag {r:<9}: " + "  ".join(f"{q} {sum(v)/len(v)*100:5.2f}%" for q, v in sorted(by_q.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
