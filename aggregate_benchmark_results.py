#!/usr/bin/env python3
"""
Aggregate raw benchmark JSON into size-binned trend summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def percentile(values: List[float], p: float) -> float:
    s = sorted(values)
    if not s:
        raise ValueError("percentile() requires at least one value")
    if len(s) == 1:
        return s[0]
    idx = (p / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


def summarize(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        raise ValueError("summarize() requires at least one value")
    return {
        "min": min(vals),
        "p25": percentile(vals, 25),
        "p50": percentile(vals, 50),
        "p75": percentile(vals, 75),
        "p95": percentile(vals, 95),
        "max": max(vals),
        "mean": statistics.mean(vals),
    }


def mib(n: float) -> float:
    return n / (1024 * 1024)


def load_pairs(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with path.open() as f:
        raw = json.load(f)

    paired: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in raw["results"]:
        key = (row["payload_name"], row["size_bytes"])
        entry = paired.setdefault(
            key,
            {
                "payload_name": row["payload_name"],
                "size_bytes": row["size_bytes"],
                "classical": None,
                "hybrid": None,
            },
        )
        entry[row["scheme"]] = row

    complete = [v for v in paired.values() if v["classical"] and v["hybrid"]]
    complete.sort(key=lambda x: (x["size_bytes"], x["payload_name"]))
    return raw, complete


def build_bin(rows: List[Dict[str, Any]], bin_index: int, total_bins: int) -> Dict[str, Any]:
    sizes = [r["size_bytes"] for r in rows]

    def metric_values(scheme: str, metric: str, field: str = "mean_ms") -> List[float]:
        return [r[scheme][metric][field] for r in rows]

    def envelope_values(scheme: str, field: str) -> List[float]:
        return [r[scheme]["envelope_bytes"][field] for r in rows]

    def pct_delta(metric: str) -> List[float]:
        vals = []
        for r in rows:
            cl = r["classical"][metric]["mean_ms"]
            hy = r["hybrid"][metric]["mean_ms"]
            vals.append(((hy - cl) / cl * 100.0) if cl else 0.0)
        return vals

    return {
        "bin_index": bin_index,
        "bin_count": total_bins,
        "payload_count": len(rows),
        "size_bytes": summarize(sizes),
        "size_mib": summarize([mib(v) for v in sizes]),
        "classical": {
            "encrypt_ms": summarize(metric_values("classical", "encrypt")),
            "db_insert_ms": summarize(metric_values("classical", "db_insert")),
            "db_fetch_ms": summarize(metric_values("classical", "db_fetch")),
            "decrypt_ms": summarize(metric_values("classical", "decrypt")),
            "end_to_end_ms": summarize(metric_values("classical", "end_to_end")),
            "amplification": summarize(envelope_values("classical", "amplification")),
            "overhead_bytes": summarize(envelope_values("classical", "overhead_bytes")),
        },
        "hybrid": {
            "encrypt_ms": summarize(metric_values("hybrid", "encrypt")),
            "db_insert_ms": summarize(metric_values("hybrid", "db_insert")),
            "db_fetch_ms": summarize(metric_values("hybrid", "db_fetch")),
            "decrypt_ms": summarize(metric_values("hybrid", "decrypt")),
            "end_to_end_ms": summarize(metric_values("hybrid", "end_to_end")),
            "amplification": summarize(envelope_values("hybrid", "amplification")),
            "overhead_bytes": summarize(envelope_values("hybrid", "overhead_bytes")),
        },
        "hybrid_overhead_pct": {
            "encrypt_ms": summarize(pct_delta("encrypt")),
            "db_insert_ms": summarize(pct_delta("db_insert")),
            "db_fetch_ms": summarize(pct_delta("db_fetch")),
            "decrypt_ms": summarize(pct_delta("decrypt")),
            "end_to_end_ms": summarize(pct_delta("end_to_end")),
        },
    }


def make_quantile_bins(rows: List[Dict[str, Any]], bin_count: int) -> List[Dict[str, Any]]:
    if not rows:
        return []
    bin_count = max(1, min(bin_count, len(rows)))
    bins: List[Dict[str, Any]] = []
    for i in range(bin_count):
        start = round(i * len(rows) / bin_count)
        end = round((i + 1) * len(rows) / bin_count)
        chunk = rows[start:end]
        if chunk:
            bins.append(build_bin(chunk, i + 1, bin_count))
    return bins


def overall_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sizes = [r["size_bytes"] for r in rows]
    return {
        "payload_count": len(rows),
        "size_bytes": summarize(sizes),
        "size_mib": summarize([mib(v) for v in sizes]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark result JSON into size-binned summaries.")
    parser.add_argument("--input", required=True, help="Path to raw benchmark JSON.")
    parser.add_argument("--output", required=True, help="Path to write aggregated JSON.")
    parser.add_argument("--bins", type=int, default=32, help="Number of size-quantile bins (default: 32).")
    args = parser.parse_args()

    source_meta, rows = load_pairs(Path(args.input))
    aggregated = {
        "source": {
            "run_at": source_meta["run_at"],
            "host": source_meta["host"],
            "payload_source": source_meta["payload_source"],
            "payload_selection": source_meta["payload_selection"],
            "iterations": source_meta["iterations"],
            "warmup_iterations": source_meta["warmup_iterations"],
            "pq_kem_id": source_meta["pq_kem_id"],
        },
        "overall": overall_summary(rows),
        "quantile_bins": make_quantile_bins(rows, args.bins),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"paired_payloads={aggregated['overall']['payload_count']}")
    print(f"bins={len(aggregated['quantile_bins'])}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
