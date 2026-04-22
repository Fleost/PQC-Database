#!/usr/bin/env python3
"""
Summarize real payload datasets in a compact table.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"


def mib(n: int) -> float:
    return n / (1024 * 1024)


def collect_sizes(pattern: str, base: Path) -> list[int]:
    return [p.stat().st_size for p in base.glob(pattern)]


def row(label: str, sizes: list[int]) -> list[str]:
    if not sizes:
        return [label, "0", "-", "-", "-"]
    return [
        label,
        str(len(sizes)),
        f"{mib(int(statistics.median(sizes))):.3f} MiB",
        f"{mib(max(sizes)):.3f} MiB",
        f"{mib(sum(sizes)):.1f} MiB",
    ]


def format_table(rows: list[list[str]]) -> str:
    headers = ["Dataset", "File Count", "Median", "Max", "Total"]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]

    def fmt(parts: list[str]) -> str:
        return " | ".join(part.ljust(widths[i]) for i, part in enumerate(parts))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt(headers), sep]
    lines.extend(fmt(r) for r in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real payload sizes.")
    parser.parse_args()

    rows = [
        row("ALL_GGVP", collect_sizes("ALL_GGVP*.vcf.gz", DATA_DIR)),
        row("FAERS", collect_sizes("*.json", DATA_DIR / "faers_records")),
    ]
    print(format_table(rows))


if __name__ == "__main__":
    main()
