#!/usr/bin/env python3
"""
Generate a latency comparison chart across ML-DSA variants (and unsigned baseline)
for the FAERS dataset.  Mirrors the style of 18_faers_latency_comparison.png but
compares DSA versions instead of KEM variants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

RESULTS_DIR = Path(__file__).parent / "results"

# Files and their display labels
DSA_FILES = {
    "Unsigned (no PQ sig)": RESULTS_DIR / "bench_unsigned_kem768_faers.json",
    "ML-DSA-44": RESULTS_DIR / "bench_mldsa44_kem768_faers.json",
    "ML-DSA-65": RESULTS_DIR / "bench_mldsa65_kem768_faers.json",
    "ML-DSA-87": RESULTS_DIR / "bench_mldsa87_kem768_faers.json",
}

COLORS = {
    "Unsigned (no PQ sig)": "#2E5EAA",
    "ML-DSA-44": "#3C7A3B",
    "ML-DSA-65": "#C96B28",
    "ML-DSA-87": "#9C2C2C",
}

NUM_BINS = 30


def load(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    return data["results"]


def bin_results(
    results: list[dict], n_bins: int
) -> list[tuple[float, float, float]]:
    """Return list of (median_size_mib, p50_latency_ms, p95_latency_ms) per bin."""
    results_sorted = sorted(results, key=lambda r: r["size_bytes"])
    total = len(results_sorted)
    bin_size = max(1, total // n_bins)

    out: list[tuple[float, float, float]] = []
    for i in range(0, total, bin_size):
        chunk = results_sorted[i : i + bin_size]
        if not chunk:
            continue
        sizes_mib = [r["size_bytes"] / (1024 * 1024) for r in chunk]
        latencies = [r["end_to_end"]["p50_ms"] for r in chunk]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        p50 = latencies_sorted[n // 2]
        p95 = latencies_sorted[min(int(n * 0.95), n - 1)]
        median_size = sorted(sizes_mib)[len(sizes_mib) // 2]
        out.append((median_size, p50, p95))

    return out


def _size_formatter(x: float, _pos: int) -> str:
    if x <= 0:
        return ""
    if x < 1:
        kib = x * 1024
        return f"{kib:.0f} KB" if kib >= 10 else f"{kib:.1f} KB"
    return f"{x:.0f} MB" if x >= 10 else f"{x:.1f} MB"


def _latency_formatter(y: float, _pos: int) -> str:
    if y <= 0:
        return ""
    return f"{y:.0f} ms" if y >= 10 else f"{y:.1f} ms"


def _log_ticks(
    min_val: float, max_val: float, *, per_decade: tuple[float, ...]
) -> list[float]:
    if min_val <= 0 or max_val <= 0:
        return []
    lo = math.floor(math.log10(min_val))
    hi = math.ceil(math.log10(max_val))
    ticks: list[float] = []
    for exp in range(lo, hi + 1):
        base = 10 ** exp
        for mult in per_decade:
            tick = mult * base
            if min_val * 0.999 <= tick <= max_val * 1.001:
                ticks.append(tick)
    return sorted(set(ticks))


def apply_size_axis(ax) -> None:
    xmin, xmax = ax.get_xlim()
    ticks = _log_ticks(xmin, xmax, per_decade=(1, 2, 5))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_size_formatter))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def apply_latency_axis(ax) -> None:
    ymin, ymax = ax.get_ylim()
    ticks = _log_ticks(ymin, ymax, per_decade=(1, 2, 5))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_latency_formatter))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, path in DSA_FILES.items():
        results = load(path)
        bins = bin_results(results, NUM_BINS)
        xs = np.array([b[0] for b in bins])
        p50s = np.array([b[1] for b in bins])
        p95s = np.array([b[2] for b in bins])
        color = COLORS[label]
        ax.plot(xs, p50s, color=color, linewidth=2.2, label=f"{label} (p50)")
        ax.fill_between(xs, p50s, p95s, color=color, alpha=0.18)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Payload Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("FAERS End-to-End Latency: DSA Variant Comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    apply_size_axis(ax)
    apply_latency_axis(ax)
    fig.tight_layout()

    output = RESULTS_DIR / "20_faers_dsa_latency_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
