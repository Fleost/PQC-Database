#!/usr/bin/env python3
"""
Generate a storage amplification comparison chart across ML-DSA variants (and
unsigned baseline) for the FAERS dataset.  Mirrors the style of the latency
comparison but shows stored_bytes / plaintext_bytes vs payload size.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

RESULTS_DIR = Path(__file__).parent / "results"

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
        return json.load(f)["results"]


def bin_results(results: list[dict], n_bins: int) -> list[tuple[float, float]]:
    """Return list of (median_size_mib, median_amplification) per bin."""
    results_sorted = sorted(results, key=lambda r: r["size_bytes"])
    total = len(results_sorted)
    bin_size = max(1, total // n_bins)

    out: list[tuple[float, float]] = []
    for i in range(0, total, bin_size):
        chunk = results_sorted[i : i + bin_size]
        if not chunk:
            continue
        sizes_mib = sorted(r["size_bytes"] / (1024 * 1024) for r in chunk)
        amps = sorted(r["envelope_bytes"]["amplification"] for r in chunk)
        n = len(amps)
        out.append((sizes_mib[n // 2], amps[n // 2]))

    return out


def _size_formatter(x: float, _pos: int) -> str:
    if x <= 0:
        return ""
    if x < 1:
        kib = x * 1024
        return f"{kib:.0f} KB" if kib >= 10 else f"{kib:.1f} KB"
    return f"{x:.0f} MB" if x >= 10 else f"{x:.1f} MB"


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


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, path in DSA_FILES.items():
        results = load(path)
        bins = bin_results(results, NUM_BINS)
        xs = np.array([b[0] for b in bins])
        amps = np.array([b[1] for b in bins])
        ax.plot(xs, amps, color=COLORS[label], linewidth=2.2, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Payload Size")
    ax.set_ylabel("Storage Amplification (stored bytes / plaintext bytes)")
    ax.set_title("FAERS Storage Amplification: DSA Variant Comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    apply_size_axis(ax)
    fig.tight_layout()

    output = RESULTS_DIR / "21_faers_dsa_storage_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
