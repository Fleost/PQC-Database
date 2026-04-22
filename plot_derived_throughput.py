#!/usr/bin/env python3
"""
Plot derived end-to-end throughput from existing benchmark results.

Throughput here is computed as:
    payload_size_bytes / end_to_end_latency_seconds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

CLASSICAL_COLOR = "#2E5EAA"
HYBRID_COLOR = "#C96B28"


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def fmt_payload_size_mib(mib: float) -> str:
    if mib < 1:
        kb = mib * 1024
        return f"{kb:.0f} KB" if kb >= 10 else f"{kb:.1f} KB"
    return f"{mib:.0f} MB" if mib >= 10 else f"{mib:.1f} MB"


def fmt_throughput_bps(bps: float) -> str:
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.1f} MB/s"
    if bps >= 1_000:
        return f"{bps / 1_000:.1f} KB/s"
    return f"{bps:.0f} B/s"


def aggregate_series(data: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bins = data["quantile_bins"]
    x = np.array([b["size_mib"]["p50"] for b in bins], dtype=float)
    out = {}
    for scheme in ("classical", "hybrid"):
        size_bytes = np.array([b["size_bytes"]["p50"] for b in bins], dtype=float)
        p50_ms = np.array([b[scheme]["end_to_end_ms"]["p50"] for b in bins], dtype=float)
        p95_ms = np.array([b[scheme]["end_to_end_ms"]["p95"] for b in bins], dtype=float)
        out[f"{scheme}_p50"] = np.divide(size_bytes * 1000.0, p50_ms, out=np.zeros_like(size_bytes), where=p50_ms > 0)
        out[f"{scheme}_p95"] = np.divide(size_bytes * 1000.0, p95_ms, out=np.zeros_like(size_bytes), where=p95_ms > 0)
    return x, out


def exact_series(data: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by_scheme: dict[str, list[tuple[float, float]]] = {"classical": [], "hybrid": []}
    for row in data["results"]:
        scheme = row["scheme"]
        size_mib = row["size_bytes"] / (1024 * 1024)
        p50_ms = row["end_to_end"]["p50_ms"]
        if p50_ms <= 0:
            continue
        bps = row["size_bytes"] * 1000.0 / p50_ms
        by_scheme[scheme].append((size_mib, bps))

    out = {}
    for scheme, points in by_scheme.items():
        points.sort(key=lambda item: item[0])
        out[scheme] = (
            np.array([x for x, _ in points], dtype=float),
            np.array([y for _, y in points], dtype=float),
        )
    return out


def plot(faers: dict, ggvp: dict, output: Path) -> None:
    x, faers_series = aggregate_series(faers)
    ggvp_series = exact_series(ggvp)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, faers_series["classical_p50"], color=CLASSICAL_COLOR, linewidth=2.2, marker="o", markersize=4, label="FAERS classical")
    ax.plot(x, faers_series["hybrid_p50"], color=HYBRID_COLOR, linewidth=2.2, marker="s", markersize=4, label="FAERS hybrid")
    ax.fill_between(x, faers_series["classical_p95"], faers_series["classical_p50"], color=CLASSICAL_COLOR, alpha=0.14, label="_nolegend_")
    ax.fill_between(x, faers_series["hybrid_p95"], faers_series["hybrid_p50"], color=HYBRID_COLOR, alpha=0.14, label="_nolegend_")

    gx, gy = ggvp_series["classical"]
    ax.scatter(gx, gy, marker="o", s=32, facecolors="white", edgecolors=CLASSICAL_COLOR, linewidths=1.4, zorder=6, label="GGVP classical")
    gx, gy = ggvp_series["hybrid"]
    ax.scatter(gx, gy, marker="s", s=32, facecolors="white", edgecolors=HYBRID_COLOR, linewidths=1.4, zorder=6, label="GGVP hybrid")

    ax.set_xscale("log")
    ax.set_xlabel("Payload Size")
    ax.set_ylabel("Derived Throughput")
    ax.set_title("End-to-End Throughput: Classical vs Hybrid")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_payload_size_mib(v) if v > 0 else ""))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ymax = max(
        np.max(faers_series["classical_p50"]),
        np.max(faers_series["hybrid_p50"]),
        np.max(ggvp_series["classical"][1]),
        np.max(ggvp_series["hybrid"][1]),
    )
    tick_step = 10_000_000
    upper = tick_step * int(np.ceil(ymax / tick_step))
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1_000_000:.0f} MB/s" if v >= 0 else ""))
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot derived end-to-end throughput from benchmark results.")
    parser.add_argument("--faers-input", default=str(RESULTS_DIR / "benchmark_results_faers_full_aggregated.json"))
    parser.add_argument("--ggvp-input", default=str(RESULTS_DIR / "benchmark_results_all_ggvp_full.json"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "17_derived_throughput.png"))
    args = parser.parse_args()

    plot(load(Path(args.faers_input)), load(Path(args.ggvp_input)), Path(args.output))


if __name__ == "__main__":
    main()
