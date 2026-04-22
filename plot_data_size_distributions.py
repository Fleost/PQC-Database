#!/usr/bin/env python3
"""
Plot size-distribution charts for the real payload datasets used by the project.

Generates charts for:
- ALL_GGVP chromosome VCF payloads
- FAERS record JSON payloads
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
RESULTS_DIR = ROOT / "results"
ALL_GGVP_BENCHMARK_RESULTS = RESULTS_DIR / "benchmark_results_all_ggvp_full.json"


def bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)


def bytes_to_decimal_mb(n: int) -> float:
    return n / 1_000_000


def load_sizes() -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    all_ggvp = sorted(
        ((p.name, p.stat().st_size) for p in DATA_DIR.glob("ALL_GGVP*.vcf.gz")),
        key=lambda x: x[1],
        reverse=True,
    )
    faers = sorted(
        ((p.name, p.stat().st_size) for p in (DATA_DIR / "faers_records").glob("*.json")),
        key=lambda x: x[1],
        reverse=True,
    )
    return all_ggvp, faers


def load_all_ggvp_tested_sizes() -> list[tuple[str, int]]:
    if not ALL_GGVP_BENCHMARK_RESULTS.exists():
        return []

    with ALL_GGVP_BENCHMARK_RESULTS.open() as f:
        data = json.load(f)

    tested: dict[str, int] = {}
    for row in data.get("results", []):
        payload_name = row.get("payload_name")
        size_bytes = row.get("size_bytes")
        if payload_name and isinstance(size_bytes, int):
            tested[payload_name] = size_bytes

    return sorted(tested.items(), key=lambda x: x[1], reverse=True)


def summarize(label: str, items: list[tuple[str, int]]) -> str:
    sizes = [size for _, size in items]
    return (
        f"{label}: n={len(sizes)}, "
        f"min={bytes_to_mb(min(sizes)):.3f} MB, "
        f"median={bytes_to_mb(int(statistics.median(sizes))):.3f} MB, "
        f"max={bytes_to_mb(max(sizes)):.3f} MB, "
        f"total={bytes_to_mb(sum(sizes)):.1f} MB"
    )


def plot_all_ggvp_bar(items: list[tuple[str, int]], output: Path) -> None:
    labels = [name.split(".")[1].removeprefix("chr") for name, _ in items]
    sizes_mb = [bytes_to_decimal_mb(size) for _, size in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, sizes_mb, color="#2E5EAA", edgecolor="#1A335C", linewidth=0.8)

    ax.set_title("ALL_GGVP Tested Payload Sizes by Chromosome")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Tested Payload Size (MB)")
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, sizes_mb):
        label = f"{val:.2f}" if val < 10 else f"{val:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes_mb) * 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_faers_histogram(items: list[tuple[str, int]], output: Path) -> None:
    sizes_kb = np.array([size / 1024 for _, size in items], dtype=float)
    clipped = np.clip(sizes_kb, 0.5, None)
    bins = np.logspace(math.log10(clipped.min()), math.log10(clipped.max()), 50)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(clipped, bins=bins, color="#C96B28", edgecolor="#8B4513", alpha=0.9)
    ax.set_xscale("log")
    ax.set_title("FAERS Record File Size Distribution")
    ax.set_xlabel("JSON File Size (KB)")
    ax.set_ylabel("Record Count")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:g}")
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis="y", alpha=0.25)

    median_kb = float(np.median(clipped))
    p95_kb = float(np.percentile(clipped, 95))
    ax.axvline(median_kb, color="#1F1F1F", linestyle="--", linewidth=1.4, label=f"median={median_kb:.1f} KB")
    ax.axvline(p95_kb, color="#7A0F0F", linestyle=":", linewidth=1.8, label=f"p95={p95_kb:.1f} KB")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_dataset_comparison(
    all_ggvp: list[tuple[str, int]], faers: list[tuple[str, int]], output: Path
) -> None:
    g_sizes_mb = np.array([bytes_to_mb(size) for _, size in all_ggvp], dtype=float)
    f_sizes_mb = np.array([bytes_to_mb(size) for _, size in faers], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    parts = ax.violinplot([g_sizes_mb, f_sizes_mb], showmeans=True, showmedians=True, widths=0.85)

    colors = ["#2E5EAA", "#C96B28"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.5)

    for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1.0)

    ax.set_xticks([1, 2], ["ALL_GGVP", "FAERS"])
    ax.set_ylabel("File Size (MB)")
    ax.set_title("Payload Size Comparison Across Real Datasets")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot size distributions for ALL_GGVP and FAERS payload files.")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR), help="Directory to write chart PNGs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ggvp, faers = load_sizes()
    all_ggvp_tested = load_all_ggvp_tested_sizes() or all_ggvp
    if not all_ggvp:
        raise SystemExit("No ALL_GGVP VCF files found in Data/")
    if not faers:
        raise SystemExit("No FAERS JSON files found in Data/faers_records/")

    print(summarize("ALL_GGVP", all_ggvp))
    print(summarize("FAERS", faers))

    plot_all_ggvp_bar(all_ggvp_tested, output_dir / "09_all_ggvp_file_sizes.png")
    plot_faers_histogram(faers, output_dir / "10_faers_file_size_distribution.png")
    plot_dataset_comparison(all_ggvp, faers, output_dir / "11_payload_size_comparison.png")


if __name__ == "__main__":
    main()
