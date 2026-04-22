#!/usr/bin/env python3
"""
Plot total storage footprint summaries for the real FAERS and GGVP benchmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def summarize_totals(data: dict) -> dict[str, int]:
    seen: set[tuple[str, str]] = set()
    plaintext = 0
    classical = 0
    hybrid = 0

    for row in data["results"]:
        key = (row["scheme"], row["payload_name"])
        if key in seen:
            continue
        seen.add(key)

        size_bytes = row["size_bytes"]
        total_stored = row["envelope_bytes"]["total_stored_bytes"]
        plaintext += size_bytes if row["scheme"] == "classical" else 0
        if row["scheme"] == "classical":
            classical += total_stored
        elif row["scheme"] == "hybrid":
            hybrid += total_stored

    return {"plaintext": plaintext, "classical": classical, "hybrid": hybrid}


def bytes_to_gib(n: int) -> float:
    return n / (1024 ** 3)


def plot_footprint(faers: dict[str, int], ggvp: dict[str, int], output: Path) -> None:
    combined = {
        key: faers[key] + ggvp[key]
        for key in ("plaintext", "classical", "hybrid")
    }

    groups = ["FAERS", "GGVP", "Combined"]
    series = {
        "Plaintext": [faers["plaintext"], ggvp["plaintext"], combined["plaintext"]],
        "Classical": [faers["classical"], ggvp["classical"], combined["classical"]],
        "Hybrid": [faers["hybrid"], ggvp["hybrid"], combined["hybrid"]],
    }
    colors = {"Plaintext": "#7A7A7A", "Classical": "#2E5EAA", "Hybrid": "#C96B28"}

    x = np.arange(len(groups))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_by_series = {}
    for idx, label in enumerate(["Plaintext", "Classical", "Hybrid"]):
        offsets = x + (idx - 1) * width
        values_gib = [bytes_to_gib(v) for v in series[label]]
        bars_by_series[label] = ax.bar(
            offsets,
            values_gib,
            width,
            label=label,
            color=colors[label],
            edgecolor="#333333",
            linewidth=0.8,
        )

    ax.set_xticks(x, groups)
    ax.set_ylabel("Total Footprint (GiB)")
    ax.set_title("Measured Storage Footprint for Real Payloads")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for label, bars in bars_by_series.items():
        raw_values = series[label]
        for bar, raw in zip(bars, raw_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bytes_to_gib(v) for v in combined.values()) * 0.01,
                f"{raw / 1_000_000_000:.3f} GB",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot measured storage footprint totals.")
    parser.add_argument("--faers-input", default=str(RESULTS_DIR / "benchmark_results_faers_full.json"))
    parser.add_argument("--ggvp-input", default=str(RESULTS_DIR / "benchmark_results_all_ggvp_full.json"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "16_storage_footprint_totals.png"))
    args = parser.parse_args()

    faers = summarize_totals(load(Path(args.faers_input)))
    ggvp = summarize_totals(load(Path(args.ggvp_input)))
    plot_footprint(faers, ggvp, Path(args.output))


if __name__ == "__main__":
    main()
