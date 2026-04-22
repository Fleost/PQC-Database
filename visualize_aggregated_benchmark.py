#!/usr/bin/env python3
"""
Visualize aggregated benchmark trend data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def x_values(bins: list[dict]) -> np.ndarray:
    return np.array([b["size_mib"]["p50"] for b in bins], dtype=float)


def y_series(bins: list[dict], *keys: str) -> np.ndarray:
    cur = []
    for b in bins:
        node = b
        for key in keys:
            node = node[key]
        cur.append(node)
    return np.array(cur, dtype=float)


def _size_tick_formatter(x: float, pos: int) -> str:
    if x <= 0:
        return ""
    if x < 1:
        kib = x * 1024
        return f"{kib:.0f} KB" if kib >= 10 else f"{kib:.1f} KB"
    return f"{x:.0f} MB" if x >= 10 else f"{x:.1f} MB"


def _latency_tick_formatter(y: float, pos: int) -> str:
    if y <= 0:
        return ""
    return f"{y:.0f} ms" if y >= 10 else f"{y:.1f} ms"


def _log_ticks(min_val: float, max_val: float, *, per_decade: tuple[float, ...]) -> list[float]:
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
    ax.xaxis.set_major_formatter(FuncFormatter(_size_tick_formatter))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def apply_sparse_size_axis(ax, ticks: list[float]) -> None:
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_size_tick_formatter))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def apply_latency_axis(ax) -> None:
    ymin, ymax = ax.get_ylim()
    ticks = _log_ticks(ymin, ymax, per_decade=(1, 2, 5))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_latency_tick_formatter))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def plot_latency_trends(data: dict, output: Path) -> None:
    bins = data["quantile_bins"]
    x = x_values(bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    for scheme, color in [("classical", "#2E5EAA"), ("hybrid", "#C96B28")]:
        p50 = y_series(bins, scheme, "end_to_end_ms", "p50")
        p95 = y_series(bins, scheme, "end_to_end_ms", "p95")
        ax.plot(x, p50, color=color, linewidth=2.2, label=f"{scheme} p50")
        ax.fill_between(x, p50, p95, color=color, alpha=0.18, label=f"{scheme} p50-p95")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Payload Size (MiB, median per bin)")
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_title("FAERS Benchmark Trend: End-to-End Latency vs Payload Size")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    apply_size_axis(ax)
    apply_latency_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_overhead_trends(data: dict, output: Path) -> None:
    bins = data["quantile_bins"]
    x = x_values(bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric, color, label in [
        ("encrypt_ms", "#9C2C2C", "Encrypt"),
        ("db_insert_ms", "#3C7A3B", "DB Insert"),
        ("db_fetch_ms", "#7A5C2C", "DB Fetch"),
        ("decrypt_ms", "#5B4B9A", "Decrypt"),
        ("end_to_end_ms", "#111111", "End-to-End"),
    ]:
        median = y_series(bins, "hybrid_overhead_pct", metric, "p50")
        ax.plot(x, median, linewidth=2.0 if metric == "end_to_end_ms" else 1.5, color=color, label=label)

    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Payload Size (MiB, median per bin)")
    ax.set_ylabel("Hybrid Overhead vs Classical (%)")
    ax.set_title("FAERS Benchmark Trend: Hybrid Overhead by Payload Size")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    apply_size_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_storage_trends(data: dict, output: Path) -> None:
    bins = data["quantile_bins"]
    x = x_values(bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    for scheme, color in [("classical", "#2E5EAA"), ("hybrid", "#C96B28")]:
        amp = y_series(bins, scheme, "amplification", "p50")
        ax.plot(x, amp, linewidth=2.0, color=color, label=f"{scheme} amplification")

    ax.set_xscale("log")
    ax.set_xlabel("Payload Size (MiB, median per bin)")
    ax.set_ylabel("Stored Bytes / Plaintext Bytes")
    ax.set_title("FAERS Benchmark Trend: Storage Amplification vs Payload Size")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    apply_size_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def load_exact_storage_series(data: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, list[tuple[float, float]]] = {"classical": [], "hybrid": []}
    for row in data["results"]:
        scheme = row["scheme"]
        if scheme not in series:
            continue
        size_mib = row["size_bytes"] / (1024 * 1024)
        amp = row["envelope_bytes"]["amplification"]
        series[scheme].append((size_mib, amp))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scheme, points in series.items():
        points.sort(key=lambda item: item[0])
        out[scheme] = (
            np.array([x for x, _ in points], dtype=float),
            np.array([y for _, y in points], dtype=float),
        )
    return out


def plot_storage_comparison(primary: dict, secondary_exact: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = primary["quantile_bins"]
    x = x_values(bins)
    for scheme, color in [("classical", "#2E5EAA"), ("hybrid", "#C96B28")]:
        amp = y_series(bins, scheme, "amplification", "p50")
        ax.plot(
            x,
            amp,
            linewidth=2.0,
            color=color,
            linestyle="-",
            label=f"FAERS {scheme}",
        )

    ggvp_series = load_exact_storage_series(secondary_exact)
    for scheme, color in [("classical", "#2E5EAA"), ("hybrid", "#C96B28")]:
        x, amp = ggvp_series[scheme]
        ax.plot(
            x,
            amp,
            linewidth=1.8,
            color=color,
            linestyle="--",
            marker="o",
            markersize=3.5,
            label=f"GGVP {scheme}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Payload Size")
    ax.set_ylabel("Stored Bytes / Plaintext Bytes")
    ax.set_title("Storage Amplification vs Payload Size: FAERS and GGVP")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    apply_sparse_size_axis(ax, [0.002, 0.01, 0.05, 0.2, 5, 7, 50, 200])
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_storage_comparison_binned(primary: dict, secondary: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for data, dataset_label, colors, linestyles in [
        (primary, "FAERS", {"classical": "#2E5EAA", "hybrid": "#C96B28"}, {"classical": "-", "hybrid": "-"}),
        (secondary, "GGVP", {"classical": "#2E5EAA", "hybrid": "#C96B28"}, {"classical": "--", "hybrid": "--"}),
    ]:
        bins = data["quantile_bins"]
        x = x_values(bins)
        for scheme in ("classical", "hybrid"):
            amp = y_series(bins, scheme, "amplification", "p50")
            ax.plot(
                x,
                amp,
                linewidth=2.0,
                color=colors[scheme],
                linestyle=linestyles[scheme],
                label=f"{dataset_label} {scheme}",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Payload Size")
    ax.set_ylabel("Stored Bytes / Plaintext Bytes")
    ax.set_title("Storage Amplification vs Payload Size: FAERS and GGVP")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    apply_sparse_size_axis(ax, [0.002, 0.01, 0.05, 0.2, 5, 50, 200])
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def load_hybrid_only_series(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, amp) for the hybrid scheme from a raw benchmark JSON."""
    points: list[tuple[float, float]] = []
    for row in data["results"]:
        if row["scheme"] != "hybrid":
            continue
        size_mib = row["size_bytes"] / (1024 * 1024)
        amp = row["envelope_bytes"]["amplification"]
        points.append((size_mib, amp))
    points.sort(key=lambda item: item[0])
    return (
        np.array([x for x, _ in points], dtype=float),
        np.array([y for _, y in points], dtype=float),
    )


def plot_storage_kem_comparison(
    faers_base: dict,
    ggvp_base: dict,
    faers_512: dict,
    faers_1024: dict,
    ggvp_512: dict,
    ggvp_1024: dict,
    output: Path,
) -> None:
    """Compare classical and hybrid KEM variants (512/768/1024) across FAERS and GGVP."""
    kem_colors = {
        "classical": "#2E5EAA",
        "768": "#C96B28",
        "512": "#3C7A3B",
        "1024": "#5B4B9A",
    }
    kem_labels = {
        "classical": "classical",
        "768": "hybrid ML-KEM-768",
        "512": "hybrid ML-KEM-512",
        "1024": "hybrid ML-KEM-1024",
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    # FAERS base: classical + hybrid-768 (aggregated bins)
    faers_bins = faers_base["quantile_bins"]
    faers_x = x_values(faers_bins)
    for scheme, kem_key in [("classical", "classical"), ("hybrid", "768")]:
        amp = y_series(faers_bins, scheme, "amplification", "p50")
        ax.plot(faers_x, amp, linewidth=2.0, color=kem_colors[kem_key],
                linestyle="-", label=f"FAERS {kem_labels[kem_key]}")

    # FAERS 512 + 1024 (raw exact points, hybrid only)
    for raw, kem_key in [(faers_512, "512"), (faers_1024, "1024")]:
        xs, amps = load_hybrid_only_series(raw)
        ax.plot(xs, amps, linewidth=2.0, color=kem_colors[kem_key],
                linestyle="-", label=f"FAERS {kem_labels[kem_key]}")

    # GGVP base: classical + hybrid-768 (raw exact points)
    ggvp_base_series = load_exact_storage_series(ggvp_base)
    for scheme, kem_key in [("classical", "classical"), ("hybrid", "768")]:
        xs, amps = ggvp_base_series[scheme]
        ax.plot(xs, amps, linewidth=1.8, color=kem_colors[kem_key],
                linestyle="--", marker="o", markersize=3,
                label=f"GGVP {kem_labels[kem_key]}")

    # GGVP 512 + 1024 (raw exact points, hybrid only)
    for raw, kem_key in [(ggvp_512, "512"), (ggvp_1024, "1024")]:
        xs, amps = load_hybrid_only_series(raw)
        ax.plot(xs, amps, linewidth=1.8, color=kem_colors[kem_key],
                linestyle="--", marker="o", markersize=3,
                label=f"GGVP {kem_labels[kem_key]}")

    ax.set_xscale("log")
    ax.set_xlabel("Payload Size")
    ax.set_ylabel("Stored Bytes / Plaintext Bytes")
    ax.set_title("Storage Amplification vs Payload Size: KEM Variant Comparison (FAERS and GGVP)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    apply_sparse_size_axis(ax, [0.002, 0.01, 0.05, 0.2, 5, 50, 200])
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_component_mix(data: dict, output: Path) -> None:
    bins = data["quantile_bins"]
    x = x_values(bins)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, scheme, title in [
        (axes[0], "classical", "Classical Time Composition"),
        (axes[1], "hybrid", "Hybrid Time Composition"),
    ]:
        encrypt = y_series(bins, scheme, "encrypt_ms", "p50")
        db_insert = y_series(bins, scheme, "db_insert_ms", "p50")
        db_fetch = y_series(bins, scheme, "db_fetch_ms", "p50")
        decrypt = y_series(bins, scheme, "decrypt_ms", "p50")
        total = encrypt + db_insert + db_fetch + decrypt

        shares = [
            np.divide(encrypt, total, out=np.zeros_like(total), where=total != 0),
            np.divide(db_insert, total, out=np.zeros_like(total), where=total != 0),
            np.divide(db_fetch, total, out=np.zeros_like(total), where=total != 0),
            np.divide(decrypt, total, out=np.zeros_like(total), where=total != 0),
        ]
        ax.stackplot(
            x,
            shares,
            labels=["Encrypt", "DB Insert", "DB Fetch", "Decrypt"],
            colors=["#9C2C2C", "#3C7A3B", "#7A5C2C", "#5B4B9A"],
            alpha=0.85,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of Total")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Payload Size (MiB, median per bin)")
    axes[0].legend(frameon=False, ncol=4, loc="upper center")
    apply_size_axis(axes[1])
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_component_mix_comparison(primary: dict, secondary: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    component_labels = ["Encrypt", "DB Insert", "DB Fetch", "Decrypt"]
    component_colors = ["#9C2C2C", "#3C7A3B", "#7A5C2C", "#5B4B9A"]

    for col, (dataset_label, data) in enumerate([("FAERS", primary), ("GGVP", secondary)]):
        bins = data["quantile_bins"]
        x = x_values(bins)
        for row, scheme in enumerate(["classical", "hybrid"]):
            ax = axes[row][col]
            encrypt = y_series(bins, scheme, "encrypt_ms", "p50")
            db_insert = y_series(bins, scheme, "db_insert_ms", "p50")
            db_fetch = y_series(bins, scheme, "db_fetch_ms", "p50")
            decrypt = y_series(bins, scheme, "decrypt_ms", "p50")
            total = encrypt + db_insert + db_fetch + decrypt

            shares = [
                np.divide(encrypt, total, out=np.zeros_like(total), where=total != 0),
                np.divide(db_insert, total, out=np.zeros_like(total), where=total != 0),
                np.divide(db_fetch, total, out=np.zeros_like(total), where=total != 0),
                np.divide(decrypt, total, out=np.zeros_like(total), where=total != 0),
            ]
            ax.stackplot(x, shares, labels=component_labels, colors=component_colors, alpha=0.85)
            ax.set_ylim(0, 1)
            ax.set_xscale("log")
            ax.grid(alpha=0.2)
            ax.set_title(f"{dataset_label} {'Classical' if scheme == 'classical' else 'Hybrid'}")

            if col == 0:
                ax.set_ylabel("Share of Total")
            if row == 1:
                ax.set_xlabel("Payload Size")
                if dataset_label == "FAERS":
                    apply_sparse_size_axis(ax, [0.002, 0.01, 0.05, 0.2])
                else:
                    apply_sparse_size_axis(ax, [7, 50, 200])

    axes[0][0].legend(frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize aggregated benchmark trends.")
    parser.add_argument("--input", required=True, help="Path to aggregated benchmark JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG outputs.")
    parser.add_argument("--compare-input", help="Optional second aggregated benchmark JSON for comparison plots.")
    parser.add_argument("--compare-raw-input", help="Optional raw benchmark JSON for exact comparison-series plots.")
    parser.add_argument("--kem512-faers-input", help="Raw FAERS benchmark JSON for ML-KEM-512 hybrid.")
    parser.add_argument("--kem1024-faers-input", help="Raw FAERS benchmark JSON for ML-KEM-1024 hybrid.")
    parser.add_argument("--kem512-ggvp-input", help="Raw GGVP benchmark JSON for ML-KEM-512 hybrid.")
    parser.add_argument("--kem1024-ggvp-input", help="Raw GGVP benchmark JSON for ML-KEM-1024 hybrid.")
    args = parser.parse_args()

    data = load(Path(args.input))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_latency_trends(data, outdir / "12_faers_latency_trends.png")
    plot_overhead_trends(data, outdir / "13_faers_overhead_trends.png")
    plot_storage_trends(data, outdir / "14_faers_storage_amplification_trends.png")
    plot_component_mix(data, outdir / "15_faers_time_composition.png")
    if args.compare_input:
        compare_data = load(Path(args.compare_input))
        plot_component_mix_comparison(data, compare_data, outdir / "15_faers_time_composition.png")
        if args.compare_raw_input:
            compare_raw_data = load(Path(args.compare_raw_input))
            plot_storage_comparison(data, compare_raw_data, outdir / "14_combined_storage_amplification_trends.png")
        else:
            plot_storage_comparison_binned(data, compare_data, outdir / "14_combined_storage_amplification_trends.png")

    if all([args.kem512_faers_input, args.kem1024_faers_input,
            args.kem512_ggvp_input, args.kem1024_ggvp_input,
            args.compare_raw_input]):
        ggvp_base_raw = load(Path(args.compare_raw_input))
        faers_512 = load(Path(args.kem512_faers_input))
        faers_1024 = load(Path(args.kem1024_faers_input))
        ggvp_512 = load(Path(args.kem512_ggvp_input))
        ggvp_1024 = load(Path(args.kem1024_ggvp_input))
        plot_storage_kem_comparison(
            data, ggvp_base_raw,
            faers_512, faers_1024,
            ggvp_512, ggvp_1024,
            outdir / "19_kem_storage_amplification_comparison.png",
        )


if __name__ == "__main__":
    main()
