"""
visualize.py
============
Generate charts from benchmark_results.json.

Usage:
    python visualize.py [--input results/benchmark_results.json] [--output-dir results/]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────────

CLASSICAL_COLOR = "#4C72B0"
HYBRID_COLOR    = "#DD8452"
METRICS         = ["encrypt", "db_insert", "db_fetch", "decrypt", "end_to_end"]
METRIC_LABELS   = ["Encrypt", "DB Insert", "DB Fetch", "Decrypt", "End-to-End"]
STACK_COLORS    = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":   150,
})


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _by_scheme(data: dict) -> tuple[list[dict], list[dict]]:
    classical = [r for r in data["results"] if r["scheme"] == "classical"]
    hybrid    = [r for r in data["results"] if r["scheme"] == "hybrid"]
    return classical, hybrid


def _size_labels(results: list[dict]) -> list[str]:
    return [r["size_label"] for r in results]


# ── Chart 1: End-to-End Mean Latency — Classical vs Hybrid ───────────────────

def plot_e2e_comparison(data: dict, out_dir: Path) -> None:
    classical, hybrid = _by_scheme(data)
    labels = _size_labels(classical)
    x      = np.arange(len(labels))
    w      = 0.35

    # Use p50 (median) as the bar height — robust against the large outliers
    # seen at hybrid ≥16 KB where mean/p50 diverges by up to 1.9×.
    cl_p50  = [r["end_to_end"]["p50_ms"]  for r in classical]
    hy_p50  = [r["end_to_end"]["p50_ms"]  for r in hybrid]
    cl_mean = [r["end_to_end"]["mean_ms"] for r in classical]
    hy_mean = [r["end_to_end"]["mean_ms"] for r in hybrid]
    cl_p95  = [r["end_to_end"]["p95_ms"]  for r in classical]
    hy_p95  = [r["end_to_end"]["p95_ms"]  for r in hybrid]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, cl_p50, w, label="Classical", color=CLASSICAL_COLOR, alpha=0.9)
    ax.bar(x + w/2, hy_p50, w, label=f"Hybrid ({data['pq_kem_id']})", color=HYBRID_COLOR, alpha=0.9)

    # Mean markers (diamond) — shows skew when above bar
    ax.scatter(x - w/2, cl_mean, marker="D", s=40, color=CLASSICAL_COLOR, zorder=5, label="_nolegend_")
    ax.scatter(x + w/2, hy_mean, marker="D", s=40, color=HYBRID_COLOR,    zorder=5, label="_nolegend_")

    # p95 markers (tick)
    ax.scatter(x - w/2, cl_p95, marker="_", s=200, color=CLASSICAL_COLOR, zorder=5, linewidths=2)
    ax.scatter(x + w/2, hy_p95, marker="_", s=200, color=HYBRID_COLOR,    zorder=5, linewidths=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Record Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-End Latency: Classical vs Hybrid PQC\n(bars = p50, ◆ = mean, tick = p95)")
    ax.legend()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "01_e2e_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 2: Per-Metric Mean Latency — line plot across sizes ────────────────

def plot_per_metric_lines(data: dict, out_dir: Path) -> None:
    classical, hybrid = _by_scheme(data)
    labels = _size_labels(classical)
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes = axes.flatten()

    for i, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = axes[i]
        cl_p50  = [r[metric]["p50_ms"]  for r in classical]
        hy_p50  = [r[metric]["p50_ms"]  for r in hybrid]
        cl_p95  = [r[metric]["p95_ms"]  for r in classical]
        hy_p95  = [r[metric]["p95_ms"]  for r in hybrid]

        ax.plot(x, cl_p50, "o-", color=CLASSICAL_COLOR, label="Classical", linewidth=2, markersize=5)
        ax.plot(x, hy_p50, "s-", color=HYBRID_COLOR,    label="Hybrid",    linewidth=2, markersize=5)
        ax.fill_between(x, cl_p50, cl_p95, color=CLASSICAL_COLOR, alpha=0.12)
        ax.fill_between(x, hy_p50, hy_p95, color=HYBRID_COLOR,    alpha=0.12)

        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("ms")
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    # Hide unused subplot
    axes[-1].set_visible(False)

    fig.suptitle(
        f"Per-Metric Latency: Classical vs Hybrid PQC ({data['pq_kem_id']})\n"
        "(solid line = p50, shaded band = p50→p95)",
        fontsize=13,
    )
    fig.tight_layout()
    path = out_dir / "02_per_metric_lines.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 3: Overhead (%) of Hybrid vs Classical per metric ──────────────────

def plot_overhead_pct(data: dict, out_dir: Path) -> None:
    classical, hybrid = _by_scheme(data)
    labels = _size_labels(classical)
    x      = np.arange(len(labels))
    w      = 0.13
    offsets = np.linspace(-(len(METRICS)-1)/2, (len(METRICS)-1)/2, len(METRICS)) * w

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        cl_vals = np.array([r[metric]["p50_ms"] for r in classical])
        hy_vals = np.array([r[metric]["p50_ms"] for r in hybrid])
        pct     = (hy_vals - cl_vals) / cl_vals * 100
        ax.bar(x + offsets[i], pct, w * 0.9, label=label, color=STACK_COLORS[i], alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Record Size")
    ax.set_ylabel("Overhead vs Classical (%)")
    ax.set_title(f"Hybrid PQC Overhead vs Classical — per Metric (p50)\n({data['pq_kem_id']})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "03_overhead_pct.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 4: Stacked bar — component breakdown of End-to-End (mean) ──────────

def plot_stacked_breakdown(data: dict, out_dir: Path) -> None:
    classical, hybrid = _by_scheme(data)
    labels = _size_labels(classical)
    x      = np.arange(len(labels))
    w      = 0.35

    component_metrics = ["encrypt", "db_insert", "db_fetch", "decrypt"]
    component_labels  = ["Encrypt", "DB Insert", "DB Fetch", "Decrypt"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, results, scheme_label in [
        (axes[0], classical, "Classical"),
        (axes[1], hybrid,    f"Hybrid ({data['pq_kem_id']})"),
    ]:
        bottoms = np.zeros(len(labels))
        for j, (metric, mlabel) in enumerate(zip(component_metrics, component_labels)):
            vals = np.array([r[metric]["p50_ms"] for r in results])
            ax.bar(x, vals, w, bottom=bottoms, label=mlabel, color=STACK_COLORS[j], alpha=0.88)
            bottoms += vals

        # e2e p50 as a dot on top to show any unattributed gap
        e2e = np.array([r["end_to_end"]["p50_ms"] for r in results])
        ax.scatter(x, e2e, color="black", zorder=5, s=40, label="E2E p50", marker="D")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Record Size")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(scheme_label)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Operation Breakdown (p50 Latency per Component)", fontsize=13)
    fig.tight_layout()
    path = out_dir / "04_stacked_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 5: Latency percentile fan (p50 / p95 / p99) — End-to-End ──────────

def plot_percentile_fan(data: dict, out_dir: Path) -> None:
    classical, hybrid = _by_scheme(data)
    labels = _size_labels(classical)
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, results, color, scheme_label in [
        (axes[0], classical, CLASSICAL_COLOR, "Classical"),
        (axes[1], hybrid,    HYBRID_COLOR,    f"Hybrid ({data['pq_kem_id']})"),
    ]:
        p50 = np.array([r["end_to_end"]["p50_ms"] for r in results])
        p95 = np.array([r["end_to_end"]["p95_ms"] for r in results])
        p99 = np.array([r["end_to_end"]["p99_ms"] for r in results])

        ax.fill_between(x, p50, p99, alpha=0.18, color=color, label="p50–p99 band")
        ax.fill_between(x, p50, p95, alpha=0.28, color=color, label="p50–p95 band")
        ax.plot(x, p50, "o-", color=color, linewidth=2, label="p50")
        ax.plot(x, p95, "s--", color=color, linewidth=1.5, label="p95")
        ax.plot(x, p99, "^:", color=color, linewidth=1.5, label="p99")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Record Size")
        ax.set_ylabel("End-to-End Latency (ms)")
        ax.set_title(f"{scheme_label}\nEnd-to-End Percentile Fan")
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "05_percentile_fan.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 6: Total Stored Bytes — Classical vs Hybrid ────────────────────────

def plot_storage_total(data: dict, out_dir: Path) -> None:
    classical, hybrid = _by_scheme(data)
    if not classical[0].get("envelope_bytes"):
        return

    labels = _size_labels(classical)
    x = np.arange(len(labels))
    w = 0.35

    cl_total = [r["envelope_bytes"]["total_stored_bytes"] for r in classical]
    hy_total = [r["envelope_bytes"]["total_stored_bytes"] for r in hybrid]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_cl = ax.bar(x - w/2, cl_total, w, label="Classical", color=CLASSICAL_COLOR, alpha=0.9)
    bars_hy = ax.bar(x + w/2, hy_total, w, label=f"Hybrid ({data['pq_kem_id']})", color=HYBRID_COLOR, alpha=0.9)

    # Annotate delta on top of each hybrid bar
    for xi, (cl, hy) in enumerate(zip(cl_total, hy_total)):
        delta = hy - cl
        ax.annotate(
            f"+{delta:,} B",
            xy=(xi + w/2, hy),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5, color=HYBRID_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Plaintext Size")
    ax.set_ylabel("Total Stored Bytes")
    ax.set_title(
        f"Total Envelope Size: Classical vs Hybrid PQC ({data['pq_kem_id']})\n"
        "Annotations show extra bytes added by the PQ KEM layer"
    )
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "06_storage_total.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 7: Envelope Composition — stacked breakdown ────────────────────────

def plot_storage_composition(data: dict, out_dir: Path) -> None:
    classical, hybrid = _by_scheme(data)
    if not classical[0].get("envelope_bytes"):
        return

    labels = _size_labels(classical)
    x = np.arange(len(labels))
    w = 0.5

    # Layers: ciphertext (payload), header (fixed crypto metadata), pq_ct (hybrid-only)
    LAYER_COLORS = ["#6BAED6", "#74C476", "#FB6A4A"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, results, scheme_label in [
        (axes[0], classical, "Classical"),
        (axes[1], hybrid,    f"Hybrid ({data['pq_kem_id']})"),
    ]:
        env = [r["envelope_bytes"] for r in results]

        cipher  = np.array([e["ciphertext_bytes"] for e in env])
        header  = np.array([e["header_bytes"]     for e in env])
        pq_ct   = np.array([e["pq_ct_bytes"]      for e in env])

        ax.bar(x, cipher, w, label="Payload ciphertext", color=LAYER_COLORS[0], alpha=0.9)
        ax.bar(x, header, w, bottom=cipher, label="Crypto header\n(nonce+tag+DEK+PK+salt+info+AAD)", color=LAYER_COLORS[1], alpha=0.9)
        if pq_ct.any():
            ax.bar(x, pq_ct, w, bottom=cipher + header, label=f"PQ-KEM ciphertext\n({data['pq_kem_id']})", color=LAYER_COLORS[2], alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Plaintext Size")
        ax.set_ylabel("Bytes Stored")
        ax.set_title(scheme_label)
        ax.legend(fontsize=8, loc="upper left")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Envelope Composition: What Gets Stored per Record", fontsize=13)
    fig.tight_layout()
    path = out_dir / "07_storage_composition.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 8: Storage Amplification Ratio ─────────────────────────────────────

def plot_storage_amplification(data: dict, out_dir: Path) -> None:
    classical, hybrid = _by_scheme(data)
    if not classical[0].get("envelope_bytes"):
        return

    labels = _size_labels(classical)
    x = np.arange(len(labels))

    cl_amp = [r["envelope_bytes"]["amplification"] for r in classical]
    hy_amp = [r["envelope_bytes"]["amplification"] for r in hybrid]

    # Also compute overhead-only ratio (overhead / plaintext) so readers can see
    # how many extra bytes per plaintext byte are paid beyond the payload itself.
    cl_overhead_ratio = [
        r["envelope_bytes"]["overhead_bytes"] / r["envelope_bytes"]["plaintext_bytes"]
        for r in classical
    ]
    hy_overhead_ratio = [
        r["envelope_bytes"]["overhead_bytes"] / r["envelope_bytes"]["plaintext_bytes"]
        for r in hybrid
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total amplification (total_stored / plaintext)
    ax = axes[0]
    ax.plot(x, cl_amp, "o-", color=CLASSICAL_COLOR, linewidth=2, markersize=6, label="Classical")
    ax.plot(x, hy_amp, "s-", color=HYBRID_COLOR,    linewidth=2, markersize=6, label=f"Hybrid ({data['pq_kem_id']})")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", label="1.0× (no overhead)")
    for xi, (cl, hy) in enumerate(zip(cl_amp, hy_amp)):
        ax.annotate(f"{hy:.3f}×", (xi, hy), textcoords="offset points", xytext=(4, 4), fontsize=7.5, color=HYBRID_COLOR)
        ax.annotate(f"{cl:.3f}×", (xi, cl), textcoords="offset points", xytext=(4, -12), fontsize=7.5, color=CLASSICAL_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Plaintext Size")
    ax.set_ylabel("Amplification (total stored / plaintext)")
    ax.set_title("Storage Amplification Ratio\n(total stored ÷ plaintext bytes)")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Right: overhead bytes per plaintext byte (overhead only, excludes payload)
    ax = axes[1]
    ax.plot(x, cl_overhead_ratio, "o-", color=CLASSICAL_COLOR, linewidth=2, markersize=6, label="Classical overhead")
    ax.plot(x, hy_overhead_ratio, "s-", color=HYBRID_COLOR,    linewidth=2, markersize=6, label="Hybrid overhead")
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Plaintext Size")
    ax.set_ylabel("Overhead bytes per plaintext byte")
    ax.set_title("Envelope Overhead Ratio\n(non-payload bytes ÷ plaintext bytes)")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"Storage Overhead Convergence — PQ overhead shrinks relative to payload size\n({data['pq_kem_id']})",
        fontsize=13,
    )
    fig.tight_layout()
    path = out_dir / "08_storage_amplification.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Visualize PQC benchmark results.")
    p.add_argument("--input",      default="results/benchmark_results.json")
    p.add_argument("--output-dir", default="results/")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load(args.input)
    print(f"\n  Loaded results from {args.input}")
    print(f"  Run at : {data['run_at']}   Iterations: {data['iterations']}")
    print(f"  PQ KEM : {data['pq_kem_id']}")
    print(f"  Output : {out_dir}/\n")

    plot_e2e_comparison(data, out_dir)
    plot_per_metric_lines(data, out_dir)
    plot_overhead_pct(data, out_dir)
    plot_stacked_breakdown(data, out_dir)
    plot_percentile_fan(data, out_dir)
    plot_storage_total(data, out_dir)
    plot_storage_composition(data, out_dir)
    plot_storage_amplification(data, out_dir)

    print("\n  Done. 8 charts saved.")


if __name__ == "__main__":
    main()
