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


def _aggregated_size_labels(bins: list[dict]) -> list[str]:
    labels: list[str] = []
    for b in bins:
        mib = b["size_mib"]["p50"]
        if mib < 1:
            kb = mib * 1024
            labels.append(f"{kb:.0f} KB" if kb >= 10 else f"{kb:.1f} KB")
        else:
            labels.append(f"{mib:.0f} MB" if mib >= 10 else f"{mib:.1f} MB")
    return labels


def _fmt_payload_size_mib(mib: float) -> str:
    if mib < 1:
        kb = mib * 1024
        return f"{kb:.0f} KB" if kb >= 10 else f"{kb:.1f} KB"
    return f"{mib:.0f} MB" if mib >= 10 else f"{mib:.1f} MB"


def _exact_e2e_series(data: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, list[tuple[float, float]]] = {"classical": [], "hybrid": []}
    for row in data.get("results", []):
        scheme = row.get("scheme")
        if scheme not in series:
            continue
        size_mib = row["size_bytes"] / (1024 * 1024)
        p50_ms = row["end_to_end"]["p50_ms"]
        series[scheme].append((size_mib, p50_ms))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scheme, points in series.items():
        points.sort(key=lambda item: item[0])
        out[scheme] = (
            np.array([x for x, _ in points], dtype=float),
            np.array([y for _, y in points], dtype=float),
        )
    return out


# ── Chart 1: End-to-End Mean Latency — Classical vs Hybrid ───────────────────

def plot_e2e_comparison(
    data: dict,
    out_dir: Path,
    overlay_data: dict | None = None,
    *,
    output_name: str = "01_e2e_comparison.png",
    title_override: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    pq_kem_id = data.get("pq_kem_id", data.get("source", {}).get("pq_kem_id", "PQC"))

    if "quantile_bins" in data:
        bins = data["quantile_bins"]
        x = np.array([b["size_mib"]["p50"] for b in bins], dtype=float)

        cl_p50 = [b["classical"]["end_to_end_ms"]["p50"] for b in bins]
        hy_p50 = [b["hybrid"]["end_to_end_ms"]["p50"] for b in bins]
        cl_mean = [b["classical"]["end_to_end_ms"]["mean"] for b in bins]
        hy_mean = [b["hybrid"]["end_to_end_ms"]["mean"] for b in bins]
        cl_p95 = [b["classical"]["end_to_end_ms"]["p95"] for b in bins]
        hy_p95 = [b["hybrid"]["end_to_end_ms"]["p95"] for b in bins]
        xlabel = "Payload Size"
        title = "End-to-End Latency: Classical vs hybrid"

        ax.plot(x, cl_p50, "o-", color=CLASSICAL_COLOR, linewidth=2.2, markersize=5, label="Classical")
        ax.plot(x, hy_p50, "s-", color=HYBRID_COLOR, linewidth=2.2, markersize=5, label="Hybrid")
        ax.fill_between(x, cl_p50, cl_p95, color=CLASSICAL_COLOR, alpha=0.14, label="_nolegend_")
        ax.fill_between(x, hy_p50, hy_p95, color=HYBRID_COLOR, alpha=0.14, label="_nolegend_")
        ax.scatter(x, cl_mean, marker="D", s=28, color=CLASSICAL_COLOR, zorder=5, label="_nolegend_")
        ax.scatter(x, hy_mean, marker="D", s=28, color=HYBRID_COLOR, zorder=5, label="_nolegend_")
        if overlay_data is not None:
            overlay = _exact_e2e_series(overlay_data)
            gx, gy = overlay["classical"]
            ax.scatter(
                gx,
                gy,
                marker="o",
                s=34,
                facecolors="white",
                edgecolors=CLASSICAL_COLOR,
                linewidths=1.4,
                zorder=6,
                label="GGVP classical",
            )
            gx, gy = overlay["hybrid"]
            ax.scatter(
                gx,
                gy,
                marker="s",
                s=34,
                facecolors="white",
                edgecolors=HYBRID_COLOR,
                linewidths=1.4,
                zorder=6,
                label="GGVP hybrid",
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: _fmt_payload_size_mib(v) if v > 0 else ""))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        classical, hybrid = _by_scheme(data)
        if not classical and not hybrid:
            print(f"  WARNING: no results to plot in {output_name}, skipping.")
            plt.close(fig)
            return
        # Build x-axis from whichever series is non-empty (prefer classical for labels)
        _ref = classical if classical else hybrid
        labels = _size_labels(_ref)
        x = np.arange(len(labels))
        xlabel = "Payload Size"
        title = "End-to-End Latency: Classical vs Hybrid"

        if classical and hybrid and len(classical) != len(hybrid):
            print(f"  WARNING: classical ({len(classical)}) and hybrid ({len(hybrid)}) result counts differ; "
                  "aligning on common size_labels only.")
            cl_sizes = {r["size_label"]: r for r in classical}
            hy_sizes = {r["size_label"]: r for r in hybrid}
            common = [lbl for lbl in labels if lbl in cl_sizes and lbl in hy_sizes]
            classical = [cl_sizes[lbl] for lbl in common]
            hybrid    = [hy_sizes[lbl] for lbl in common]
            labels = common
            x = np.arange(len(labels))

        if classical:
            cl_p50  = [r["end_to_end"]["p50_ms"]  for r in classical]
            cl_mean = [r["end_to_end"]["mean_ms"] for r in classical]
            cl_p95  = [r["end_to_end"]["p95_ms"]  for r in classical]
            ax.plot(x, cl_p50, "o-", color=CLASSICAL_COLOR, linewidth=2.2, markersize=5, label="Classical")
            ax.fill_between(x, cl_p50, cl_p95, color=CLASSICAL_COLOR, alpha=0.14, label="_nolegend_")
            ax.scatter(x, cl_mean, marker="D", s=28, color=CLASSICAL_COLOR, zorder=5, label="_nolegend_")
        if hybrid:
            hy_p50  = [r["end_to_end"]["p50_ms"]  for r in hybrid]
            hy_mean = [r["end_to_end"]["mean_ms"] for r in hybrid]
            hy_p95  = [r["end_to_end"]["p95_ms"]  for r in hybrid]
            if len(hy_p50) != len(x):
                print(f"  WARNING: hybrid result count ({len(hy_p50)}) != x-axis length ({len(x)}); skipping hybrid line.")
            else:
                ax.plot(x, hy_p50, "s-", color=HYBRID_COLOR, linewidth=2.2, markersize=5, label="Hybrid")
                ax.fill_between(x, hy_p50, hy_p95, color=HYBRID_COLOR, alpha=0.14, label="_nolegend_")
                ax.scatter(x, hy_mean, marker="D", s=28, color=HYBRID_COLOR, zorder=5, label="_nolegend_")

    if "quantile_bins" in data:
        ax.set_xlabel(xlabel)
    else:
        if len(x):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title_override or title)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / output_name
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Chart 2: Per-Metric Mean Latency — line plot across sizes ────────────────

def _exact_metric_series(data: dict, metric: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, list[tuple[float, float]]] = {"classical": [], "hybrid": []}
    for row in data.get("results", []):
        scheme = row.get("scheme")
        if scheme not in series:
            continue
        size_mib = row["size_bytes"] / (1024 * 1024)
        p50_ms = row[metric]["p50_ms"]
        series[scheme].append((size_mib, p50_ms))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scheme, points in series.items():
        points.sort(key=lambda item: item[0])
        out[scheme] = (
            np.array([x for x, _ in points], dtype=float),
            np.array([y for _, y in points], dtype=float),
        )
    return out


def plot_per_metric_lines(
    data: dict,
    out_dir: Path,
    overlay_data: dict | None = None,
    *,
    output_name: str = "02_per_metric_lines.png",
) -> None:
    aggregated = "quantile_bins" in data
    if not aggregated:
        classical, hybrid = _by_scheme(data)
        _ref = classical if classical else hybrid
        if not _ref:
            print("  WARNING: no results for per-metric lines, skipping.")
            return
        labels = _size_labels(_ref)
        x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes = axes.flatten()

    for i, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = axes[i]
        if aggregated:
            metric_key = f"{metric}_ms"
            bins = data["quantile_bins"]
            x = np.array([b["size_mib"]["p50"] for b in bins], dtype=float)
            cl_p50 = [b["classical"][metric_key]["p50"] for b in bins]
            hy_p50 = [b["hybrid"][metric_key]["p50"] for b in bins]
            cl_p95 = [b["classical"][metric_key]["p95"] for b in bins]
            hy_p95 = [b["hybrid"][metric_key]["p95"] for b in bins]

            ax.plot(x, cl_p50, "o-", color=CLASSICAL_COLOR, label="Classical", linewidth=2, markersize=4)
            ax.plot(x, hy_p50, "s-", color=HYBRID_COLOR, label="Hybrid", linewidth=2, markersize=4)
            ax.fill_between(x, cl_p50, cl_p95, color=CLASSICAL_COLOR, alpha=0.12)
            ax.fill_between(x, hy_p50, hy_p95, color=HYBRID_COLOR, alpha=0.12)
            if overlay_data is not None:
                overlay = _exact_metric_series(overlay_data, metric)
                gx, gy = overlay["classical"]
                ax.scatter(gx, gy, marker="o", s=24, facecolors="white", edgecolors=CLASSICAL_COLOR, linewidths=1.2, zorder=6, label="GGVP classical")
                gx, gy = overlay["hybrid"]
                ax.scatter(gx, gy, marker="s", s=24, facecolors="white", edgecolors=HYBRID_COLOR, linewidths=1.2, zorder=6, label="GGVP hybrid")
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: _fmt_payload_size_mib(v) if v > 0 else ""))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_xlabel("Payload Size")
        else:
            if classical:
                cl_p50 = [r[metric]["p50_ms"] for r in classical]
                cl_p95 = [r[metric]["p95_ms"] for r in classical]
                ax.plot(x, cl_p50, "o-", color=CLASSICAL_COLOR, label="Classical", linewidth=2, markersize=5)
                ax.fill_between(x, cl_p50, cl_p95, color=CLASSICAL_COLOR, alpha=0.12)
            if hybrid:
                hy_p50 = [r[metric]["p50_ms"] for r in hybrid]
                hy_p95 = [r[metric]["p95_ms"] for r in hybrid]
                if len(hy_p50) == len(x):
                    ax.plot(x, hy_p50, "s-", color=HYBRID_COLOR, label="Hybrid", linewidth=2, markersize=5)
                    ax.fill_between(x, hy_p50, hy_p95, color=HYBRID_COLOR, alpha=0.12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

        ax.set_title(label)
        ax.set_ylabel("ms")
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    # Hide unused subplot
    axes[-1].set_visible(False)

    pq_kem_id = data.get("pq_kem_id", data.get("source", {}).get("pq_kem_id", "PQC"))
    if aggregated and overlay_data is not None:
        title = f"Per-Metric Latency: Classical vs Hybrid ({pq_kem_id})\n(solid line = FAERS p50, shaded band = FAERS p50→p95, points = GGVP)"
    elif aggregated:
        title = f"Per-Metric Latency: Classical vs Hybrid ({pq_kem_id})\n(solid line = p50, shaded band = p50→p95)"
    else:
        title = f"Per-Metric Latency: Classical vs Hybrid PQC ({pq_kem_id})\n(solid line = p50, shaded band = p50→p95)"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path = out_dir / output_name
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
    p.add_argument("--overlay-input", help="Optional raw benchmark JSON to overlay exact points on chart 1.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load(args.input)
    overlay_data = _load(args.overlay_input) if args.overlay_input else None
    source = data.get("source", {})
    run_at = data.get("run_at", source.get("run_at", "unknown"))
    iterations = data.get("iterations", source.get("iterations", "unknown"))
    pq_kem_id = data.get("pq_kem_id", source.get("pq_kem_id", "unknown"))
    print(f"\n  Loaded results from {args.input}")
    print(f"  Run at : {run_at}   Iterations: {iterations}")
    print(f"  PQ KEM : {pq_kem_id}")
    print(f"  Output : {out_dir}/\n")

    plot_e2e_comparison(data, out_dir, overlay_data=overlay_data)
    if "quantile_bins" not in data:
        plot_per_metric_lines(data, out_dir)
        # Charts 3–8 require both classical and hybrid series.
        classical, hybrid = _by_scheme(data)
        has_both = bool(classical and hybrid)
        if has_both:
            plot_overhead_pct(data, out_dir)
            plot_stacked_breakdown(data, out_dir)
            plot_percentile_fan(data, out_dir)
            plot_storage_total(data, out_dir)
            plot_storage_composition(data, out_dir)
            plot_storage_amplification(data, out_dir)
            n_charts = 8
        else:
            missing = "classical" if not classical else "hybrid"
            print(f"  NOTE: skipping charts 3–8 (require both schemes; '{missing}' absent in this file).")
            n_charts = 2
        if data.get("policy_summary"):
            plot_policy_tier_summary(data, out_dir)
            n_charts += 1
        print(f"\n  Done. {n_charts} charts saved.")
    else:
        plot_per_metric_lines(data, out_dir, overlay_data=overlay_data)
        print("\n  Done. 2 charts saved from aggregated real-payload results.")


# ---------------------------------------------------------------------------
# Policy tier visualization
# ---------------------------------------------------------------------------

def extract_policy_chart_data(data: dict) -> dict:
    """Extract arrays from a JSON benchmark report for policy tier plotting.

    Returns a dict with keys suited for bar/scatter plots:

    ``tier_labels``
        List of tier label strings in order (e.g. ["baseline", "strong"]).
    ``record_counts`` / ``record_fractions``
        Per-tier integer counts and float fractions.
    ``plaintext_mib`` / ``plaintext_byte_fractions``
        Per-tier plaintext volume in MiB and byte fractions.
    ``stored_mib``
        Per-tier total stored bytes in MiB.
    ``storage_amplifications``
        Per-tier storage amplification factors.
    ``mean_e2e_ms`` / ``p50_e2e_ms`` / ``p95_e2e_ms``
        Per-tier end-to-end latency statistics (ms).
    ``component_ms``
        Dict mapping component name → per-tier mean latency array, for
        stacked-bar time-composition charts.
    ``escalated_record_fraction``
        Scalar: fraction of records that were escalated (0–1).
    ``escalated_plaintext_byte_fraction``
        Scalar: fraction of plaintext bytes that were escalated (0–1).
    ``overall_amplification``
        Scalar: overall storage amplification across all tiers.
    ``policy_mode``
        String policy mode label.
    ``threshold_bytes``
        Integer threshold used for adaptive mode.
    """
    ps = data.get("policy_summary", {})
    tiers = ps.get("tier_summaries", [])

    return {
        "tier_labels":                      [t["tier_label"] for t in tiers],
        "record_counts":                    [t["record_count"] for t in tiers],
        "record_fractions":                 [t["record_fraction"] for t in tiers],
        "plaintext_mib":                    [t["plaintext_bytes"] / (1024 * 1024) for t in tiers],
        "plaintext_byte_fractions":         [t["plaintext_byte_fraction"] for t in tiers],
        "stored_mib":                       [t["total_stored_bytes"] / (1024 * 1024) for t in tiers],
        "storage_amplifications":           [t["storage_amplification"] for t in tiers],
        "mean_e2e_ms":                      [t["mean_end_to_end_ms"] for t in tiers],
        "p50_e2e_ms":                       [t["p50_end_to_end_ms"] for t in tiers],
        "p95_e2e_ms":                       [t["p95_end_to_end_ms"] for t in tiers],
        "component_ms": {
            "encrypt":   [t["mean_encrypt_ms"]   for t in tiers],
            "sign":      [t["mean_sign_ms"]       for t in tiers],
            "db_insert": [t["mean_db_insert_ms"]  for t in tiers],
            "db_fetch":  [t["mean_db_fetch_ms"]   for t in tiers],
            "verify":    [t["mean_verify_ms"]     for t in tiers],
            "decrypt":   [t["mean_decrypt_ms"]    for t in tiers],
        },
        "escalated_record_fraction":        ps.get("escalated_record_fraction", 0.0),
        "escalated_plaintext_byte_fraction": ps.get("escalated_plaintext_byte_fraction", 0.0),
        "overall_amplification":            ps.get("overall_storage_amplification", 0.0),
        "policy_mode":                      ps.get("policy_mode", "n/a"),
        "threshold_bytes":                  ps.get("threshold_bytes"),
    }


# Colour palette for the two policy tiers.
_TIER_COLORS = {
    "baseline": "#4C72B0",
    "strong":   "#DD8452",
    "n/a":      "#8B8B8B",
}

_COMPONENT_COLORS = {
    "encrypt":   "#4878CF",
    "sign":      "#D65F5F",
    "db_insert": "#6ACC65",
    "db_fetch":  "#B47CC7",
    "verify":    "#C4AD66",
    "decrypt":   "#77BEDB",
}


def plot_policy_tier_summary(
    data: dict,
    out_dir: Path,
    *,
    output_name: str = "policy_tier_summary.png",
) -> None:
    """Produce a 2×2 figure summarising the adaptive policy tier breakdown.

    Panels:
      (top-left)  Record count and byte fraction by tier — grouped bar chart.
      (top-right) Storage amplification by tier — bar chart.
      (bottom-left) End-to-end latency (mean + p95) by tier — bar chart.
      (bottom-right) Time composition by tier — stacked bar chart.
    """
    ps = data.get("policy_summary")
    if not ps or not ps.get("tier_summaries"):
        print("  Skipping policy_tier_summary.png — no policy_summary in data.")
        return

    cd = extract_policy_chart_data(data)
    tiers = cd["tier_labels"]
    if not tiers:
        return

    colors = [_TIER_COLORS.get(t, "#8B8B8B") for t in tiers]
    x = np.arange(len(tiers))
    bar_w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    policy_mode = cd["policy_mode"] or "n/a"
    fig.suptitle(f"Policy Tier Summary — {policy_mode}", fontsize=13, y=1.01)

    # ── Panel 1: record and byte coverage fractions ───────────────────────────
    ax = axes[0, 0]
    ax.bar(x - bar_w / 2, [v * 100 for v in cd["record_fractions"]],
           bar_w, color=colors, alpha=0.85, label="Records")
    ax.bar(x + bar_w / 2, [v * 100 for v in cd["plaintext_byte_fractions"]],
           bar_w, color=colors, alpha=0.50, hatch="//", label="Plaintext bytes")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Fraction (%)")
    ax.set_title("Record and byte coverage")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 115)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel 2: storage amplification ───────────────────────────────────────
    ax = axes[0, 1]
    bars = ax.bar(x, cd["storage_amplifications"], 0.5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Amplification (×)")
    ax.set_title("Storage amplification by tier")
    for bar, v in zip(bars, cd["storage_amplifications"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}×", ha="center", va="bottom", fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel 3: E2E latency (mean + p95) ────────────────────────────────────
    ax = axes[1, 0]
    ax.bar(x - bar_w / 2, cd["mean_e2e_ms"], bar_w, color=colors, alpha=0.85, label="Mean")
    ax.bar(x + bar_w / 2, cd["p95_e2e_ms"],  bar_w, color=colors, alpha=0.45, label="P95")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-end latency by tier")
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel 4: time composition ─────────────────────────────────────────────
    ax = axes[1, 1]
    components = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt"]
    bottoms = np.zeros(len(tiers))
    for comp in components:
        vals = np.array(cd["component_ms"][comp])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, 0.5, bottom=bottoms,
               color=_COMPONENT_COLORS.get(comp, "#AAAAAA"),
               alpha=0.85, label=comp)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Time composition by tier")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = out_dir / output_name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
