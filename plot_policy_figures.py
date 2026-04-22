"""
plot_policy_figures.py
======================
Policy-comparison visualization for the PQC per-record-policy benchmark.

Research question
-----------------
Can a per-record workload-aware PQ parameter assignment policy protect larger
biomedical records (genomic data) with stronger standardized PQ parameters
while keeping system cost near the Category-3 baseline?

Generated figures
-----------------
01_cost_vs_coverage.png              — main figure: latency & storage cost vs byte coverage
02_escalation_coverage_bars.png      — escalated records vs escalated bytes (adaptive only)
03_policy_cost_comparison.png        — latency and storage amplification by policy mode
04_record_size_vs_tier.png           — per-record tier assignment under adaptive policy
05_per_tier_latency_adaptive.png     — baseline-tier vs strong-tier latency inside adaptive
06_time_composition.png              — stacked component latencies by policy mode

Usage
-----
python plot_policy_figures.py \\
    --pilot-dir  results/policy_pilot/20260417_011356/json \\
    --output-dir results/policy_pilot/20260417_011356/policy_figures

The script auto-discovers files matching {workload}_{policy_suffix}.json:
    workload      ∈ {synthetic, faers, ggvp}
    policy_suffix ∈ {uniform_baseline, adaptive_threshold, uniform_strong}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     150,
})

# ── Catalogue ─────────────────────────────────────────────────────────────────

WORKLOADS = ["synthetic", "faers", "ggvp"]
WORKLOAD_LABELS = {"synthetic": "Synthetic", "faers": "FAERS", "ggvp": "GGVP"}

POLICY_SUFFIXES = {
    "baseline": "uniform_baseline",
    "adaptive": "adaptive_threshold",
    "strong":   "uniform_strong",
}
POLICY_ORDER  = ["baseline", "adaptive", "strong"]
POLICY_SHORT  = {"baseline": "Baseline", "adaptive": "Adaptive", "strong": "Strong"}
POLICY_LABELS = {
    "baseline": "Baseline  (ML-KEM-768 / ML-DSA-65)",
    "adaptive": "Adaptive  (10 MiB threshold)",
    "strong":   "Strong    (ML-KEM-1024 / ML-DSA-87)",
}


def _adaptive_label_from_threshold(threshold_bytes: int | None) -> str:
    """Human-readable adaptive policy label derived from the threshold value."""
    if threshold_bytes is None:
        return "Adaptive  (threshold)"
    if threshold_bytes >= 1024 * 1024:
        return f"Adaptive  ({threshold_bytes // (1024 * 1024)} MiB threshold)"
    if threshold_bytes >= 1024:
        return f"Adaptive  ({threshold_bytes // 1024} KiB threshold)"
    return f"Adaptive  ({threshold_bytes} B threshold)"

# Colour / marker palettes — kept consistent across every figure
WORKLOAD_COLORS  = {"synthetic": "#9B59B6", "faers": "#3498DB", "ggvp": "#27AE60"}
WORKLOAD_MARKERS = {"synthetic": "o",       "faers": "s",       "ggvp": "D"}
POLICY_COLORS    = {"baseline": "#4C72B0",  "adaptive": "#55A868", "strong": "#C44E52"}
TIER_COLORS      = {"baseline": "#4C72B0",  "strong": "#C44E52"}

COMPONENT_ORDER  = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt"]
COMPONENT_LABELS = {
    "encrypt":   "Encrypt",   "sign":      "Sign",
    "db_insert": "DB Insert", "db_fetch":  "DB Fetch",
    "verify":    "Verify",    "decrypt":   "Decrypt",
}
COMPONENT_COLORS = {
    "encrypt":   "#4878CF", "sign":      "#D65F5F",
    "db_insert": "#6ACC65", "db_fetch":  "#B47CC7",
    "verify":    "#C4AD66", "decrypt":   "#77BEDB",
}

DEFAULT_THRESHOLD_BYTES = 10 * 1024 * 1024   # 10 MiB

# ── Data helpers ──────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_pilot_set(pilot_dir: Path) -> dict[str, dict[str, dict]]:
    """
    Discover and load all policy-pilot JSON files.
    Returns  data[workload][policy_key] = loaded JSON dict.
    """
    pilot: dict[str, dict[str, dict]] = {wl: {} for wl in WORKLOADS}
    found, missing = [], []
    for wl in WORKLOADS:
        for policy_key, suffix in POLICY_SUFFIXES.items():
            path = pilot_dir / f"{wl}_{suffix}.json"
            if path.exists():
                pilot[wl][policy_key] = _load(path)
                found.append(path.name)
            else:
                missing.append(path.name)
    print(f"  Loaded {len(found)} pilot files.")
    for m in missing:
        print(f"  WARNING: pilot file not found: {m}")
    return pilot


def _ps(data: dict | None) -> dict:
    """Return the policy_summary dict, or empty dict."""
    return (data or {}).get("policy_summary") or {}


def _tier_summary(ps: dict, tier_label: str) -> dict | None:
    for t in ps.get("tier_summaries", []):
        if t.get("tier_label") == tier_label:
            return t
    return None


def _weighted_component_ms(ps: dict, comp: str) -> float:
    """Weighted mean of mean_{comp}_ms across all tier summaries."""
    tiers = ps.get("tier_summaries", [])
    total = sum(t.get("record_count", 0) for t in tiers)
    if total == 0:
        return 0.0
    key = f"mean_{comp}_ms"
    return sum(t.get(key, 0.0) * t.get("record_count", 0) for t in tiers) / total


# ── Visual 1: Cost vs Coverage scatter ───────────────────────────────────────

def plot_cost_vs_coverage(pilot: dict, out_dir: Path) -> None:
    """
    Main figure.
    Left panel  — x: fraction of plaintext bytes under strong protection,
                  y: mean end-to-end latency (ms).
    Right panel — same x, y: overall storage amplification.

    One trajectory line per workload (baseline→adaptive→strong).
    Colour encodes policy mode; marker shape encodes workload.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for wl in WORKLOADS:
        wl_data = pilot.get(wl, {})
        pts_lat: list[tuple[float, float, str]] = []
        pts_amp: list[tuple[float, float, str]] = []

        for pk in POLICY_ORDER:
            ps = _ps(wl_data.get(pk))
            if not ps:
                continue
            x_cov = ps.get("escalated_plaintext_byte_fraction", 0.0)
            pts_lat.append((x_cov, ps.get("mean_end_to_end_ms", 0.0),         pk))
            pts_amp.append((x_cov, ps.get("overall_storage_amplification", 1.0), pk))

        for ax, pts in zip(axes, [pts_lat, pts_amp]):
            if len(pts) >= 2:
                ax.plot(
                    [p[0] for p in pts], [p[1] for p in pts],
                    "-", color=WORKLOAD_COLORS[wl], alpha=0.30, linewidth=1.4, zorder=1,
                )
            for (x, y, pk) in pts:
                ax.scatter(
                    x, y,
                    marker=WORKLOAD_MARKERS[wl], s=120,
                    color=POLICY_COLORS[pk],
                    edgecolors=WORKLOAD_COLORS[wl], linewidths=1.8,
                    zorder=5,
                )
            # Annotate workload name next to the baseline point (x≈0)
            baseline_pts = [(x, y) for (x, y, pk) in pts if pk == "baseline"]
            if baseline_pts:
                bx, by = baseline_pts[0]
                ax.annotate(
                    WORKLOAD_LABELS[wl], (bx, by),
                    xytext=(-6, 6), textcoords="offset points",
                    fontsize=8, color=WORKLOAD_COLORS[wl],
                    ha="right",
                )

    # Latency panel
    ax = axes[0]
    ax.set_xlabel("Fraction of Plaintext Bytes Under Strong Protection")
    ax.set_ylabel("Mean End-to-End Latency (ms)")
    ax.set_title("Policy Cost vs Strong-Protection Byte Coverage\n(latency)")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Storage panel
    ax = axes[1]
    ax.set_xlabel("Fraction of Plaintext Bytes Under Strong Protection")
    ax.set_ylabel("Overall Storage Amplification (×)")
    ax.set_title("Policy Cost vs Strong-Protection Byte Coverage\n(storage)")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Two legends: workload shapes (left panel) + policy colours (right panel)
    wl_handles = [
        Line2D([0], [0], marker=WORKLOAD_MARKERS[wl], color=WORKLOAD_COLORS[wl],
               markerfacecolor=WORKLOAD_COLORS[wl], markersize=8,
               label=WORKLOAD_LABELS[wl], linewidth=0)
        for wl in WORKLOADS
    ]
    pol_handles = [
        Line2D([0], [0], marker="o", color=POLICY_COLORS[pk],
               markerfacecolor=POLICY_COLORS[pk], markersize=8,
               label=POLICY_SHORT[pk], linewidth=0)
        for pk in POLICY_ORDER
    ]
    axes[0].legend(handles=wl_handles,  title="Workload (shape)",  loc="upper left",  fontsize=9)
    axes[1].legend(handles=pol_handles, title="Policy (colour)", loc="lower right", fontsize=9)

    fig.suptitle(
        "Adaptive Policy: High Byte Coverage at Near-Baseline Cost\n"
        "(trajectory = one workload; colour = policy; shape = workload)",
        fontsize=13,
    )
    fig.tight_layout()
    path = out_dir / "01_cost_vs_coverage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Visual 2: Escalated records vs escalated bytes ───────────────────────────

def plot_escalation_coverage_bars(pilot: dict, out_dir: Path) -> None:
    """
    Adaptive policy only.
    For each workload: grouped bar — escalated_record_fraction vs
    escalated_plaintext_byte_fraction.
    Exposes the key insight: a small fraction of records holds the vast
    majority of bytes, so adaptive selectively upgrades most bytes cheaply.
    """
    wls = [wl for wl in WORKLOADS if _ps(pilot.get(wl, {}).get("adaptive"))]
    if not wls:
        print("  WARNING: no adaptive data — skipping escalation coverage chart.")
        return

    x = np.arange(len(wls))
    bar_w = 0.35

    rec_pct  = [_ps(pilot[wl]["adaptive"]).get("escalated_record_fraction",          0.0) * 100 for wl in wls]
    byte_pct = [_ps(pilot[wl]["adaptive"]).get("escalated_plaintext_byte_fraction",  0.0) * 100 for wl in wls]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [WORKLOAD_COLORS[wl] for wl in wls]

    bars_rec  = ax.bar(x - bar_w / 2, rec_pct,  bar_w, color=colors, alpha=0.88,
                       label="Escalated records")
    bars_byte = ax.bar(x + bar_w / 2, byte_pct, bar_w, color=colors, alpha=0.45,
                       hatch="//", label="Escalated plaintext bytes")

    for bar, v in zip(list(bars_rec) + list(bars_byte), rec_pct + byte_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_LABELS[wl] for wl in wls])
    ax.set_ylabel("Fraction Escalated to Strong Tier (%)")
    ax.set_ylim(0, 115)
    ax.set_title(
        "Adaptive Policy: Escalated Records vs Escalated Plaintext Bytes\n"
        "(a small share of records accounts for nearly all protected bytes)"
    )
    ax.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "02_escalation_coverage_bars.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Visual 3: Policy comparison cost bars ─────────────────────────────────────

def plot_policy_cost_bars(pilot: dict, out_dir: Path) -> None:
    """
    Two-panel bar chart comparing all three policy modes across workloads.
    Left panel:  mean end-to-end latency (ms).
    Right panel: overall storage amplification.
    """
    wls = [wl for wl in WORKLOADS if pilot.get(wl)]
    if not wls:
        print("  WARNING: no pilot data — skipping policy cost comparison.")
        return

    x      = np.arange(len(wls))
    bar_w  = 0.22
    offsets = np.linspace(-(len(POLICY_ORDER) - 1) / 2,
                           (len(POLICY_ORDER) - 1) / 2,
                           len(POLICY_ORDER)) * bar_w

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric_key, ylabel, title in [
        (axes[0], "mean_end_to_end_ms",           "Mean End-to-End Latency (ms)",     "End-to-End Latency by Policy Mode"),
        (axes[1], "overall_storage_amplification", "Overall Storage Amplification (×)", "Storage Amplification by Policy Mode"),
    ]:
        for pi, pk in enumerate(POLICY_ORDER):
            vals = [
                _ps(pilot.get(wl, {}).get(pk)).get(metric_key, 0.0)
                for wl in wls
            ]
            bars = ax.bar(x + offsets[pi], vals, bar_w * 0.88,
                          label=POLICY_SHORT[pk], color=POLICY_COLORS[pk], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v == 0:
                    continue
                fmt = f"{v:.4f}×" if "amplif" in metric_key else f"{v:.0f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        fmt, ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([WORKLOAD_LABELS[wl] for wl in wls])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    fig.suptitle(
        "System Cost by Policy Mode\n"
        "(does adaptive stay close to baseline? how much cheaper than strong?)",
        fontsize=13,
    )
    fig.tight_layout()
    path = out_dir / "03_policy_cost_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Visual 4: Record size vs assigned tier ────────────────────────────────────

def plot_record_size_vs_tier(pilot: dict, out_dir: Path) -> None:
    """
    Per-record scatter: x = record size (log scale), y = assigned tier.
    Adaptive policy only.  One subplot per workload.
    Vertical dashed line marks the escalation threshold.
    """
    wls = [wl for wl in WORKLOADS if pilot.get(wl, {}).get("adaptive")]
    if not wls:
        print("  WARNING: no adaptive data — skipping record-size-vs-tier chart.")
        return

    fig, axes = plt.subplots(1, len(wls), figsize=(5 * len(wls), 4), sharey=False)
    if len(wls) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, wl in zip(axes, wls):
        results = pilot[wl]["adaptive"].get("results", [])
        if not results:
            ax.set_visible(False)
            continue

        threshold = (pilot[wl]["adaptive"].get("policy_threshold_bytes")
                     or DEFAULT_THRESHOLD_BYTES)
        thresh_mib = threshold / (1024 * 1024)
        thresh_label = (f"{threshold // (1024*1024)} MiB" if threshold >= 1024*1024
                        else f"{threshold // 1024} KiB")

        sizes_mib = [r["size_bytes"] / (1024 * 1024) for r in results]
        x_min = min(sizes_mib) * 0.4
        x_max = max(sizes_mib) * 2.5

        # Downsample for scatter rendering performance (max 5000 points per tier)
        _MAX_PTS = 5000
        from collections import defaultdict
        by_tier: dict[str, list] = defaultdict(list)
        for r in results:
            by_tier[r.get("tier_label", "baseline")].append(r)
        sampled: list[dict] = []
        for tier_recs in by_tier.values():
            if len(tier_recs) > _MAX_PTS:
                idx = rng.choice(len(tier_recs), size=_MAX_PTS, replace=False)
                sampled.extend(tier_recs[i] for i in idx)
            else:
                sampled.extend(tier_recs)

        tier_y = {"baseline": 0, "strong": 1}
        sizes_x = np.array([r["size_bytes"] / (1024 * 1024) for r in sampled])
        tiers_v = np.array([tier_y.get(r.get("tier_label", "baseline"), 0) for r in sampled])
        jitter  = rng.uniform(-0.10, 0.10, size=len(sampled))
        colors_v = [TIER_COLORS.get(r.get("tier_label", "baseline"), "#888888") for r in sampled]
        ax.scatter(
            sizes_x, tiers_v + jitter,
            marker=WORKLOAD_MARKERS[wl], s=65,
            c=colors_v, alpha=0.60, zorder=4,
        )

        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)

        # Threshold line and region shading
        ax.axvline(thresh_mib, color="#E74C3C", linewidth=1.5, linestyle="--",
                   label=f"Threshold  {thresh_label}", zorder=3)
        ax.axvspan(x_min,      thresh_mib, alpha=0.06, color=TIER_COLORS["baseline"], zorder=0)
        ax.axvspan(thresh_mib, x_max,      alpha=0.06, color=TIER_COLORS["strong"],   zorder=0)

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: (
                f"{v * 1024:.0f} KB" if v < 1 else
                (f"{v:.0f} MiB" if v >= 10 else f"{v:.1f} MiB")
            )
        ))
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Baseline\n(ML-KEM-768)", "Strong\n(ML-KEM-1024)"])
        ax.set_xlabel("Record Size")
        ax.set_title(WORKLOAD_LABELS[wl])
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.30)
        ax.set_axisbelow(True)

    # Shared tier legend
    tier_handles = [
        Line2D([0], [0], marker="o", color=TIER_COLORS[t], markerfacecolor=TIER_COLORS[t],
               markersize=8, label=t.capitalize(), linewidth=0)
        for t in ["baseline", "strong"]
    ]
    axes[-1].legend(handles=tier_handles, title="Assigned tier", fontsize=9, loc="lower right")

    fig.suptitle(
        "Adaptive Policy Assignment by Record Size\n"
        "(records right of threshold receive ML-KEM-1024 / ML-DSA-87)",
        fontsize=13,
    )
    fig.tight_layout()
    path = out_dir / "04_record_size_vs_tier.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Visual 5: Per-tier latency inside adaptive ────────────────────────────────

def plot_per_tier_latency(pilot: dict, out_dir: Path) -> None:
    """
    Within adaptive_threshold: baseline-tier vs strong-tier mean E2E latency.
    Log-scale y-axis to show the order-of-magnitude difference clearly.
    """
    wls = [wl for wl in WORKLOADS if _ps(pilot.get(wl, {}).get("adaptive")).get("tier_summaries")]
    if not wls:
        print("  WARNING: no adaptive tier data — skipping per-tier latency chart.")
        return

    x = np.arange(len(wls))
    bar_w = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, tier in enumerate(["baseline", "strong"]):
        offset = (i - 0.5) * bar_w
        vals = []
        for wl in wls:
            t = _tier_summary(_ps(pilot[wl]["adaptive"]), tier)
            vals.append(t["mean_end_to_end_ms"] if t else 0.0)

        bars = ax.bar(x + offset, vals, bar_w * 0.88,
                      label=f"{tier.capitalize()} tier",
                      color=TIER_COLORS[tier], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08,
                        f"{v:,.0f} ms", ha="center", va="bottom", fontsize=9)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_LABELS[wl] for wl in wls])
    ax.set_ylabel("Mean End-to-End Latency (ms, log scale)")
    ax.set_title(
        "Per-Tier Latency Under Adaptive Policy\n"
        "(strong-tier records are larger — higher latency is expected)"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "05_per_tier_latency_adaptive.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Visual 6: Stacked time composition ────────────────────────────────────────

def plot_time_composition(pilot: dict, out_dir: Path) -> None:
    """
    Stacked bar chart of mean component latencies.
    One subplot per workload; three bars per subplot (one per policy mode).
    Component means are weighted across tier summaries.
    """
    wls = [wl for wl in WORKLOADS if pilot.get(wl)]
    if not wls:
        return

    fig, axes = plt.subplots(1, len(wls), figsize=(5 * len(wls), 6), sharey=False)
    if len(wls) == 1:
        axes = [axes]

    x = np.arange(len(POLICY_ORDER))

    for ax, wl in zip(axes, wls):
        bottoms = np.zeros(len(POLICY_ORDER))
        legend_drawn = False

        for comp in COMPONENT_ORDER:
            vals = np.array([
                _weighted_component_ms(_ps(pilot.get(wl, {}).get(pk)), comp)
                for pk in POLICY_ORDER
            ])
            if vals.sum() == 0:
                continue
            ax.bar(
                x, vals, 0.55, bottom=bottoms,
                color=COMPONENT_COLORS[comp], alpha=0.88,
                label=COMPONENT_LABELS[comp],
            )
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_SHORT[pk] for pk in POLICY_ORDER], rotation=15, ha="right")
        ax.set_ylabel("Mean Latency (ms)")
        ax.set_title(WORKLOAD_LABELS[wl])
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Time Composition by Policy Mode\n"
        "(weighted mean across all records; stacked = latency components)",
        fontsize=13,
    )
    fig.tight_layout()
    path = out_dir / "06_time_composition.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate policy-comparison figures for the PQC benchmark."
    )
    p.add_argument(
        "--pilot-dir",
        default="results/policy_pilot/20260417_011356/json",
        help="Directory containing the 9 policy-pilot JSON files.",
    )
    p.add_argument(
        "--output-dir",
        default="results/policy_pilot/20260417_011356/policy_figures",
        help="Output directory for generated figures.",
    )
    p.add_argument(
        "--faers-adaptive-override",
        metavar="PATH",
        default=None,
        help=(
            "Path to an alternative FAERS adaptive JSON file.  "
            "When provided, replaces the FAERS adaptive entry loaded from --pilot-dir.  "
            "Baseline and strong are still sourced from --pilot-dir."
        ),
    )
    args = p.parse_args()

    pilot_dir = Path(args.pilot_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pilot_dir.exists():
        print(f"ERROR: --pilot-dir does not exist: {pilot_dir}")
        return

    print(f"\n  Loading pilot data from {pilot_dir}")
    pilot = load_pilot_set(pilot_dir)

    if args.faers_adaptive_override:
        override_path = Path(args.faers_adaptive_override)
        if not override_path.exists():
            print(f"ERROR: --faers-adaptive-override path does not exist: {override_path}")
            return
        print(f"  Overriding FAERS adaptive with: {override_path}")
        override_data = _load(override_path)
        pilot.setdefault("faers", {})["adaptive"] = override_data
        threshold_bytes = override_data.get("policy_threshold_bytes") or (
            override_data.get("policy_summary", {}).get("threshold_bytes")
        )
        POLICY_LABELS["adaptive"] = _adaptive_label_from_threshold(threshold_bytes)

    print(f"  Writing figures to  {out_dir}/\n")

    funcs = [
        plot_cost_vs_coverage,
        plot_escalation_coverage_bars,
        plot_policy_cost_bars,
        plot_record_size_vs_tier,
        plot_per_tier_latency,
        plot_time_composition,
    ]
    n_saved = 0
    for fn in funcs:
        try:
            fn(pilot, out_dir)
            n_saved += 1
        except Exception as exc:
            print(f"  ERROR in {fn.__name__}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\n  Done. {n_saved}/{len(funcs)} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
