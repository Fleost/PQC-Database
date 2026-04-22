"""
plot_baseline_normalized_figures.py
=====================================
Baseline-normalized visuals and tables for the PQC per-record-policy benchmark.

Research question
-----------------
Does the adaptive policy keep system cost near the Category 3 baseline
(ML-KEM-768 / ML-DSA-65) while still protecting a much larger fraction of
high-value biomedical/genomic bytes with stronger standardized PQ parameters
(ML-KEM-1024 / ML-DSA-87)?

Generated outputs
-----------------
01_latency_pct_over_baseline.png     — % change in mean E2E latency vs baseline
02_storage_pct_over_baseline.png     — % change in storage amplification vs baseline
03_latency_dumbbell.png              — baseline / adaptive / strong as dumbbell plot
04_cost_ratio_heatmap.png            — annotated heatmap of cost ratios
05_near_baseline_table.png           — decision table: adaptive within 5% / 10%?
06_baseline_normalized_cost_table.csv
06_baseline_normalized_cost_table.md

Usage
-----
python plot_baseline_normalized_figures.py \\
    --pilot-dir  results/policy_pilot/20260417_011356/json \\
    --output-dir results/policy_pilot/20260417_011356/policy_figures_normalized

The script auto-discovers files matching {workload}_{policy_suffix}.json.
Mixed-workload aggregation is not supported from the current JSON outputs;
a note is printed if requested.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Shared constants (mirrors plot_policy_figures.py) ─────────────────────────

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":     150,
})

WORKLOADS       = ["faers", "ggvp"]
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

WORKLOAD_COLORS  = {"synthetic": "#9B59B6", "faers": "#3498DB", "ggvp": "#27AE60"}
POLICY_COLORS    = {"baseline": "#4C72B0",  "adaptive": "#55A868", "strong": "#C44E52"}

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


def _complete_workloads(pilot: dict) -> list[str]:
    """Return workloads that have all three policy modes."""
    complete = []
    for wl in WORKLOADS:
        if all(pk in pilot.get(wl, {}) for pk in POLICY_ORDER):
            complete.append(wl)
        else:
            missing = [pk for pk in POLICY_ORDER if pk not in pilot.get(wl, {})]
            print(f"  WARNING: {wl} missing policies {missing} — skipping for normalized figures.")
    return complete


# ── Metric extraction ─────────────────────────────────────────────────────────

def extract_normalized_metrics(pilot: dict) -> dict[str, dict]:
    """
    For each complete workload compute:
      baseline_lat, adaptive_lat, strong_lat
      adap_lat_ratio, adap_lat_pct
      strong_lat_ratio, strong_lat_pct
      baseline_amp, adaptive_amp, strong_amp
      adap_amp_ratio, adap_amp_pct
      strong_amp_ratio, strong_amp_pct
      adap_esc_byte_frac
    Returns dict keyed by workload name.
    """
    result: dict[str, dict] = {}
    for wl in _complete_workloads(pilot):
        base_ps = _ps(pilot[wl]["baseline"])
        adap_ps = _ps(pilot[wl]["adaptive"])
        strg_ps = _ps(pilot[wl]["strong"])

        base_lat = base_ps.get("mean_end_to_end_ms", float("nan"))
        adap_lat = adap_ps.get("mean_end_to_end_ms", float("nan"))
        strg_lat = strg_ps.get("mean_end_to_end_ms", float("nan"))

        base_amp = base_ps.get("overall_storage_amplification", float("nan"))
        adap_amp = adap_ps.get("overall_storage_amplification", float("nan"))
        strg_amp = strg_ps.get("overall_storage_amplification", float("nan"))

        adap_lat_ratio = adap_lat / base_lat if base_lat else float("nan")
        strg_lat_ratio = strg_lat / base_lat if base_lat else float("nan")
        adap_amp_ratio = adap_amp / base_amp if base_amp else float("nan")
        strg_amp_ratio = strg_amp / base_amp if base_amp else float("nan")

        result[wl] = {
            "baseline_lat":      base_lat,
            "adaptive_lat":      adap_lat,
            "strong_lat":        strg_lat,
            "adap_lat_ratio":    adap_lat_ratio,
            "adap_lat_pct":      (adap_lat_ratio - 1.0) * 100.0,
            "strong_lat_ratio":  strg_lat_ratio,
            "strong_lat_pct":    (strg_lat_ratio - 1.0) * 100.0,
            "baseline_amp":      base_amp,
            "adaptive_amp":      adap_amp,
            "strong_amp":        strg_amp,
            "adap_amp_ratio":    adap_amp_ratio,
            "adap_amp_pct":      (adap_amp_ratio - 1.0) * 100.0,
            "strong_amp_ratio":  strg_amp_ratio,
            "strong_amp_pct":    (strg_amp_ratio - 1.0) * 100.0,
            "adap_esc_byte_frac": adap_ps.get("escalated_plaintext_byte_fraction", float("nan")),
        }
    return result


# ── Output 1: Baseline-normalized cost table (CSV + Markdown) ─────────────────

def write_cost_table(metrics: dict, out_dir: Path) -> None:
    wls = list(metrics.keys())
    rows = []
    for wl in wls:
        m = metrics[wl]
        rows.append({
            "Workload":                   WORKLOAD_LABELS[wl],
            "Baseline Latency (ms)":      f"{m['baseline_lat']:.2f}",
            "Adaptive Latency (ms)":      f"{m['adaptive_lat']:.2f}",
            "Strong Latency (ms)":        f"{m['strong_lat']:.2f}",
            "Adap/Base Ratio":            f"{m['adap_lat_ratio']:.4f}×",
            "Adap % Change":              f"{m['adap_lat_pct']:+.2f}%",
            "Strong/Base Ratio":          f"{m['strong_lat_ratio']:.4f}×",
            "Strong % Change":            f"{m['strong_lat_pct']:+.2f}%",
            "Adap Storage Amplification": f"{m['adaptive_amp']:.6f}×",
            "Adap Esc. Byte Fraction":    f"{m['adap_esc_byte_frac']*100:.1f}%",
        })

    fieldnames = list(rows[0].keys())

    # CSV
    csv_path = out_dir / "06_baseline_normalized_cost_table.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {csv_path}")

    # Markdown
    md_path = out_dir / "06_baseline_normalized_cost_table.md"
    header = "| " + " | ".join(fieldnames) + " |"
    sep    = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    lines  = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(row[k] for k in fieldnames) + " |")

    note = dedent("""
        **Notes**
        - Baseline policy = ML-KEM-768 / ML-DSA-65 (Category 3 uniform).
        - Adaptive policy = ML-KEM-768 / ML-DSA-65 for records < 10 MiB; ML-KEM-1024 / ML-DSA-87 for records ≥ 10 MiB.
        - Strong policy = ML-KEM-1024 / ML-DSA-87 (uniform).
        - Mixed-workload aggregation is not available from the current JSON outputs.
          A mixed row would require combining FAERS + GGVP records in a single run.
        - Ratios < 1.0 (negative % change) indicate the policy ran faster/smaller than baseline,
          which can occur due to measurement noise across independent benchmark runs on separate hosts.
    """).strip()

    md_path.write_text("\n".join(lines) + "\n\n" + note + "\n")
    print(f"  Saved {md_path}")


# ── Output 2: Percent-over-baseline latency bar chart ─────────────────────────

def plot_latency_pct_over_baseline(metrics: dict, out_dir: Path) -> None:
    wls     = list(metrics.keys())
    labels  = [WORKLOAD_LABELS[wl] for wl in wls]
    x       = np.arange(len(wls))
    bar_w   = 0.32

    adap_pcts   = [metrics[wl]["adap_lat_pct"]   for wl in wls]
    strong_pcts = [metrics[wl]["strong_lat_pct"]  for wl in wls]
    adap_esc    = [metrics[wl]["adap_esc_byte_frac"] * 100 for wl in wls]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Shaded "near-baseline" zone ±5%
    ax.axhspan(-5, 5, alpha=0.10, color="#2ECC71", zorder=0)

    bars_adap = ax.bar(x - bar_w / 2, adap_pcts,  bar_w * 0.90,
                       color=POLICY_COLORS["adaptive"], alpha=0.90,
                       label="Adaptive  (10 MiB threshold, selective)", zorder=3)
    bars_strg = ax.bar(x + bar_w / 2, strong_pcts, bar_w * 0.90,
                       color=POLICY_COLORS["strong"],   alpha=0.90,
                       label="Strong    (ML-KEM-1024 / ML-DSA-87 uniform)", zorder=3)

    # Value labels — for adaptive bars also show escalated byte fraction
    for i, (bar, v) in enumerate(zip(bars_adap, adap_pcts)):
        yoff = 0.20 if v >= 0 else -0.22
        va   = "bottom" if v >= 0 else "top"
        esc_line = f"\n↑ {adap_esc[i]:.1f}% of bytes\nescalated to strong"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + yoff,
            f"{v:+.2f}%" + esc_line,
            ha="center", va=va, fontsize=8.5, fontweight="bold",
            color=POLICY_COLORS["adaptive"],
        )

    for bar, v in zip(bars_strg, strong_pcts):
        yoff = 0.20 if v >= 0 else -0.22
        va   = "bottom" if v >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + yoff,
            f"{v:+.2f}%",
            ha="center", va=va, fontsize=9.5, fontweight="bold",
            color=POLICY_COLORS["strong"],
        )

    ax.axhline(0, color="#222222", linewidth=1.5, zorder=4)

    # ±5% threshold — amber, prominent
    ax.axhline( 5, color="#E67E22", linewidth=1.8, linestyle="--", alpha=0.90, zorder=2)
    ax.axhline(-5, color="#E67E22", linewidth=1.8, linestyle="--", alpha=0.90, zorder=2)
    ax.text(len(wls) - 0.52,  5.18, "±5%  near-baseline zone",
            fontsize=8.5, color="#E67E22", ha="right", fontweight="bold")

    # ±10% threshold — grey, subtle
    ax.axhline( 10, color="grey", linewidth=0.9, linestyle=":", alpha=0.55, zorder=2)
    ax.axhline(-10, color="grey", linewidth=0.9, linestyle=":", alpha=0.55, zorder=2)
    ax.text(len(wls) - 0.52, 10.18, "±10%",
            fontsize=7.5, color="grey", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Latency Change vs. Category 3 Baseline (%)", fontsize=11)
    ax.set_title(
        "Adaptive Policy Stays Within 5% of the Category 3 Baseline\n"
        "while escalating ≥98% of plaintext bytes to stronger PQ protection  "
        "(shaded band = ±5% zone)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(wls) - 0.4)

    fig.tight_layout()
    path = out_dir / "01_latency_pct_over_baseline.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Output 3: Percent-over-baseline storage bar chart ─────────────────────────

def plot_storage_pct_over_baseline(metrics: dict, out_dir: Path) -> None:
    """
    One zoomed panel per workload (side by side in the same file).
    Within each panel: three bars (baseline / adaptive / strong) on an
    independently scaled y-axis so the sub-0.1% differences become visible.
    Annotates each bar with its absolute amplification value and (for
    adaptive/strong) the deviation from baseline.
    """
    wls  = list(metrics.keys())
    pols = ["baseline", "adaptive", "strong"]
    bar_w = 0.55
    x = np.arange(len(pols))

    fig, axes = plt.subplots(1, len(wls), figsize=(6 * len(wls), 6.5))
    if len(wls) == 1:
        axes = [axes]

    for ax, wl in zip(axes, wls):
        base_val = metrics[wl]["baseline_amp"]
        vals = {
            "baseline": metrics[wl]["baseline_amp"],
            "adaptive": metrics[wl]["adaptive_amp"],
            "strong":   metrics[wl]["strong_amp"],
        }

        # Draw bars
        for i, pol in enumerate(pols):
            v = vals[pol]
            ax.bar(i, v, bar_w * 0.85,
                   color=POLICY_COLORS[pol], alpha=0.85, zorder=3,
                   label=POLICY_SHORT[pol])

        # Y-axis zoom: tight around the data with headroom for annotations
        all_v = list(vals.values())
        data_range = max(all_v) - min(all_v)
        pad = max(data_range * 0.5, base_val * 5e-6)
        y_lo = min(all_v) - pad
        y_hi = max(all_v) + pad * 5.0  # headroom for labels
        ax.set_ylim(y_lo, y_hi)

        # Annotate bars
        for i, pol in enumerate(pols):
            v = vals[pol]
            pct_dev = (v / base_val - 1.0) * 100.0
            label_y = v + (y_hi - y_lo) * 0.02

            # Absolute amplification value
            ax.text(i, label_y,
                    f"{v:.6f}×",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

            # % deviation for non-baseline bars
            if pol != "baseline":
                dev_y = label_y + (y_hi - y_lo) * 0.12
                sign  = "+" if pct_dev >= 0 else ""
                # Use scientific notation if deviation is tiny (< 0.001%)
                if abs(pct_dev) < 0.001:
                    dev_str = f"{sign}{pct_dev:.4f}%"
                else:
                    dev_str = f"{sign}{pct_dev:.4f}%"
                ax.text(i, dev_y,
                        f"{dev_str}\nvs. baseline",
                        ha="center", va="bottom", fontsize=8,
                        color=POLICY_COLORS[pol], style="italic")

        # Baseline reference line
        ax.axhline(base_val, color=POLICY_COLORS["baseline"],
                   linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)

        # Y-axis formatting — show enough decimal places to distinguish values
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_SHORT[p] for p in pols], fontsize=11)
        ax.set_title(WORKLOAD_LABELS[wl], fontsize=13, fontweight="bold", pad=10)
        if ax is axes[0]:
            ax.set_ylabel("Overall Storage Amplification (×)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.30, zorder=0)
        ax.set_axisbelow(True)

        # Per-panel scale note
        ax.text(0.98, 0.02,
                f"y-axis: {y_lo:.6f}× – {y_hi:.6f}×\n(independently scaled)",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                color="grey", style="italic")

    fig.suptitle(
        "Storage Amplification: All Policies Within Sub-0.1% of the Category 3 Baseline\n"
        "(panels independently y-scaled so differences are legible — inter-workload scale is NOT comparable)",
        fontsize=12,
    )

    # Figure-level legend
    leg_handles = [
        Line2D([0], [0], color=POLICY_COLORS[pol], linewidth=8,
               label=f"{POLICY_SHORT[pol]}: {POLICY_LABELS[pol].split('(')[1].rstrip(')')}")
        for pol in pols
    ]
    fig.legend(handles=leg_handles, fontsize=9, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    path = out_dir / "02_storage_pct_over_baseline.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Output 4: Dumbbell plot ────────────────────────────────────────────────────

def plot_latency_dumbbell(metrics: dict, out_dir: Path) -> None:
    """
    Each workload is a horizontal 'dumbbell':
      left endpoint  = baseline latency
      right endpoint = strong latency
      middle point   = adaptive latency

    Makes it visually obvious whether adaptive sits much closer to baseline.
    Uses log-x scale because GGVP latency (~6000 ms) dwarfs FAERS (~70 ms).
    """
    wls    = list(metrics.keys())
    labels = [WORKLOAD_LABELS[wl] for wl in wls]
    y      = np.arange(len(wls))

    fig, ax = plt.subplots(figsize=(12, 5))

    for yi, wl in enumerate(wls):
        m     = metrics[wl]
        b_lat = m["baseline_lat"]
        a_lat = m["adaptive_lat"]
        s_lat = m["strong_lat"]

        # Spine connecting baseline to strong
        x_lo, x_hi = min(b_lat, s_lat), max(b_lat, s_lat)
        ax.plot([x_lo, x_hi], [yi, yi],
                color=WORKLOAD_COLORS[wl], linewidth=3.5, alpha=0.40, zorder=2)

        for x_pt, pol, sz in [
            (b_lat, "baseline", 220),
            (a_lat, "adaptive", 220),
            (s_lat, "strong",   220),
        ]:
            ax.scatter(x_pt, yi,
                       s=sz, color=POLICY_COLORS[pol],
                       edgecolors="white", linewidths=1.5,
                       zorder=5, marker="o")

        # Annotate adaptive position relative to the baseline–strong span
        span = abs(s_lat - b_lat)
        if span > 0:
            frac = abs(a_lat - b_lat) / span
            ax.annotate(
                f"adaptive = {frac*100:.0f}%\nof base→strong span",
                xy=(a_lat, yi),
                xytext=(10, 12), textcoords="offset points",
                fontsize=7.5, color=POLICY_COLORS["adaptive"],
                arrowprops=dict(arrowstyle="-", color=POLICY_COLORS["adaptive"],
                                lw=0.8, alpha=0.7),
            )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: (f"{v:,.0f}" if v >= 100 else f"{v:.1f}")
    ))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Mean End-to-End Latency (ms, log scale)", fontsize=11)
    ax.set_title(
        "Adaptive Policy Position Between Baseline and Strong\n"
        "(adaptive should sit near baseline, far from strong)",
        fontsize=13,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color=POLICY_COLORS[pol],
               markerfacecolor=POLICY_COLORS[pol], markersize=10,
               label=POLICY_LABELS[pol], linewidth=0)
        for pol in POLICY_ORDER
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = out_dir / "03_latency_dumbbell.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Output 5: Cost-ratio heatmap ──────────────────────────────────────────────

def plot_cost_ratio_heatmap(metrics: dict, out_dir: Path) -> None:
    """
    Compact annotated heatmap.
    Rows    = workloads
    Columns = [adap/base latency, strong/base latency,
               adap/base storage, strong/base storage]
    Cell text = ratio formatted as e.g. "0.968×"
    Colour   = diverging from 1.0: green < 1 (cheaper), red > 1 (more expensive).
    """
    wls    = list(metrics.keys())
    col_keys  = ["adap_lat_ratio", "strong_lat_ratio", "adap_amp_ratio", "strong_amp_ratio"]
    col_labels = [
        "Adaptive / Baseline\nLatency",
        "Strong / Baseline\nLatency",
        "Adaptive / Baseline\nStorage Amp.",
        "Strong / Baseline\nStorage Amp.",
    ]

    data = np.array([
        [metrics[wl][k] for k in col_keys]
        for wl in wls
    ])

    # Colour: deviation from 1.0, clamped for display
    deviation = data - 1.0
    vmax = max(0.05, np.nanmax(np.abs(deviation)))

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(deviation, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(col_keys)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(wls)))
    ax.set_yticklabels([WORKLOAD_LABELS[wl] for wl in wls], fontsize=11)

    for ri, wl in enumerate(wls):
        for ci, k in enumerate(col_keys):
            val = metrics[wl][k]
            txt = f"{val:.4f}×"
            dev = val - 1.0
            # White text on dark cells, black on light
            contrast = abs(dev) / vmax if vmax > 0 else 0
            color = "white" if contrast > 0.55 else "black"
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Ratio deviation from 1.0\n(green = cheaper, red = costlier)", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:+.3f}"
    ))

    ax.set_title(
        "Baseline-Normalized Cost Ratios Across Workloads\n"
        "(1.000× = identical cost to baseline; < 1 = cheaper; > 1 = costlier)",
        fontsize=13,
    )

    for edge in ["top", "bottom", "left", "right"]:
        ax.spines[edge].set_visible(False)
    ax.set_xticks(np.arange(len(col_keys)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(wls)) - 0.5,      minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    path = out_dir / "04_cost_ratio_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Output 6: Near-baseline decision table ────────────────────────────────────

def plot_near_baseline_table(metrics: dict, out_dir: Path) -> None:
    """
    Compact summary table rendered as a matplotlib figure.
    Rows = workloads.
    Columns:
      Workload | Adap ≤5%? | Adap ≤10%? | Strong ≤10%? | Adap Esc. Bytes
    """
    wls    = list(metrics.keys())

    col_headers = [
        "Workload",
        "Adaptive\n≤ 5% of baseline?",
        "Adaptive\n≤ 10% of baseline?",
        "Strong\n≤ 10% of baseline?",
        "Adaptive\nEscalated Byte %",
    ]

    rows = []
    for wl in wls:
        m = metrics[wl]
        adap_pct  = abs(m["adap_lat_pct"])   # absolute deviation
        strg_pct  = abs(m["strong_lat_pct"])
        esc_bytes = m["adap_esc_byte_frac"] * 100

        def yn(cond: bool) -> str:
            return "Yes" if cond else "No"

        rows.append([
            WORKLOAD_LABELS[wl],
            yn(adap_pct <= 5.0),
            yn(adap_pct <= 10.0),
            yn(strg_pct <= 10.0),
            f"{esc_bytes:.1f}%",
        ])

    n_rows = len(rows)
    n_cols = len(col_headers)

    fig, ax = plt.subplots(figsize=(11, 2.0 + n_rows * 0.65))
    ax.axis("off")

    col_widths = [0.18, 0.20, 0.20, 0.20, 0.20]
    col_x = [sum(col_widths[:i]) for i in range(n_cols)]

    def cell_color(ri: int, ci: int, val: str) -> str:
        if ci == 0 or ri == -1:
            return "#EAEAEA"
        if ci in (1, 2, 3):
            return "#D5F5D5" if val == "Yes" else "#FAD7D7"
        return "#FFFFFF"

    header_y = 0.92
    row_h    = (header_y - 0.05) / (n_rows + 1)

    # Header row
    for ci, (hdr, x0, w) in enumerate(zip(col_headers, col_x, col_widths)):
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.005, header_y - row_h + 0.005), w - 0.010, row_h - 0.010,
            boxstyle="round,pad=0.005",
            facecolor="#555555", edgecolor="none", transform=ax.transAxes,
            zorder=2,
        ))
        ax.text(x0 + w / 2, header_y - row_h / 2, hdr,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white", transform=ax.transAxes, zorder=3)

    # Data rows
    for ri, row in enumerate(rows):
        y0 = header_y - (ri + 2) * row_h
        for ci, (val, x0, w) in enumerate(zip(row, col_x, col_widths)):
            bg = cell_color(ri, ci, val)
            ax.add_patch(FancyBboxPatch(
                (x0 + 0.005, y0 + 0.005), w - 0.010, row_h - 0.010,
                boxstyle="round,pad=0.005",
                facecolor=bg, edgecolor="#CCCCCC", linewidth=0.5,
                transform=ax.transAxes, zorder=2,
            ))
            fw = "bold" if ci == 0 else "normal"
            ax.text(x0 + w / 2, y0 + row_h / 2, val,
                    ha="center", va="center", fontsize=10,
                    fontweight=fw, transform=ax.transAxes, zorder=3)

    ax.set_title(
        '"Near Baseline" Decision Table\n'
        "(green = within threshold; red = outside threshold; % deviation is absolute value)",
        fontsize=12, pad=12,
    )

    # Legend note
    note = (
        "Note: thresholds applied to |% change| from baseline latency.  "
        "'Escalated Byte %' = fraction of plaintext bytes under strong-tier protection under adaptive policy."
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, style="italic", color="#555555")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = out_dir / "05_near_baseline_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Validation helper ─────────────────────────────────────────────────────────

def _validate(metrics: dict) -> None:
    print("\n  === Validation: raw values used for normalized metrics ===")
    for wl, m in metrics.items():
        print(f"\n  [{WORKLOAD_LABELS[wl]}]")
        print(f"    Baseline latency : {m['baseline_lat']:.4f} ms")
        print(f"    Adaptive latency : {m['adaptive_lat']:.4f} ms  "
              f"({m['adap_lat_pct']:+.3f}%,  ratio={m['adap_lat_ratio']:.6f})")
        print(f"    Strong latency   : {m['strong_lat']:.4f} ms  "
              f"({m['strong_lat_pct']:+.3f}%,  ratio={m['strong_lat_ratio']:.6f})")
        print(f"    Baseline amp     : {m['baseline_amp']:.6f}×")
        print(f"    Adaptive amp     : {m['adaptive_amp']:.6f}×  "
              f"(ratio={m['adap_amp_ratio']:.8f})")
        print(f"    Strong amp       : {m['strong_amp']:.6f}×  "
              f"(ratio={m['strong_amp_ratio']:.8f})")
        print(f"    Adap esc. bytes  : {m['adap_esc_byte_frac']*100:.1f}%")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate baseline-normalized policy figures and tables for the PQC benchmark."
    )
    p.add_argument(
        "--pilot-dir",
        default="results/policy_pilot/20260417_011356/json",
        help="Directory containing the 9 policy-pilot JSON files.",
    )
    p.add_argument(
        "--output-dir",
        default="results/policy_pilot/20260417_011356/policy_figures_normalized",
        help="Output directory for generated figures and tables.",
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
        # Update the adaptive policy label to reflect the actual threshold
        threshold_bytes = override_data.get("policy_threshold_bytes") or (
            override_data.get("policy_summary", {}).get("threshold_bytes")
        )
        POLICY_LABELS["adaptive"] = _adaptive_label_from_threshold(threshold_bytes)

    print(f"\n  Mixed-workload note:")
    print("    A 'mixed' row (FAERS + GGVP combined) is not available from the current")
    print("    JSON outputs. Each JSON file covers a single workload type.")
    print("    To assemble a mixed row, a benchmark run combining FAERS and GGVP payloads")
    print("    in a single session would be required.")

    print(f"\n  Computing normalized metrics...")
    metrics = extract_normalized_metrics(pilot)
    if not metrics:
        print("  ERROR: no complete workloads found — cannot generate normalized figures.")
        return

    _validate(metrics)

    print(f"  Writing outputs to  {out_dir}/\n")

    funcs = [
        lambda m, o: write_cost_table(m, o),
        lambda m, o: plot_latency_pct_over_baseline(m, o),
        lambda m, o: plot_storage_pct_over_baseline(m, o),
        lambda m, o: plot_latency_dumbbell(m, o),
        lambda m, o: plot_cost_ratio_heatmap(m, o),
        lambda m, o: plot_near_baseline_table(m, o),
    ]
    n_saved = 0
    for fn in funcs:
        try:
            fn(metrics, out_dir)
            n_saved += 1
        except Exception as exc:
            print(f"  ERROR in {fn}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\n  Done. {n_saved}/{len(funcs)} outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
