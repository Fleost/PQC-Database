#!/usr/bin/env python3
"""Latency comparison: faers_adaptive_50kb vs faers_full_matched_baseline."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict

ADAPTIVE_JSON = Path("results/faers_adaptive_50kb/20260418_153046/json/faers_adaptive_threshold_50kb.json")
BASELINE_JSON = Path("results/faers_full_matched_baseline/20260418_183640/json/faers_uniform_baseline.json")
OUT_DIR       = Path("results/faers_full_visuals")
OUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "baseline": "#4878CF",
    "adaptive": "#6ACC65",
}
LABELS = {
    "baseline": "Baseline (ML-KEM-768 + ML-DSA-65, uniform)",
    "adaptive": "Adaptive (≤50KB → 768+65, >50KB → 1024+87)",
}
THRESHOLD = 51200
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

BINS = [
    (0,        2_048,        "< 2 KB"),
    (2_048,    5_120,        "2–5 KB"),
    (5_120,    10_240,       "5–10 KB"),
    (10_240,   25_600,       "10–25 KB"),
    (25_600,   51_200,       "25–50 KB"),
    (51_200,   102_400,      "50–100 KB"),
    (102_400,  512_000,      "100–500 KB"),
    (512_000,  float("inf"), "> 500 KB"),
]
BIN_LABELS = [b[2] for b in BINS]
THRESHOLD_BIN_IDX = 5  # first bin that is fully above 50KB

def bin_label(size_bytes):
    for lo, hi, label in BINS:
        if lo <= size_bytes < hi:
            return label
    return "> 500 KB"

def load(path):
    with open(path) as f:
        return json.load(f)["results"]

print("Loading …")
baseline = load(BASELINE_JSON)
adaptive = load(ADAPTIVE_JSON)

OPS = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt", "end_to_end"]

def aggregate(results):
    buckets = defaultdict(lambda: defaultdict(list))
    for r in results:
        bl = bin_label(r["size_bytes"])
        for op in OPS:
            if op in r:
                buckets[bl][op].append(r[op]["mean_ms"])
    return buckets

agg_b = aggregate(baseline)
agg_a = aggregate(adaptive)

def med(lst):  return float(np.median(lst))   if lst else 0.0
def p95(lst):  return float(np.percentile(lst, 95)) if lst else 0.0
def p5(lst):   return float(np.percentile(lst, 5))  if lst else 0.0

# ── Figure 1: End-to-end latency by size bin ─────────────────────────────────
def fig_e2e_bar():
    vb = [med(agg_b[bl]["end_to_end"]) for bl in BIN_LABELS]
    va = [med(agg_a[bl]["end_to_end"]) for bl in BIN_LABELS]

    x = np.arange(len(BIN_LABELS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, vb, w, color=COLORS["baseline"], label=LABELS["baseline"], alpha=0.85)
    ax.bar(x + w/2, va, w, color=COLORS["adaptive"], label=LABELS["adaptive"], alpha=0.85)

    # shade escalated region
    ax.axvspan(THRESHOLD_BIN_IDX - 0.5, len(BIN_LABELS) - 0.5,
               alpha=0.07, color="orange", label="Escalated to strong (>50KB)")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("End-to-End Latency (ms, median)")
    ax.set_title("End-to-End Latency by Payload Size — Adaptive vs Baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    fig.tight_layout()
    out = OUT_DIR / "13_adaptive_vs_baseline_e2e.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

# ── Figure 2: Per-operation latency across all bins (line) ───────────────────
def fig_per_op_lines():
    ops_plot = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt"]
    op_labels = ["Encrypt", "Sign", "DB Insert", "DB Fetch", "Verify", "Decrypt"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.flatten()
    x = np.arange(len(BIN_LABELS))

    for ax, op, label in zip(axes, ops_plot, op_labels):
        vb = [med(agg_b[bl][op]) for bl in BIN_LABELS]
        va = [med(agg_a[bl][op]) for bl in BIN_LABELS]
        eb = [med(agg_b[bl][op]) - p5(agg_b[bl][op]) for bl in BIN_LABELS]
        ea = [med(agg_a[bl][op]) - p5(agg_a[bl][op]) for bl in BIN_LABELS]
        tb = [p95(agg_b[bl][op]) - med(agg_b[bl][op]) for bl in BIN_LABELS]
        ta = [p95(agg_a[bl][op]) - med(agg_a[bl][op]) for bl in BIN_LABELS]

        ax.axvspan(THRESHOLD_BIN_IDX - 0.5, len(BIN_LABELS) - 0.5,
                   alpha=0.07, color="orange")
        ax.errorbar(x, vb, yerr=[eb, tb], fmt="o-", color=COLORS["baseline"],
                    label=LABELS["baseline"], capsize=3, lw=1.5, ms=5)
        ax.errorbar(x, va, yerr=[ea, ta], fmt="s-", color=COLORS["adaptive"],
                    label=LABELS["adaptive"], capsize=3, lw=1.5, ms=5)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("ms (median ± p5/p95)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Per-Operation Latency — Adaptive vs Baseline (orange = escalated zone)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "14_adaptive_vs_baseline_per_op.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

# ── Figure 3: CDF of end-to-end latency split by escalated / not ─────────────
def fig_cdf():
    bl_all    = [r["end_to_end"]["mean_ms"] for r in baseline]
    ad_all    = [r["end_to_end"]["mean_ms"] for r in adaptive]
    ad_esc    = [r["end_to_end"]["mean_ms"] for r in adaptive if r["escalated"]]
    ad_noesc  = [r["end_to_end"]["mean_ms"] for r in adaptive if not r["escalated"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    for data, label, color, ls in [
        (bl_all,   LABELS["baseline"],                    COLORS["baseline"], "-"),
        (ad_all,   LABELS["adaptive"] + " (all)",         COLORS["adaptive"], "-"),
        (ad_noesc, "Adaptive — below threshold (86%)",    "#2ca02c",          "--"),
        (ad_esc,   "Adaptive — escalated >50KB (14%)",    "#d62728",          ":"),
    ]:
        s = np.sort(data)
        cdf = np.arange(1, len(s)+1) / len(s)
        ax.plot(s, cdf, color=color, ls=ls, lw=1.8, label=f"{label} (n={len(data):,})")

    ax.set_xlabel("End-to-End Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("End-to-End Latency CDF — Adaptive vs Baseline")
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "15_adaptive_vs_baseline_cdf.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

# ── Figure 4: Latency delta (adaptive − baseline) per size bin ───────────────
def fig_delta_bar():
    ops_plot  = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt", "end_to_end"]
    op_labels = ["Encrypt", "Sign", "DB Insert", "DB Fetch", "Verify", "Decrypt", "E2E"]

    # Build paired delta per record
    ad_idx = {r["payload_name"]: r for r in adaptive}
    bl_idx = {r["payload_name"]: r for r in baseline}

    bin_deltas = defaultdict(lambda: defaultdict(list))
    for name in ad_idx:
        a = ad_idx[name]; b = bl_idx[name]
        bl = bin_label(a["size_bytes"])
        for op in ops_plot:
            if op in a and op in b:
                bin_deltas[bl][op].append(a[op]["mean_ms"] - b[op]["mean_ms"])

    x = np.arange(len(BIN_LABELS))
    n_ops = len(ops_plot)
    cmap = plt.get_cmap("tab10")
    op_colors = [cmap(i) for i in range(n_ops)]

    fig, ax = plt.subplots(figsize=(13, 5))
    w = 0.10
    for i, (op, label, color) in enumerate(zip(ops_plot, op_labels, op_colors)):
        deltas = [med(bin_deltas[bl][op]) for bl in BIN_LABELS]
        ax.bar(x + (i - n_ops/2 + 0.5)*w, deltas, w, label=label, color=color, alpha=0.85)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvspan(THRESHOLD_BIN_IDX - 0.5, len(BIN_LABELS) - 0.5,
               alpha=0.07, color="orange", label="Escalated zone (>50KB)")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Δ Latency (ms, median)  [adaptive − baseline]")
    ax.set_title("Latency Delta: Adaptive − Baseline by Payload Size and Operation")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "16_adaptive_vs_baseline_delta.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

print("Generating charts …")
fig_e2e_bar()
fig_per_op_lines()
fig_cdf()
fig_delta_bar()
print("Done.")
