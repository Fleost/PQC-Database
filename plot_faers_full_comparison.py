#!/usr/bin/env python3
"""Compare faers_full_matched_baseline vs faers_full_matched_strong across storage and latency."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict

BASELINE_JSON = Path("results/faers_full_matched_baseline/20260418_183640/json/faers_uniform_baseline.json")
STRONG_JSON   = Path("results/faers_full_matched_strong/20260419_161343/json/faers_uniform_strong.json")
OUT_DIR       = Path("results/faers_full_visuals")
OUT_DIR.mkdir(exist_ok=True)

# ── Colour / style ──────────────────────────────────────────────────────────
COLORS = {
    "baseline": "#4878CF",   # blue  (ML-KEM-768 + ML-DSA-65)
    "strong":   "#D65F5F",   # red   (ML-KEM-1024 + ML-DSA-87)
}
LABELS = {
    "baseline": "Baseline (ML-KEM-768 + ML-DSA-65)",
    "strong":   "Strong (ML-KEM-1024 + ML-DSA-87)",
}
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── Size bins ────────────────────────────────────────────────────────────────
BINS = [
    (0,       2_048,    "< 2 KB"),
    (2_048,   5_120,    "2–5 KB"),
    (5_120,   10_240,   "5–10 KB"),
    (10_240,  25_600,   "10–25 KB"),
    (25_600,  51_200,   "25–50 KB"),
    (51_200,  102_400,  "50–100 KB"),
    (102_400, 512_000,  "100–500 KB"),
    (512_000, float("inf"), "> 500 KB"),
]

def bin_label(size_bytes):
    for lo, hi, label in BINS:
        if lo <= size_bytes < hi:
            return label
    return "> 500 KB"

BIN_LABELS = [b[2] for b in BINS]

# ── Load data ────────────────────────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)["results"]

print("Loading data …")
baseline_results = load(BASELINE_JSON)
strong_results   = load(STRONG_JSON)

# ── Aggregate helpers ────────────────────────────────────────────────────────
LATENCY_OPS = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt", "end_to_end"]

def aggregate(results):
    """Return {bin_label: {op: [mean_ms, ...], 'envelope': {...}}}"""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in results:
        bl = bin_label(r["size_bytes"])
        for op in LATENCY_OPS:
            if op in r:
                buckets[bl][op].append(r[op]["mean_ms"])
        env = r.get("envelope_bytes", {})
        for field in ["overhead_bytes", "total_stored_bytes", "amplification",
                      "pq_ct_bytes", "signature_bytes", "plaintext_bytes"]:
            if field in env:
                buckets[bl][field].append(env[field])
    return buckets

print("Aggregating …")
agg_b = aggregate(baseline_results)
agg_s = aggregate(strong_results)

def median(lst): return float(np.median(lst)) if lst else 0.0
def p95(lst):    return float(np.percentile(lst, 95)) if lst else 0.0

# ── Helper: grouped bar chart ────────────────────────────────────────────────
def grouped_bar(ax, values_b, values_s, bin_labels, ylabel, title, log=False):
    x = np.arange(len(bin_labels))
    w = 0.38
    bars_b = ax.bar(x - w/2, values_b, w, color=COLORS["baseline"], label=LABELS["baseline"], alpha=0.85)
    bars_s = ax.bar(x + w/2, values_s, w, color=COLORS["strong"],   label=LABELS["strong"],   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    if log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", alpha=0.3)

# ── Figure 1: End-to-end latency (median) ───────────────────────────────────
def e2e_latency_chart():
    vb = [median(agg_b[bl]["end_to_end"]) for bl in BIN_LABELS]
    vs = [median(agg_s[bl]["end_to_end"]) for bl in BIN_LABELS]
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(ax, vb, vs, BIN_LABELS, "Latency (ms, median)", "End-to-End Latency by Payload Size")
    fig.tight_layout()
    path = OUT_DIR / "01_e2e_latency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 2: Per-operation latency breakdown (small/medium/large panels) ───
def per_op_latency_chart():
    # Choose three representative bins
    rep_bins = ["< 2 KB", "10–25 KB", "100–500 KB"]
    ops_clean = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt"]
    op_labels = ["Encrypt", "Sign", "DB Insert", "DB Fetch", "Verify", "Decrypt"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for ax, bl in zip(axes, rep_bins):
        vb = [median(agg_b[bl][op]) for op in ops_clean]
        vs = [median(agg_s[bl][op]) for op in ops_clean]
        x = np.arange(len(ops_clean))
        w = 0.38
        ax.bar(x - w/2, vb, w, color=COLORS["baseline"], alpha=0.85, label=LABELS["baseline"])
        ax.bar(x + w/2, vs, w, color=COLORS["strong"],   alpha=0.85, label=LABELS["strong"])
        ax.set_xticks(x)
        ax.set_xticklabels(op_labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"Payload bin: {bl}")
        ax.set_ylabel("Latency (ms, median)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Per-Operation Latency — Representative Payload Sizes", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "02_per_op_latency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 3: All-ops latency heatmap (median ms) ───────────────────────────
def latency_heatmap():
    ops_clean = ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt", "end_to_end"]
    op_labels = ["Encrypt", "Sign", "DB Insert", "DB Fetch", "Verify", "Decrypt", "End-to-End"]

    def build_matrix(agg):
        return np.array([[median(agg[bl][op]) for bl in BIN_LABELS] for op in ops_clean])

    mat_b = build_matrix(agg_b)
    mat_s = build_matrix(agg_s)
    diff  = mat_s - mat_b   # strong minus baseline

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title, cmap in [
        (axes[0], mat_b, "Baseline (ms)", "Blues"),
        (axes[1], mat_s, "Strong (ms)",   "Reds"),
        (axes[2], diff,  "Δ Strong − Baseline (ms)", "RdBu_r"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(op_labels)))
        ax.set_yticklabels(op_labels, fontsize=9)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(op_labels)):
            for j in range(len(BIN_LABELS)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat[i,j]) > mat.max()*0.6 else "black")
    fig.suptitle("Latency Heatmap (median ms per operation × payload size)", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "03_latency_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 4: Storage overhead (bytes) ──────────────────────────────────────
def storage_overhead_chart():
    vb = [median(agg_b[bl]["overhead_bytes"]) for bl in BIN_LABELS]
    vs = [median(agg_s[bl]["overhead_bytes"]) for bl in BIN_LABELS]
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(ax, vb, vs, BIN_LABELS, "Overhead (bytes, median)", "Cryptographic Overhead by Payload Size")
    fig.tight_layout()
    path = OUT_DIR / "04_storage_overhead.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 5: Total stored bytes ─────────────────────────────────────────────
def total_stored_chart():
    vb = [median(agg_b[bl]["total_stored_bytes"]) for bl in BIN_LABELS]
    vs = [median(agg_s[bl]["total_stored_bytes"]) for bl in BIN_LABELS]
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(ax, vb, vs, BIN_LABELS, "Total stored (bytes, median)", "Total Stored Bytes by Payload Size")
    fig.tight_layout()
    path = OUT_DIR / "05_total_stored_bytes.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 6: Storage amplification ratio ───────────────────────────────────
def amplification_chart():
    vb = [median(agg_b[bl]["amplification"]) for bl in BIN_LABELS]
    vs = [median(agg_s[bl]["amplification"]) for bl in BIN_LABELS]
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(ax, vb, vs, BIN_LABELS, "Amplification ratio (median)", "Storage Amplification by Payload Size")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}×"))
    fig.tight_layout()
    path = OUT_DIR / "06_amplification.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 7: Overhead component breakdown — constant bytes ─────────────────
def overhead_components_chart():
    """Show pq_ct_bytes and signature_bytes side by side for each scheme (constant per scheme)."""
    # These are constant across payloads within a scheme, so just take median of first bin with data
    def get_const(agg, field):
        for bl in BIN_LABELS:
            vals = agg[bl][field]
            if vals:
                return median(vals)
        return 0.0

    components = ["pq_ct_bytes", "signature_bytes"]
    comp_labels = ["KEM Ciphertext", "Signature"]
    vb = [get_const(agg_b, c) for c in components]
    vs = [get_const(agg_s, c) for c in components]

    x = np.arange(len(comp_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w/2, vb, w, color=COLORS["baseline"], label=LABELS["baseline"], alpha=0.85)
    ax.bar(x + w/2, vs, w, color=COLORS["strong"],   label=LABELS["strong"],   alpha=0.85)
    for xi, v in zip(x - w/2, vb):
        ax.text(xi, v + 20, f"{int(v):,}", ha="center", va="bottom", fontsize=10)
    for xi, v in zip(x + w/2, vs):
        ax.text(xi, v + 20, f"{int(v):,}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=11)
    ax.set_ylabel("Bytes")
    ax.set_title("Fixed Overhead Components per Scheme")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    path = OUT_DIR / "07_overhead_components.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 8: Latency CDF for end_to_end ────────────────────────────────────
def e2e_cdf_chart():
    all_b = [r["end_to_end"]["mean_ms"] for r in baseline_results]
    all_s = [r["end_to_end"]["mean_ms"] for r in strong_results]
    fig, ax = plt.subplots(figsize=(9, 5))
    for data, key in [(all_b, "baseline"), (all_s, "strong")]:
        sorted_d = np.sort(data)
        cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
        ax.plot(sorted_d, cdf, color=COLORS[key], label=LABELS[key], lw=1.5)
    ax.set_xlabel("End-to-End Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("End-to-End Latency CDF (all payloads)")
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "08_e2e_latency_cdf.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 9: Overhead vs payload size scatter ───────────────────────────────
def overhead_scatter():
    fig, ax = plt.subplots(figsize=(10, 5))
    for results, key in [(baseline_results, "baseline"), (strong_results, "strong")]:
        xs = [r["size_bytes"] for r in results]
        ys = [r["envelope_bytes"]["overhead_bytes"] for r in results if "envelope_bytes" in r]
        xs2 = [r["size_bytes"] for r in results if "envelope_bytes" in r]
        ax.scatter(xs2, ys, s=1, alpha=0.15, color=COLORS[key], label=LABELS[key])
    ax.set_xscale("log")
    ax.set_xlabel("Payload size (bytes, log scale)")
    ax.set_ylabel("Overhead bytes")
    ax.set_title("Cryptographic Overhead vs Payload Size")
    ax.legend(fontsize=9, markerscale=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "09_overhead_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 10: Amplification vs payload size scatter ────────────────────────
def amplification_scatter():
    fig, ax = plt.subplots(figsize=(10, 5))
    for results, key in [(baseline_results, "baseline"), (strong_results, "strong")]:
        xs = [r["size_bytes"] for r in results if "envelope_bytes" in r]
        ys = [r["envelope_bytes"]["amplification"] for r in results if "envelope_bytes" in r]
        ax.scatter(xs, ys, s=1, alpha=0.15, color=COLORS[key], label=LABELS[key])
    ax.set_xscale("log")
    ax.set_xlabel("Payload size (bytes, log scale)")
    ax.set_ylabel("Amplification ratio (total / plaintext)")
    ax.set_title("Storage Amplification vs Payload Size")
    ax.legend(fontsize=9, markerscale=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "10_amplification_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 11: Sign + Verify latency comparison ──────────────────────────────
def sig_latency_chart():
    ops = ["sign", "verify"]
    op_labels = ["Sign", "Verify"]
    vb = [median(agg_b["< 2 KB"][op]) for op in ops]
    vs = [median(agg_s["< 2 KB"][op]) for op in ops]

    # Also show across bins in a line plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, op, label in zip(axes, ops, op_labels):
        vb_line = [median(agg_b[bl][op]) for bl in BIN_LABELS]
        vs_line = [median(agg_s[bl][op]) for bl in BIN_LABELS]
        x = np.arange(len(BIN_LABELS))
        ax.plot(x, vb_line, "o-", color=COLORS["baseline"], label=LABELS["baseline"])
        ax.plot(x, vs_line, "s-", color=COLORS["strong"],   label=LABELS["strong"])
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Latency (ms, median)")
        ax.set_title(f"{label} Latency by Payload Size")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Signing & Verification Latency Comparison", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "11_sign_verify_latency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 12: Encrypt + Decrypt latency ────────────────────────────────────
def enc_dec_latency_chart():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, op, label in zip(axes, ["encrypt", "decrypt"], ["Encrypt", "Decrypt"]):
        vb_line = [median(agg_b[bl][op]) for bl in BIN_LABELS]
        vs_line = [median(agg_s[bl][op]) for bl in BIN_LABELS]
        x = np.arange(len(BIN_LABELS))
        ax.plot(x, vb_line, "o-", color=COLORS["baseline"], label=LABELS["baseline"])
        ax.plot(x, vs_line, "s-", color=COLORS["strong"],   label=LABELS["strong"])
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Latency (ms, median)")
        ax.set_title(f"{label} Latency by Payload Size")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Encrypt & Decrypt Latency Comparison", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "12_enc_dec_latency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ── Run all ──────────────────────────────────────────────────────────────────
print("Generating charts …")
e2e_latency_chart()
per_op_latency_chart()
latency_heatmap()
storage_overhead_chart()
total_stored_chart()
amplification_chart()
overhead_components_chart()
e2e_cdf_chart()
overhead_scatter()
amplification_scatter()
sig_latency_chart()
enc_dec_latency_chart()
print("Done. All charts saved to", OUT_DIR)
