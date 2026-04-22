#!/usr/bin/env python3
"""
DSA variant latency comparison at discrete representative payload sizes.
Shows all 4 variants (Unsigned, ML-DSA-44, ML-DSA-65, ML-DSA-87) as grouped bars.
"""

from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RESULTS_DIR = Path("results")

DSA_FILES = {
    "Unsigned":  RESULTS_DIR / "bench_unsigned_kem768_faers.json",
    "ML-DSA-44": RESULTS_DIR / "bench_mldsa44_kem768_faers.json",
    "ML-DSA-65": RESULTS_DIR / "bench_mldsa65_kem768_faers.json",
    "ML-DSA-87": RESULTS_DIR / "bench_mldsa87_kem768_faers.json",
}

COLORS = {
    "Unsigned":  "#2E5EAA",
    "ML-DSA-44": "#3C7A3B",
    "ML-DSA-65": "#C96B28",
    "ML-DSA-87": "#9C2C2C",
}

# Representative target sizes and their display labels
TARGETS = [
    (2_000,     "2 KB"),
    (5_000,     "5 KB"),
    (10_000,    "10 KB"),
    (25_000,    "25 KB"),
    (50_000,    "50 KB"),
    (100_000,   "100 KB"),
    (500_000,   "500 KB"),
    (1_000_000, "1 MB"),
    (5_000_000, "5 MB"),
]
# Window around each target (±20%) to collect records for aggregation
WINDOW = 0.20

def load(path: Path) -> dict[int, list[float]]:
    """Return {size_bytes: [end_to_end p50_ms, ...]} for all records."""
    with path.open() as f:
        results = json.load(f)["results"]
    by_size: dict[int, list[float]] = {}
    for r in results:
        by_size.setdefault(r["size_bytes"], []).append(r["end_to_end"]["p50_ms"])
    return by_size

def get_latency(by_size: dict, target: int) -> tuple[float, float, int]:
    """Return (median, p95, n) of records within ±WINDOW of target."""
    lo, hi = target * (1 - WINDOW), target * (1 + WINDOW)
    vals = [v for sz, vs in by_size.items() if lo <= sz <= hi for v in vs]
    if not vals:
        return (0.0, 0.0, 0)
    return float(np.median(vals)), float(np.percentile(vals, 95)), len(vals)

print("Loading data …")
data = {label: load(path) for label, path in DSA_FILES.items()}

labels = list(DSA_FILES.keys())
size_labels = [t[1] for t in TARGETS]
n_sizes = len(TARGETS)
n_labels = len(labels)

# Build arrays: medians[label][size_idx], p95s[label][size_idx]
medians = {lbl: [] for lbl in labels}
p95s    = {lbl: [] for lbl in labels}
counts  = {lbl: [] for lbl in labels}

for target, _ in TARGETS:
    for lbl in labels:
        med, p95, n = get_latency(data[lbl], target)
        medians[lbl].append(med)
        p95s[lbl].append(p95)
        counts[lbl].append(n)

# ── Figure: small-multiples grid, one panel per payload size ─────────────────
NCOLS = 3
NROWS = (n_sizes + NCOLS - 1) // NCOLS

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, 4 * NROWS))
axes = axes.flatten()

x = np.arange(n_labels)
w = 0.65

def fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.2f} s"
    if v >= 10:
        return f"{v:.1f} ms"
    return f"{v:.2f} ms"

for j, (target, size_lbl) in enumerate(TARGETS):
    ax = axes[j]
    vals   = [medians[lbl][j] for lbl in labels]
    uppers = [p95s[lbl][j] - medians[lbl][j] for lbl in labels]
    colors = [COLORS[lbl] for lbl in labels]

    bars = ax.bar(x, vals, w, color=colors, alpha=0.88, zorder=3)
    ax.errorbar(x, vals, yerr=[np.zeros(n_labels), uppers],
                fmt="none", color="black", capsize=4, lw=1.2, zorder=4)

    # Value labels on top of each bar
    y_top = max(v + u for v, u in zip(vals, uppers)) if vals else 1
    for xi, v, u in zip(x, vals, uppers):
        ax.text(xi, v + u + y_top * 0.03, fmt_ms(v),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_title(size_lbl, fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (p50)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_ms(v)))
    ax.set_ylim(0, y_top * 1.22)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    n = counts["ML-DSA-65"][j]
    ax.text(0.98, 0.97, f"n ≈ {n}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="gray")

# Hide unused panels
for j in range(n_sizes, len(axes)):
    axes[j].set_visible(False)

# Shared legend at top
handles = [plt.Rectangle((0,0),1,1, color=COLORS[lbl], alpha=0.88) for lbl in labels]
fig.legend(handles, labels, loc="upper center", ncol=n_labels,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.01))

fig.suptitle("FAERS End-to-End Latency — DSA Variant Comparison\n(p50 bar, p95 error cap)",
             fontsize=13, y=1.04)
fig.tight_layout()
out = RESULTS_DIR / "20_faers_dsa_latency_comparison.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
