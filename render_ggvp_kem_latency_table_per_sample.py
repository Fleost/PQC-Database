#!/usr/bin/env python3
"""
Render a table comparing per-sample % latency increase from ML-KEM-768 to
ML-KEM-1024 (both with ML-DSA-65) on the GGVP dataset.

For each payload the % increase is computed individually, then those
per-sample percentages are averaged — avoiding the bias that arises when
averaging raw latencies across payloads of very different sizes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

FILES = {
    "ML-KEM-768":  RESULTS_DIR / "KEM768_DSA65_GGVP.json",
    "ML-KEM-1024": RESULTS_DIR / "KEM1024_DSA65_GGVP.json",
}

OPERATIONS = [
    ("encrypt",    "Encrypt"),
    ("sign",       "Sign"),
    ("db_insert",  "DB Insert"),
    ("db_fetch",   "DB Fetch"),
    ("verify",     "Verify"),
    ("decrypt",    "Decrypt"),
    ("end_to_end", "End-to-End"),
]


def load_by_size(path: Path) -> dict[int, dict[str, float]]:
    """Return {size_bytes: {op_key: mean_ms}} for every result entry."""
    with path.open() as f:
        results = json.load(f)["results"]
    out: dict[int, dict[str, float]] = {}
    for r in results:
        out[r["size_bytes"]] = {op: r[op]["mean_ms"] for op, _ in OPERATIONS}
    return out


def main() -> None:
    data = {label: load_by_size(path) for label, path in FILES.items()}

    kem768  = data["ML-KEM-768"]
    kem1024 = data["ML-KEM-1024"]

    # Only compare payloads present in both runs
    common_sizes = sorted(set(kem768) & set(kem1024))
    n = len(common_sizes)

    # Per-operation: collect per-sample % increases, then average
    col_headers = [
        "Operation",
        "ML-KEM-768 avg (ms)",
        "ML-KEM-1024 avg (ms)",
        "Avg % increase per sample",
    ]
    rows = []
    for op_key, op_label in OPERATIONS:
        pct_per_sample = []
        v768_vals  = []
        v1024_vals = []
        for sz in common_sizes:
            v768  = kem768[sz][op_key]
            v1024 = kem1024[sz][op_key]
            v768_vals.append(v768)
            v1024_vals.append(v1024)
            pct_per_sample.append((v1024 - v768) / v768 * 100)

        avg_pct  = float(np.mean(pct_per_sample))
        avg_768  = float(np.mean(v768_vals))
        avg_1024 = float(np.mean(v1024_vals))
        sign = "+" if avg_pct >= 0 else ""
        rows.append([
            op_label,
            f"{avg_768:,.2f}",
            f"{avg_1024:,.2f}",
            f"{sign}{avg_pct:.1f}%",
        ])

    # ------------------------------------------------------------------ render
    n_rows = len(rows)
    n_cols = len(col_headers)

    fig_h = 0.5 + n_rows * 0.42 + 0.9
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.55)

    # Style header row
    header_color = "#2E3A4E"
    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading + colour-code the % column
    for row_idx in range(1, n_rows + 1):
        bg = "#F5F5F5" if row_idx % 2 == 0 else "white"
        for col in range(n_cols):
            tbl[row_idx, col].set_facecolor(bg)
            tbl[row_idx, col].set_edgecolor("#CCCCCC")

        pct_cell = tbl[row_idx, 3]
        pct_val  = float(pct_cell.get_text().get_text().replace("%", "").replace("+", ""))
        if pct_val > 5:
            pct_cell.set_facecolor("#FDDEDE")
        elif pct_val < -5:
            pct_cell.set_facecolor("#DEF0DD")
        else:
            pct_cell.set_facecolor("#FFF8E1")

    # Left-align the Operation column
    for row_idx in range(n_rows + 1):
        tbl[row_idx, 0].set_text_props(ha="left")

    ax.set_title(
        "GGVP Average Latency: ML-KEM-768 vs ML-KEM-1024 (both with ML-DSA-65)\n"
        f"% increase computed per sample then averaged across {n} GGVP payloads",
        fontsize=11,
        pad=12,
        loc="left",
    )

    fig.tight_layout()
    output = RESULTS_DIR / "26_ggvp_kem_latency_table_per_sample.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output}")

    # Also print to stdout
    col_w = [max(len(col_headers[c]), *(len(r[c]) for r in rows)) for c in range(n_cols)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(col_w[i]) for i, c in enumerate(cells)) + " |"

    print()
    print(sep)
    print(fmt_row(col_headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)


if __name__ == "__main__":
    main()
