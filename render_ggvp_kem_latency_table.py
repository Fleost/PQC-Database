#!/usr/bin/env python3
"""
Render a table comparing average per-operation latency between
ML-KEM-768 + ML-DSA-65 and ML-KEM-1024 + ML-DSA-65 on the GGVP dataset,
and the % increase from KEM-768 to KEM-1024.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

FILES = {
    "ML-KEM-768": RESULTS_DIR / "KEM768_DSA65_GGVP.json",
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


def load_avg_latencies(path: Path) -> dict[str, float]:
    """Return {op_key: mean of mean_ms across all payloads}."""
    with path.open() as f:
        results = json.load(f)["results"]
    avgs: dict[str, list[float]] = {op: [] for op, _ in OPERATIONS}
    for r in results:
        for op, _ in OPERATIONS:
            avgs[op].append(r[op]["mean_ms"])
    return {op: float(np.mean(vals)) for op, vals in avgs.items()}


def main() -> None:
    avgs = {label: load_avg_latencies(path) for label, path in FILES.items()}

    kem768  = avgs["ML-KEM-768"]
    kem1024 = avgs["ML-KEM-1024"]

    # Build table rows
    col_headers = ["Operation", "ML-KEM-768 (ms)", "ML-KEM-1024 (ms)", "% Increase"]
    rows = []
    for op_key, op_label in OPERATIONS:
        v768  = kem768[op_key]
        v1024 = kem1024[op_key]
        pct   = (v1024 - v768) / v768 * 100
        sign  = "+" if pct >= 0 else ""
        rows.append([
            op_label,
            f"{v768:,.2f}",
            f"{v1024:,.2f}",
            f"{sign}{pct:.1f}%",
        ])

    # ------------------------------------------------------------------ render
    n_rows = len(rows)
    n_cols = len(col_headers)

    fig_h = 0.5 + n_rows * 0.42 + 0.6   # dynamic height
    fig, ax = plt.subplots(figsize=(9, fig_h))
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
    for row_idx, (op_key, _) in enumerate(OPERATIONS, start=1):
        bg = "#F5F5F5" if row_idx % 2 == 0 else "white"
        for col in range(n_cols):
            tbl[row_idx, col].set_facecolor(bg)
            tbl[row_idx, col].set_edgecolor("#CCCCCC")

        # Colour-code the % increase cell
        pct_cell = tbl[row_idx, 3]
        pct_text = pct_cell.get_text().get_text()
        pct_val = float(pct_text.replace("%", "").replace("+", ""))
        if pct_val > 5:
            pct_cell.set_facecolor("#FDDEDE")   # red-tint  → slower
        elif pct_val < -5:
            pct_cell.set_facecolor("#DEF0DD")   # green-tint → faster
        else:
            pct_cell.set_facecolor("#FFF8E1")   # yellow-tint → negligible

    # Left-align the Operation column
    for row_idx in range(n_rows + 1):
        tbl[row_idx, 0].set_text_props(ha="left")

    ax.set_title(
        "GGVP Average Latency: ML-KEM-768 vs ML-KEM-1024 (both with ML-DSA-65)\n"
        "Averages computed across all 23 GGVP payloads (mean of per-payload mean_ms)",
        fontsize=11,
        pad=12,
        loc="left",
    )

    fig.tight_layout()
    output = RESULTS_DIR / "25_ggvp_kem_latency_table.png"
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
