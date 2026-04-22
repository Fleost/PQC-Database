#!/usr/bin/env python3
"""
Render a plain-text table of measured storage footprint totals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def summarize_totals(data: dict) -> dict[str, int]:
    """Summarize plaintext, classical, and hybrid storage totals from a benchmark file.

    Hybrid-only files (no classical rows) will return 0 for plaintext and classical.
    """
    seen: set[tuple[str, str]] = set()
    plaintext = 0
    classical = 0
    hybrid = 0

    for row in data["results"]:
        key = (row["scheme"], row["payload_name"])
        if key in seen:
            continue
        seen.add(key)

        size_bytes = row["size_bytes"]
        total_stored = row["envelope_bytes"]["total_stored_bytes"]
        if row["scheme"] == "classical":
            plaintext += size_bytes
            classical += total_stored
        elif row["scheme"] == "hybrid":
            hybrid += total_stored

    return {"plaintext": plaintext, "classical": classical, "hybrid": hybrid}


def merge_hybrid(base: dict[str, int], hybrid_only: dict[str, int]) -> dict[str, int]:
    """Return a totals dict using plaintext/classical from *base* and hybrid from *hybrid_only*."""
    return {
        "plaintext": base["plaintext"],
        "classical": base["classical"],
        "hybrid": hybrid_only["hybrid"],
    }


def fmt_gb(n: int) -> str:
    return f"{n / 1_000_000_000:.3f}"


def fmt_gib(n: int) -> str:
    return f"{n / (1024 ** 3):.3f}"


def build_table(
    faers_768: dict[str, int],
    ggvp_768: dict[str, int],
    faers_512: dict[str, int],
    ggvp_512: dict[str, int],
    faers_1024: dict[str, int],
    ggvp_1024: dict[str, int],
) -> str:
    configs = [
        ("KEM-768 (default)", faers_768, ggvp_768),
        ("KEM-512",           faers_512, ggvp_512),
        ("KEM-1024",          faers_1024, ggvp_1024),
    ]

    rows = []
    for config_label, faers, ggvp in configs:
        combined = {k: faers[k] + ggvp[k] for k in ("plaintext", "classical", "hybrid")}
        for dataset, totals in [
            (f"FAERS ({config_label})", faers),
            (f"GGVP ({config_label})",  ggvp),
            (f"Combined ({config_label})", combined),
        ]:
            plain = totals["plaintext"]
            classical = totals["classical"]
            hybrid = totals["hybrid"]
            rows.append(
                [
                    dataset,
                    str(plain),
                    fmt_gb(plain),
                    fmt_gib(plain),
                    str(classical),
                    fmt_gb(classical),
                    fmt_gib(classical),
                    str(hybrid),
                    fmt_gb(hybrid),
                    fmt_gib(hybrid),
                    str(hybrid - classical),
                    fmt_gb(hybrid - classical),
                ]
            )

    headers = [
        "Dataset",
        "Plaintext Bytes",
        "Plaintext GB",
        "Plaintext GiB",
        "Classical Bytes",
        "Classical GB",
        "Classical GiB",
        "Hybrid Bytes",
        "Hybrid GB",
        "Hybrid GiB",
        "Hybrid-Classical Bytes",
        "Hybrid-Classical GB",
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    out = [
        "Measured Storage Footprint Totals",
        "",
        render_row(headers),
        sep,
    ]
    for i, row in enumerate(rows):
        # Insert a blank separator line between configurations (every 3 rows)
        if i > 0 and i % 3 == 0:
            out.append("")
        out.append(render_row(row))
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a text table of measured storage footprint totals.")
    parser.add_argument("--faers-input",       default=str(RESULTS_DIR / "benchmark_results_faers_full.json"))
    parser.add_argument("--ggvp-input",        default=str(RESULTS_DIR / "benchmark_results_all_ggvp_full.json"))
    parser.add_argument("--faers-512-input",   default=str(RESULTS_DIR / "benchmark_results_faers_512.json"))
    parser.add_argument("--ggvp-512-input",    default=str(RESULTS_DIR / "benchmark_hybrid_all_ggvp_512.json"))
    parser.add_argument("--faers-1024-input",  default=str(RESULTS_DIR / "benchmark_hybrid_faers_1024.json"))
    parser.add_argument("--ggvp-1024-input",   default=str(RESULTS_DIR / "benchmark_hybrid_all_ggvp_1024.json"))
    parser.add_argument("--output",            default=str(RESULTS_DIR / "16_storage_footprint_totals.txt"))
    args = parser.parse_args()

    # KEM-768 (default) — files contain both classical and hybrid rows
    faers_768 = summarize_totals(load(Path(args.faers_input)))
    ggvp_768  = summarize_totals(load(Path(args.ggvp_input)))

    # KEM-512 — faers file is full; ggvp file is hybrid-only, so borrow plaintext/classical from KEM-768
    faers_512 = summarize_totals(load(Path(args.faers_512_input)))
    ggvp_512  = merge_hybrid(ggvp_768, summarize_totals(load(Path(args.ggvp_512_input))))

    # KEM-1024 — hybrid-only for both faers and ggvp; borrow plaintext/classical from KEM-768
    faers_1024 = merge_hybrid(faers_768, summarize_totals(load(Path(args.faers_1024_input))))
    ggvp_1024  = merge_hybrid(ggvp_768, summarize_totals(load(Path(args.ggvp_1024_input))))

    output = Path(args.output)
    output.write_text(build_table(faers_768, ggvp_768, faers_512, ggvp_512, faers_1024, ggvp_1024))


if __name__ == "__main__":
    main()
