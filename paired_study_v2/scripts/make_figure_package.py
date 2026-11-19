"""
Build the paper's figure package from the v2 analysis.

    ..\\.venv\\Scripts\\python.exe scripts/make_figure_package.py

Reads  results/analysis_v2/ (tables, summary.json, e3_per_record.csv.gz),
       Data/faers_q25k_manifest.csv (record size, quarter, content flags) and
       results/paired_v2/e3_main/chunk_*.csv.gz (sentinel rows, drift figure).
Writes analysis_package/figures/figNN_*.{pdf,png} and copies the matching
       tables to analysis_package/tables/.

numpy + matplotlib only (no pandas), so it runs wherever those two import.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# ── style ───────────────────────────────────────────────────────────────────
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_CLASSICAL, C_CAT3, C_CAT5 = MUTED, "#2a78d6", "#eb6834"   # tier identity, fixed across figures
C_SIZE, C_FLAG = INK2, "#1baf7a"                            # policy family identity
C_Q4, C_Q1 = INK2, "#2a78d6"                                # quarters (also marker shape)
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DOUBLE, SINGLE = 7.0, 3.5                                   # inches: two-column / one-column width

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 8,
    "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, out: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("  wrote", name)


def label_inside_color(fill: str) -> str:
    return INK if fill in ("#eda100",) else "white"


# ── loading (no pandas) ─────────────────────────────────────────────────────
def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by(rows, key):
    return {r[key]: r for r in rows}


def f(x):
    return float(x) if x not in ("", None) else np.nan


def load_per_record(path: Path):
    """e3_per_record.csv.gz -> {profile: {column: array}} aligned on record."""
    cols = ("size_bytes", "crypto_ms", "e2e_ms")
    data = defaultdict(lambda: defaultdict(dict))
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            for c in cols:
                data[r["profile"]][c][r["record"]] = float(r[c])
    records = sorted(data["baseline"]["crypto_ms"])
    out = {"record": np.array(records)}
    for p, d in data.items():
        out[p] = {c: np.array([d[c][k] for k in records]) for c in cols}
    out["size"] = out["baseline"]["size_bytes"]
    return out


def load_manifest(path: Path):
    return by(read_csv(path), "file")


def load_sentinels(run_dir: Path):
    """Sentinel rows of E3: per (profile, seq) the earliest time and median crypto / e2e."""
    acc = defaultdict(lambda: {"t": [], "c": [], "e": []})
    for fn in sorted(glob.glob(str(run_dir / "chunk_*.csv.gz"))):
        with gzip.open(fn, "rt", newline="") as fh:
            rd = csv.reader(fh)
            h = next(rd)
            ix = {k: h.index(k) for k in ("kind", "seq", "profile", "t_rel_s", "e2e_ms",
                                          "encrypt_ms", "sign_ms", "verify_ms", "decrypt_ms")}
            for row in rd:
                if row[ix["kind"]] != "sentinel":
                    continue
                a = acc[(row[ix["profile"]], int(row[ix["seq"]]))]
                a["t"].append(float(row[ix["t_rel_s"]]))
                a["e"].append(float(row[ix["e2e_ms"]]))
                a["c"].append(sum(float(row[ix[k]]) for k in ("encrypt_ms", "sign_ms", "verify_ms", "decrypt_ms")))
    out = defaultdict(lambda: {"t": [], "c": [], "e": []})
    for (p, _), a in sorted(acc.items(), key=lambda kv: (kv[0][0], min(kv[1]["t"]))):
        out[p]["t"].append(min(a["t"]) / 60)
        out[p]["c"].append(np.median(a["c"]))
        out[p]["e"].append(np.median(a["e"]))
    return {p: {k: np.array(v) for k, v in d.items()} for p, d in out.items()}


def boot_ci(a, b, n_boot=2000, seed=1, alpha=0.05):
    """Paired difference b - a: (mean, ci_lo, ci_hi, pct, pct_lo, pct_hi)."""
    rng = np.random.default_rng(seed)
    d = b - a
    n = len(d)
    md, pc = [], []
    for start in range(0, n_boot, 100):
        idx = rng.integers(0, n, size=(min(100, n_boot - start), n))
        dm = d[idx].mean(axis=1)
        md.append(dm)
        pc.append(dm / a[idx].mean(axis=1) * 100)
    md, pc = np.concatenate(md), np.concatenate(pc)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return (d.mean(), *np.percentile(md, q), d.mean() / a.mean() * 100, *np.percentile(pc, q))


# ── shared policy helpers ───────────────────────────────────────────────────
MAIN_FLAGS = ["death", "fatal_lt", "special", "sensitive", "severe", "severe_special", "serious"]
FLAG_SHORT = {"death": "Death", "fatal_lt": "Fatal / life-threat.", "special": "Special category",
              "sensitive": "Death + special", "severe": "Severe outcome",
              "severe_special": "Severe + special", "serious": "Any serious outcome"}


# Size-threshold sweep (must match analyze_paired.SWEEP): T in bytes -> % of records elevated.
SWEEP = {10901: 1, 5083: 5, 3558: 10, 2067: 25, 1332: 50}
COMBO = (5083, 1332)                                         # "death or size > T" policies


def sweep_label(t, sep="  "):
    return f"Largest {SWEEP[t]}%{sep}(> {t:,} B)"


def size_policies(rows):
    """Size-threshold continuum (sweep + original-paper thresholds + the two uniform tiers)."""
    keep = [r for r in rows if r["policy"].startswith(("adaptive", "uniform"))]
    return sorted(keep, key=lambda r: f(r["pct_records_elevated"]))


def flag_policies(rows):
    want = [f"flag_{r}" for r in MAIN_FLAGS] + [f"flagsize_death_{t}" for t in COMBO]
    return [r for r in rows if r["policy"] in want]


def short_name(policy):
    if policy.startswith("flag_"):
        return FLAG_SHORT.get(policy[5:], policy[5:])
    if policy.startswith("flagsize_death_"):
        return f"Death or > {int(policy.rsplit('_', 1)[1]):,} B"
    return policy


def panel_tag(ax, tag):
    ax.set_title(tag, loc="left")


# ── figures ─────────────────────────────────────────────────────────────────
def fig01_tier_ladder(T, out):
    L = by(read_csv(T / "table_tier_ladder.csv"), "policy")
    keys, names, cols = ["classical", "uniform_baseline", "uniform_strong"], ["Classical", "Cat. 3", "Cat. 5"], \
        [C_CLASSICAL, C_CAT3, C_CAT5]
    metrics = [("crypto_mean", "(a) Cryptographic time per record", "ms", 3),
               ("e2e_mean", "(b) End-to-end time per record", "ms", 2),
               ("overhead_mean_bytes", "(c) Storage overhead per record", "bytes", 0)]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5))
    for ax, (k, title, unit, dec) in zip(axes, metrics):
        v = [f(L[p][k]) for p in keys]
        ax.bar(names, v, color=cols, width=0.62, edgecolor="white", linewidth=1)
        for i, x in enumerate(v):
            lab = f"{x:,.{dec}f}"
            if i:
                lab += f"\n{x / v[0]:.1f}×" if unit == "bytes" else f"\n+{(x / v[0] - 1) * 100:.1f}%"
            ax.annotate(lab, (i, x), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=INK, linespacing=1.1)
        ax.set_ylim(0, max(v) * 1.35)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
        panel_tag(ax, title)
    fig.text(0.5, -0.04, "Percentages and multiples are relative to the classical baseline.",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout(w_pad=2)
    save(fig, out, "fig01_tier_cost_ladder")


def fig02_stage_breakdown(T, out):
    S = by(read_csv(T / "table_stage_breakdown.csv"), "metric")
    stages = [("encrypt_ms", "Encrypt"), ("sign_ms", "Sign"), ("db_insert_ms", "DB insert"),
              ("db_fetch_ms", "DB fetch"), ("verify_ms", "Verify"), ("decrypt_ms", "Decrypt")]
    tiers = [("classical_mean", "Classical"), ("baseline_mean", "Cat. 3"), ("strong_mean", "Cat. 5")]
    y = np.array([2, 1, 0])
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.2))
    left = np.zeros(3)
    for (m, lab), c in zip(stages, STAGE_COLORS):
        vals = np.array([f(S[m][col]) for col, _ in tiers])
        ax.barh(y, vals, left=left, color=c, edgecolor="white", linewidth=1.2, height=0.62, label=lab)
        for yi, l0, v in zip(y, left, vals):
            if v > 0.6:                                      # label only where it fits with padding
                ax.text(l0 + v / 2, yi, f"{lab}  {v:.2f}", ha="center", va="center", fontsize=6.8,
                        color=label_inside_color(c))
        left += vals
    for yi, t in zip(y, left):
        ax.text(t + 0.03, yi, f"{t:.2f} ms", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [n for _, n in tiers])
    ax.set_xlabel("Mean time per record (ms)")
    ax.set_xlim(0, left.max() * 1.1)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.2)
    save(fig, out, "fig02_stage_breakdown")


def fig03_primitives(summary, out):
    P = summary["e7_primitives_median_ms"]
    ops = [("enc_t_pq_encaps_ms", "ML-KEM\nencapsulate"), ("dec_t_pq_decaps_ms", "ML-KEM\ndecapsulate"),
           ("sig_sign_ms", "Signature\nsign"), ("sig_verify_ms", "Signature\nverify"),
           ("enc_t_ecdh_ms", "X25519\nkey agreement"), ("enc_t_aes_gcm_ms", "AES-256-GCM\nencrypt")]
    tiers = [("classical", "Classical (X25519 + ECDSA P-256)", C_CLASSICAL),
             ("baseline", "Cat. 3 (ML-KEM-768 + ML-DSA-65)", C_CAT3),
             ("strong", "Cat. 5 (ML-KEM-1024 + ML-DSA-87)", C_CAT5)]
    x = np.arange(len(ops))
    w = 0.27
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.7))
    for k, (p, lab, c) in enumerate(tiers):
        v = np.array([P[o][p] * 1000 for o, _ in ops])
        ax.bar(x + (k - 1) * w, v, width=w, color=c, edgecolor="white", linewidth=1, label=lab)
        for i, (o, _) in enumerate(ops):
            if o in ("sig_sign_ms", "sig_verify_ms"):
                ax.text(x[i] + (k - 1) * w, v[i] + 2, f"{v[i]:.0f}", ha="center", va="bottom", fontsize=6.8)
            if v[i] == 0:
                ax.text(x[i] + (k - 1) * w, 1.5, "none", ha="center", va="bottom", fontsize=6, color=INK2,
                        rotation=90)
    ax.set_xticks(x, [n for _, n in ops])
    ax.set_ylabel("Median time per operation (µs)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, out, "fig03_primitive_costs")


def fig04_deployment(summary, pr, out):
    e3_c3 = boot_ci(pr["classical"]["e2e_ms"], pr["baseline"]["e2e_ms"], seed=11)
    e3_c5 = boot_ci(pr["baseline"]["e2e_ms"], pr["strong"]["e2e_ms"], seed=12)

    def tup(d):
        return (d["delta_mean"], *d["delta_ci"], d["pct"], *d["pct_ci"])

    rows = [("Crypto only\n(no database, E7)", tup(summary["e7_nodb"]["classical_to_cat3"]["e2e"]),
             tup(summary["e7_nodb"]["e2e"])),
            ("Database,\nasync commit (E8)", tup(summary["e8_async_commit"]["classical_to_cat3"]["e2e"]),
             tup(summary["e8_async_commit"]["e2e"])),
            ("Database,\nsync commit (E3)", e3_c3, e3_c5)]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), sharey=True)
    y = np.arange(len(rows))[::-1]
    series = [(1, "Classical → Cat. 3", C_CAT3, "o", 0.12), (2, "Cat. 3 → Cat. 5", C_CAT5, "s", -0.12)]
    for ax, (off, scale, xl, tag) in zip(axes, [(0, 1000, "Added time per record (µs)", "(a) Absolute cost"),
                                                (3, 1, "Added time per record (%)", "(b) Relative cost")]):
        for k, lab, c, mk, dy in series:
            m = np.array([r[k][off] for r in rows]) * scale
            lo = np.array([r[k][off + 1] for r in rows]) * scale
            hi = np.array([r[k][off + 2] for r in rows]) * scale
            ax.errorbar(m, y + dy, xerr=[m - lo, hi - m], fmt=mk, color=c, ms=6, capsize=0, elinewidth=1.4,
                        label=lab, mec="white", mew=0.8)
            if off == 3:
                for mi, yi in zip(m, y + dy):
                    ax.text(mi + 2.0, yi, f"+{mi:.1f}%", va="center", fontsize=7)
        ax.set_xlabel(xl)
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        panel_tag(ax, tag)
    axes[1].set_xlim(0, 112)
    axes[0].set_xlim(0, 310)
    axes[0].set_yticks(y, [r[0] for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    axes[1].legend(loc="lower right")
    fig.tight_layout(w_pad=2)
    save(fig, out, "fig04_deployment_context")


def fig05_penalty_vs_size(pr, out):
    size = pr["size"]
    d5 = (pr["strong"]["crypto_ms"] - pr["baseline"]["crypto_ms"]) * 1000
    d3 = (pr["baseline"]["crypto_ms"] - pr["classical"]["crypto_ms"]) * 1000
    # log-spaced bins, merged left-to-right until each holds >= 200 records
    edges = np.logspace(np.log10(size.min()), np.log10(size.max() * 1.0001), 40)
    idx = np.digitize(size, edges)
    bins, cur = [], []
    for k in range(1, len(edges) + 1):
        cur.extend(np.flatnonzero(idx == k))
        if len(cur) >= 200:
            bins.append(np.array(cur))
            cur = []
    if cur:
        bins[-1] = np.concatenate([bins[-1], cur])
    xs = np.array([np.median(size[b]) for b in bins])
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(DOUBLE, 3.3), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    for d, lab, c in ((d3, "Classical → Cat. 3", C_CAT3), (d5, "Cat. 3 → Cat. 5", C_CAT5)):
        med = np.array([np.median(d[b]) for b in bins])
        q1 = np.array([np.percentile(d[b], 25) for b in bins])
        q3 = np.array([np.percentile(d[b], 75) for b in bins])
        ax.fill_between(xs, q1, q3, color=c, alpha=0.18, lw=0)
        ax.plot(xs, med, color=c, marker="o", ms=3.5, label=f"{lab}  (overall median {np.median(d):.0f} µs)")
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.set_ylabel("Added crypto time\nper record (µs)")
    ax.legend(loc="upper left", ncol=2)
    ax.text(0.99, 0.04, "line = median, band = interquartile range; size bins of ≥ 200 records",
            transform=ax.transAxes, ha="right", fontsize=6.8, color=INK2)
    hedges = np.logspace(np.log10(size.min()), np.log10(size.max()), 60)
    axh.hist(size, bins=hedges, color=MUTED, edgecolor="white", linewidth=0.4)
    axh.set_yscale("log")
    axh.set_ylabel("Records\n(log)")
    axh.set_xlabel("Plaintext record size (bytes, log scale)")
    axh.grid(axis="x", visible=False)
    ax.set_xlim(size.min() * 0.9, size.max() * 1.1)
    save(fig, out, "fig05_penalty_vs_record_size")


def fig06_size_vs_death(manifest, out):
    size = np.array([f(r["size_bytes"]) for r in manifest.values()])
    death = np.array([r["flag_death"] == "1" for r in manifest.values()])
    xs = np.logspace(np.log10(size.min()), np.log10(size.max()), 400)

    def surv(a):
        a = np.sort(a)
        return 100 * (1 - np.searchsorted(a, xs, side="right") / len(a))

    fig, ax = plt.subplots(figsize=(SINGLE + 1.0, 2.8))
    ax.plot(xs, surv(size[~death]), color=MUTED, label=f"Other reports (n = {(~death).sum():,})")
    ax.plot(xs, surv(size[death]), color=C_FLAG, label=f"Death reports (n = {death.sum():,})")
    for t in (1332, 5083):
        sd = 100 * (size[death] > t).mean()
        so = 100 * (size[~death] > t).mean()
        ax.axvline(t, color=AXIS, lw=0.8, zorder=0)
        ax.text(t * 1.05, 99, f"T = {t:,} B", fontsize=7, color=INK2, va="top", ha="left", rotation=90)
        ax.annotate(f"{sd:.0f}% of death\nreports above T", (t, sd), xytext=(t * 1.8, sd + 12),
                    fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2))
        ax.plot([t], [sd], "o", color=C_FLAG, ms=4.5, mec="white", mew=0.8)
        ax.plot([t], [so], "o", color=MUTED, ms=4.5, mec="white", mew=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0, 102)
    ax.set_xlabel("Plaintext record size (bytes, log scale)")
    ax.set_ylabel("Reports larger than size (%)")
    ax.legend(loc="upper right")
    save(fig, out, "fig06_record_size_death_vs_other")


def fig07_cost_vs_elevation(T, out):
    lat = read_csv(T / "table_policy_latency.csv")
    sto = by(read_csv(T / "table_policy_storage.csv"), "policy")
    size, flag = size_policies(lat), flag_policies(lat)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    specs = [(lambda r: f(r["e2e_delta_pct"]), "End-to-end latency vs Cat. 3 (%)", "(a) Latency"),
             (lambda r: f(sto[r["policy"]]["delta_footprint_pct"]), "Total stored bytes vs Cat. 3 (%)",
              "(b) Storage footprint")]
    show = {"flag_death", "flag_severe", "flag_serious"}
    for ax, (g, yl, tag) in zip(axes, specs):
        ax.plot([f(r["pct_records_elevated"]) for r in size], [g(r) for r in size], color=C_SIZE, marker="o",
                ms=3.5, label="Size-threshold policies")
        ax.plot([f(r["pct_records_elevated"]) for r in flag], [g(r) for r in flag], ls="none", marker="^", ms=6.5,
                color=C_FLAG, mec="white", mew=0.8, label="Content-flag policies")
        for r in flag:
            if r["policy"] in show:
                above = r["policy"] == "flag_death"          # lowest point: label above, with a leader line
                ax.annotate("Flag: " + short_name(r["policy"]).lower(), (f(r["pct_records_elevated"]), g(r)),
                            xytext=(4, 30) if above else (7, -4), textcoords="offset points",
                            ha="left" if above else "left", va="bottom" if above else "top", fontsize=7,
                            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2) if above else None)
        end = size[-1]
        ax.annotate(f"Uniform Cat. 5: +{g(end):.1f}%", (100, g(end)), xytext=(-8, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7)
        ax.set_xlabel("Records elevated to Cat. 5 (%)")
        ax.set_ylabel(yl)
        ax.set_xlim(-2, 104)
        ax.set_ylim(0, g(end) * 1.15)
        panel_tag(ax, tag)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "fig07_cost_vs_share_elevated")


def fig08_coverage(T, out):
    cov = read_csv(T / "table_policy_coverage.csv")
    size, flag = size_policies(cov), flag_policies(cov)
    eq = by(cov, "policy")
    panels = [("pct_death_protected", "(a) Death reports", "death", "flag: death"),
              ("pct_special_protected", "(b) Special-category reports", "special", "flag: special category"),
              ("pct_severe_protected", "(c) Severe-outcome reports", "severe", "flag: severe outcome")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.7), sharey=True)
    sx = np.array([f(r["pct_records_elevated"]) for r in size])
    for ax, (col, tag, hero, hero_lab) in zip(axes, panels):
        sy = np.array([f(r[col]) for r in size])
        ax.plot(sx, sy, color=C_SIZE, marker="o", ms=3.2, label="Size-threshold policies")
        ax.plot([f(r["pct_records_elevated"]) for r in flag], [f(r[col]) for r in flag], ls="none", marker="^",
                ms=6.5, color=C_FLAG, mec="white", mew=0.8, label="Content-flag policies")
        h = next(r for r in flag if r["policy"] == f"flag_{hero}")
        hx, hy = f(h["pct_records_elevated"]), f(h[col])
        # exact equal-share size policy (computed in analyze_paired), not an interpolation along the sweep
        e = eq[f"sizeeq_{hero}"]
        ex, same = f(e["pct_records_elevated"]), f(e[col])
        t_eq = int(e["label"].rsplit("T = ", 1)[1].split(" ")[0].replace(",", ""))
        ax.annotate(f"{hero_lab}\n({hx:.1f}% elevated)", (hx, hy), xytext=(max(hx + 12, 36), 88),
                    fontsize=7, va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=C_FLAG))
        ax.plot([ex], [same], "o", ms=5.5, mfc="white", mec=C_SIZE, mew=1.2, zorder=4)
        ax.annotate(f"size > {t_eq:,} B, same cost\n→ {same:.0f}%", (ex, same), xytext=(44, 9),
                    fontsize=7, va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2))
        ax.set_xlabel("Records elevated to Cat. 5 (%)")
        ax.set_xlim(-2, 104)
        ax.set_ylim(-3, 106)
        panel_tag(ax, tag)
    axes[0].set_ylabel("Reports protected at Cat. 5 (%)")
    fig.tight_layout(w_pad=1.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    save(fig, out, "fig08_protection_coverage")


def fig09_decision_cost(summary, T, out):
    D = summary["e5_policy_decision_us"]
    lat = by(read_csv(T / "table_policy_latency.csv"), "policy")
    cat5_crypto = f(lat["uniform_strong"]["crypto_delta_mean_ms"]) * 1000
    cat5_e2e = f(lat["uniform_strong"]["e2e_delta_mean_ms"]) * 1000
    death_share = f(lat["flag_death"]["pct_records_elevated"]) / 100
    rows = [("Fixed tier (no decision)", D["baseline"]["median"], D["baseline"]["p99"], MUTED, "decision"),
            ("Size threshold (T = 4,196 B)", D["adaptive:4196"]["median"], D["adaptive:4196"]["p99"], C_SIZE,
             "decision"),
            ("Flag: death", D["flag:death"]["median"], D["flag:death"]["p99"], C_FLAG, "decision"),
            ("Flag: death or > 4,196 B", D["flagsize:death:4196"]["median"], D["flagsize:death:4196"]["p99"], C_FLAG,
             "decision"),
            ("Flag: severe outcome", D["flag:severe"]["median"], D["flag:severe"]["p99"], C_FLAG, "decision"),
            ("Flag: death or special category", D["flag:sensitive"]["median"], D["flag:sensitive"]["p99"], C_FLAG,
             "decision"),
            (f"Flag: death, Cat. 5 cost averaged\nover all records ({death_share * 100:.1f}% × {cat5_e2e:.0f} µs)",
             death_share * cat5_e2e, None, C_CAT5, "ref"),
            ("Cat. 5 cost per elevated record\n(crypto only)", cat5_crypto, None, C_CAT5, "ref"),
            ("Cat. 5 cost per elevated record\n(end-to-end)", cat5_e2e, None, C_CAT5, "ref")]
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.2))
    y = np.arange(len(rows))[::-1].astype(float)
    y[-3:] -= 0.0
    y[:6] += 0.6                                             # gap between the two groups
    for yi, (lab, med, p99, c, kind) in zip(y, rows):
        ax.barh(yi, med, color=c, height=0.62, edgecolor="white", linewidth=1)
        if p99 is not None:
            ax.plot([med, p99], [yi, yi], color=INK2, lw=0.8)
            ax.plot([p99], [yi], "|", color=INK2, ms=6)
        ax.text(max(med, p99 or 0) + 2, yi, f"{med:.1f} µs" + (f"   (p99 {p99:.0f})" if p99 else ""),
                va="center", fontsize=7)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Time per record (µs)")
    ax.set_xlim(0, 175)
    ax.grid(axis="y", visible=False)
    ax.text(0, y[0] + 0.75, "Policy decision (classify the serialized record; median, whisker to p99)",
            fontsize=7.5, fontweight="bold", va="bottom")
    ax.text(0, y[6] + 0.55, "For comparison: the Cat. 5 cost the decision is buying",
            fontsize=7.5, fontweight="bold", va="bottom")
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.3)
    save(fig, out, "fig09_policy_decision_cost")


def fig10_e5_validation(summary, out):
    V = summary["e5_validation"]
    names = {"adaptive:1458": "Size > 1,458 B", "adaptive:4196": "Size > 4,196 B", "flag:death": "Flag: death",
             "flag:sensitive": "Flag: death + special", "flag:severe": "Flag: severe outcome",
             "flagsize:death:4196": "Death or > 4,196 B",
             **{f"adaptive:{t}": f"Largest {SWEEP[t]}% (> {t:,} B)" for t in SWEEP},
             **{f"flagsize:death:{t}": f"Death or > {t:,} B" for t in COMBO}}
    # order: runs in sequence; within a run, size policies by share, then flags, then combined
    runs = list(dict.fromkeys(v.get("run", "e5_direct") for v in V))
    pct_of = {v["profile"]: v["pct_elevated"] for v in V}
    fam = lambda p: {"adaptive": 0, "flag": 1, "flagsize": 2}[p.split(":")[0]]
    profs, run_of = [], {}
    for r in runs:
        ps = list(dict.fromkeys(v["profile"] for v in V if v.get("run", "e5_direct") == r))
        ps.sort(key=lambda p: (fam(p), pct_of[p]))
        profs += ps
        run_of.update({p: r for p in ps})
    gap = 0.8                                                   # vertical space between the runs
    ypos, yy = [], 0.0
    for k, p in enumerate(profs):
        if k and run_of[p] != run_of[profs[k - 1]]:
            yy += gap
        ypos.append(yy)
        yy += 1
    y = -np.array(ypos)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 0.95 + 0.23 * (len(profs) + gap * (len(runs) - 1))),
                             sharey=True)
    for ax, metric, tag in zip(axes, ("crypto_ms", "e2e_ms"), ("(a) Crypto time", "(b) End-to-end time")):
        ax.axvspan(-2, 2, color=GRID, alpha=0.45, lw=0)
        for b in (-2, 2):
            ax.axvline(b, color=AXIS, lw=0.8)
        ax.axvline(0, color=AXIS, lw=0.8)
        for yi, p in zip(y, profs):
            v = next(v for v in V if v["profile"] == p and v["metric"] == metric)
            m = v["mean_diff_ms"] / v["derived_mean"] * 100
            lo, hi = (np.array(v["ci90_ms"]) / v["derived_mean"] * 100)
            c = C_FLAG if p.startswith("flag") else C_SIZE
            ax.errorbar([m], [yi], xerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=5.5, elinewidth=1.4,
                        mec="white", mew=0.8)
        ax.text(1.9, y[0] + 0.55, "equivalence margin ±2%", ha="right", fontsize=6.8, color=INK2)
        ax.set_xlim(-3, 3)
        ax.set_xlabel("Direct − derived (% of derived mean), 90% CI")
        ax.grid(axis="y", visible=False)
        panel_tag(ax, tag)
        if len(runs) > 1:                                       # divider between the two E5 runs
            for k in range(1, len(profs)):
                if run_of[profs[k]] != run_of[profs[k - 1]]:
                    ax.axhline((y[k] + y[k - 1]) / 2, color=INK2, lw=0.6, ls=(0, (3, 2)))
    axes[0].set_yticks(y, [f"{names.get(p, p)}  ({pct_of[p]:.1f}%)" for p in profs])
    axes[0].set_ylim(y[-1] - 0.6, y[0] + 0.9)
    if len(runs) > 1:
        run_lab = {"e5_direct": "E5, 2026-11-05", "e5_direct_sweep": "E5 sweep, 2026-11-11"}
        for r in runs:
            ys = [yv for yv, p in zip(y, profs) if run_of[p] == r]
            axes[0].text(-2.95, max(ys) + 0.45, run_lab.get(r, r),
                         ha="left", va="bottom", fontsize=6.8, color=INK2, fontweight="bold")
    fig.tight_layout(w_pad=2)
    save(fig, out, "fig10_derived_vs_direct_validation")


def fig11_quarters(T, pr, manifest, out):
    q = np.array([manifest[r]["source_quarter"] for r in pr["record"]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 3.6), gridspec_kw={"width_ratios": [1, 1.7]})
    # (a) per-record Cat. 3 -> Cat. 5 penalty by quarter
    items = [("Crypto", "crypto_ms"), ("End-to-end", "e2e_ms")]
    for k, (lab, m) in enumerate(items):
        for qq, c, mk, dy in (("25Q4", C_Q4, "o", 0.12), ("26Q1", C_Q1, "s", -0.12)):
            sel = q == qq
            r = boot_ci(pr["baseline"][m][sel], pr["strong"][m][sel], seed=20 + k)
            ax1.errorbar([r[0] * 1000], [1 - k + dy], xerr=[[(r[0] - r[1]) * 1000], [(r[2] - r[0]) * 1000]],
                         fmt=mk, color=c, ms=6, elinewidth=1.4, mec="white", mew=0.8,
                         label={"25Q4": "2025 Q4", "26Q1": "2026 Q1"}[qq] if k == 0 else None)
    ax1.set_yticks([1, 0], [i[0] for i in items])
    ax1.set_ylim(-0.6, 1.6)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Cat. 3 → Cat. 5 added time per record (µs)")
    ax1.grid(axis="y", visible=False)
    ax1.legend(loc="lower right")
    panel_tag(ax1, "(a) Per-record penalty")
    # (b) policy cost per quarter
    rows = read_csv(T / "table_policy_by_quarter.csv")
    want = ([f"adaptive_{t}" for t in SWEEP] + [f"flag_{r}" for r in MAIN_FLAGS]
            + [f"flagsize_death_{t}" for t in COMBO] + ["uniform_strong"])
    label = {**{f"adaptive_{t}": sweep_label(t, " ") for t in SWEEP},
             **{f"flag_{r}": "Flag: " + FLAG_SHORT[r].lower() for r in MAIN_FLAGS},
             **{f"flagsize_death_{t}": f"Death or > {t:,} B" for t in COMBO},
             "uniform_strong": "Uniform Cat. 5"}
    R = {(r["quarter"], r["policy"]): r for r in rows}
    y = np.arange(len(want))[::-1]
    for qq, c, mk, dy in (("25Q4", C_Q4, "o", 0.15), ("26Q1", C_Q1, "s", -0.15)):
        m = np.array([f(R[(qq, p)]["e2e_delta_pct"]) for p in want])
        lo = np.array([f(R[(qq, p)]["e2e_pct_ci_lo"]) for p in want])
        hi = np.array([f(R[(qq, p)]["e2e_pct_ci_hi"]) for p in want])
        ax2.errorbar(m, y + dy, xerr=[m - lo, hi - m], fmt=mk, color=c, ms=4.5, elinewidth=1.2, mec="white",
                     mew=0.6)
    ax2.set_yticks(y, [label[p] for p in want], fontsize=7)
    ax2.set_xlabel("End-to-end latency vs Cat. 3 (%), 95% CI")
    ax2.set_xlim(left=0)
    ax2.grid(axis="y", visible=False)
    panel_tag(ax2, "(b) Policy cost by quarter")
    fig.tight_layout(w_pad=2)
    save(fig, out, "fig11_quarter_robustness")


def fig12_stability(summary, sent, out):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    tiers = [("classical", "Classical", C_CLASSICAL), ("baseline", "Cat. 3", C_CAT3), ("strong", "Cat. 5", C_CAT5)]
    for ax, key, yl, tag in ((axes[0], "c", "Crypto time (ms)", "(a) Sentinel record: crypto"),
                             (axes[1], "e", "End-to-end time (ms)", "(b) Sentinel record: end-to-end")):
        for p, lab, c in tiers:
            if p in sent:
                ax.plot(sent[p]["t"], sent[p][key], color=c, lw=1.0, label=lab)
        ax.set_xlabel("Minutes into E3")
        ax.set_ylabel(yl)
        ax.set_ylim(bottom=0)
        panel_tag(ax, tag)
    axes[0].legend(loc="lower right", ncol=3, handlelength=1.2, columnspacing=0.8)
    st = summary["stability"]
    names = {"e3_subset": "E3\nsubset", "e6_stability_r1": "E6\nstart", "e6_stability_r2": "E6\nmiddle",
             "e6_stability_r3": "E6\nend"}
    x = np.arange(len(st))
    for key, lab, c, mk, dx in (("crypto", "Crypto", INK2, "o", -0.1), ("e2e", "End-to-end", MUTED, "s", 0.1)):
        m = np.array([r[key]["delta_mean"] for r in st]) * 1000
        lo = np.array([r[key]["delta_ci"][0] for r in st]) * 1000
        hi = np.array([r[key]["delta_ci"][1] for r in st]) * 1000
        axes[2].errorbar(x + dx, m, yerr=[m - lo, hi - m], fmt=mk, color=c, ms=5, elinewidth=1.3, mec="white",
                         mew=0.8, label=lab)
    axes[2].set_xticks(x, [names.get(r["run"], r["run"]) for r in st], fontsize=7)
    axes[2].set_ylim(0, 110)
    axes[2].set_ylabel("Cat. 3 → Cat. 5 added time (µs)")
    axes[2].legend(loc="lower right")
    axes[2].grid(axis="x", visible=False)
    panel_tag(axes[2], "(c) Between runs")
    fig.tight_layout(w_pad=1.8)
    save(fig, out, "fig12_stability_and_drift")


# ── per-policy figures (policy on the x axis) ───────────────────────────────
N_RECORDS = 50_000
POLICY_ORDER = (
    [("uniform_baseline", "Uniform Cat. 3", "uniform")]
    + [(f"adaptive_{t}", f"Size > {t:,} B", "size") for t in SWEEP]
    + [(f"flag_{r}", FLAG_SHORT[r], "flag")
       for r in ("death", "special", "fatal_lt", "sensitive", "severe", "severe_special", "serious")]
    + [(f"flagsize_death_{t}", f"Death or > {t:,} B", "combo") for t in COMBO]
    + [("uniform_strong", "Uniform Cat. 5", "uniform")]
)
GROUPS = [("size", "Size threshold"), ("flag", "Content flag"), ("combo", "Flag or size")]


def _policy_style(policy, fam):
    if policy == "uniform_baseline":
        return C_CAT3, "s"
    if policy == "uniform_strong":
        return C_CAT5, "s"
    return (C_SIZE, "o") if fam == "size" else (C_FLAG, "^")


CLASSICAL_ROW = ("classical", "Classical (reference)", "classical")


def _policy_axis(ax, order, rows_pct):
    """Shared x-axis treatment: tick labels with % elevated, family separators."""
    x = np.arange(len(order))
    ax.set_xticks(x, [lab if fam == "classical" else f"{lab}  ({rows_pct[p]:.1f}%)" for p, lab, fam in order],
                  rotation=40, ha="right", rotation_mode="anchor", fontsize=7)
    fams = [fam for _, _, fam in order]
    for k in range(1, len(fams)):
        if fams[k] != fams[k - 1]:
            ax.axvline(k - 0.5, color=INK2 if fams[k - 1] == "classical" else AXIS, lw=0.8, zorder=0)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.grid(axis="x", visible=False)
    return x


def _group_headers(ax, order):
    fams = [fam for _, _, fam in order]
    for fam, title in GROUPS:
        idx = [i for i, f_ in enumerate(fams) if f_ == fam]
        ax.text(np.mean(idx), 1.02, title, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=INK2, fontweight="bold")


def _latency_by_policy(T, out, m, ylabel, name):
    """One metric (e2e or crypto), policies on x, y-axis scaled to include the classical baseline."""
    lat = by(read_csv(T / "table_policy_latency.csv"), "policy")
    lad = by(read_csv(T / "table_tier_ladder.csv"), "policy")
    order = [CLASSICAL_ROW] + POLICY_ORDER
    pct = {p: f(lat[p]["pct_records_elevated"]) for p, _, _ in POLICY_ORDER}
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.9))
    x = _policy_axis(ax, order, pct)
    cl = f(lad["classical"][f"{m}_mean"])
    base = f(lat["uniform_baseline"][f"{m}_mean"])
    top = f(lat["uniform_strong"][f"{m}_mean"])
    for xi, (p, lab, fam) in zip(x, order):
        if fam == "classical":
            ax.plot([xi], [cl], marker="s", color=C_CLASSICAL, ms=6, mec="white", mew=0.8, ls="none")
            continue
        r = lat[p]
        mean = f(r[f"{m}_mean"])
        lo, hi = f(r[f"{m}_delta_ci_lo"]), f(r[f"{m}_delta_ci_hi"])
        c, mk = _policy_style(p, fam)
        if not np.isnan(lo):                          # paired CI of the difference, centred on the policy mean
            d = f(r[f"{m}_delta_mean_ms"])
            ax.errorbar([xi], [mean], yerr=[[d - lo], [hi - d]], fmt="none", ecolor=c, elinewidth=1.4, capsize=0)
        ax.plot([xi], [mean], marker=mk, color=c, ms=7 if mk == "^" else 6, mec="white", mew=0.8, ls="none")
    for v, c in ((cl, C_CLASSICAL), (base, C_CAT3), (top, C_CAT5)):
        ax.axhline(v, color=c, lw=0.8, alpha=0.7, zorder=0)
    xl = 0.6                                           # labels start just right of the classical column
    ax.text(xl, top, f"Uniform Cat. 5  {top:.3f} ms   (+{(top / cl - 1) * 100:.1f}% vs classical, "
            f"+{(top / base - 1) * 100:.1f}% vs Cat. 3)", ha="left", va="bottom", fontsize=6.8, color=INK2)
    ax.annotate(f"Uniform Cat. 3  {base:.3f} ms   (+{(base / cl - 1) * 100:.1f}% vs classical)", (xl, base),
                xytext=(0, -7), textcoords="offset points", ha="left", va="top", fontsize=6.8, color=INK2)
    ax.text(len(x) - 0.45, cl, f"Classical  {cl:.3f} ms", ha="right", va="bottom", fontsize=6.8, color=INK2)
    span = top - cl
    ax.set_ylim(cl - 0.12 * span, top + 0.14 * span)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Elevation policy  (share of records elevated to Cat. 5)")
    _group_headers(ax, order)
    save(fig, out, name)


def fig13_latency_by_policy(T, out):
    _latency_by_policy(T, out, "e2e", "Mean end-to-end time per record (ms)", "fig13a_e2e_latency_by_policy")
    _latency_by_policy(T, out, "crypto", "Mean cryptographic time per record (ms)",
                       "fig13b_crypto_latency_by_policy")


def fig14_storage_by_policy(T, out):
    _storage_by_policy(T, out, total=False, name="fig14_storage_by_policy")


def fig15_total_storage_by_policy(T, out):
    _storage_by_policy(T, out, total=True, name="fig15_total_storage_by_policy")


def _size_histogram(a, size, quarter, tag, note_10k=True):
    """Histogram of record sizes on log-spaced bins, quarters stacked, with median / p95 / p99 marked."""
    n, total = len(size), size.sum()
    # linear-interpolation percentiles: p50 / p95 / p99 then equal the sweep's 50% / 5% / 1% thresholds
    med, mean, p95, p99 = np.median(size), size.mean(), *np.percentile(size, [95, 99])
    bins = np.logspace(np.log10(600), np.log10(500_000), 61)
    q4, q1 = size[quarter == "25Q4"], size[quarter == "26Q1"]
    a.hist([q4, q1], bins=bins, stacked=True, color=[C_Q4, C_Q1], edgecolor="white", linewidth=0.3,
           label=[f"2025 Q4 (n = {len(q4):,})", f"2026 Q1 (n = {len(q1):,})"])
    top = a.get_ylim()[1]
    for v, lab, ls in ((med, f"Median {med:,.0f} B", "--"), (p95, f"p95 {p95:,.0f} B", ":"),
                       (p99, f"p99 {p99:,.0f} B", ":")):
        a.axvline(v, color=INK, lw=0.9, ls=ls, zorder=3)
        a.text(v * 1.07, top * 0.97, lab, rotation=90, va="top", ha="left", fontsize=6.8, color=INK)
    if note_10k:
        above10k = (size > 10_240).mean() * 100
        a.annotate(f"Original paper's lowest\nthreshold (10 KB):\n{above10k:.1f}% of records above",
                   (12_000, top * 0.015), xytext=(30_000, top * 0.30), fontsize=6.8, color=INK2,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2))
    a.set_xscale("log")
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x / 1000:,.0f} KB" if x >= 1000 else f"{x:,.0f} B"))
    a.set_xlim(500, 600_000)
    a.set_xlabel("Plaintext JSON record size (log scale)")
    a.set_ylabel("Records per bin")
    a.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    panel_tag(a, f"{tag}{n:,} records, {total / 1e6:.1f} MB, mean {mean:,.0f} B")


def fig16a_size_histogram(manifest, out):
    """Panel (a) of Figure 16 on its own."""
    size = np.array([f(r["size_bytes"]) for r in manifest.values()])
    quarter = np.array([r["source_quarter"] for r in manifest.values()])
    fig, a = plt.subplots(figsize=(DOUBLE * 0.62, 2.9))
    _size_histogram(a, size, quarter, tag="", note_10k=False)
    panel_tag(a, "Dataset Size Distribution")
    # Crop the empty tail: above 30 KB the bars are too short to see (state the count in the caption).
    xmax = 30_000
    print(f"    fig16a: {(size > xmax).sum()} records ({(size > xmax).mean() * 100:.2f}%) above {xmax:,} B "
          f"not shown; largest {size.max():,.0f} B")
    a.set_xlim(600, xmax)
    a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1000, 2000, 5000, 10_000, 20_000]))
    a.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(1, 10)))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # Legend in the empty space between the median and p95 lines, above the bars.
    h, l = a.get_legend_handles_labels()
    a.legend(h, [x.replace(" (n", "\n(n") for x in l], loc="upper left", bbox_to_anchor=(0.29, 0.98),
             labelspacing=0.8, handlelength=1.2)
    fig.tight_layout()
    save(fig, out, "fig16a_record_size_histogram")


def fig16_size_distribution(manifest, out):
    """Plaintext size of the 50,000 JSON records: (a) histogram, (b) share of records and of bytes above a size."""
    size = np.array([f(r["size_bytes"]) for r in manifest.values()])
    quarter = np.array([r["source_quarter"] for r in manifest.values()])
    n, total = len(size), size.sum()
    s = np.sort(size)
    C_BYTES = "#eda100"

    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    _size_histogram(a, size, quarter, tag="(a) ")

    # (b) records vs bytes above a size
    xs = np.logspace(np.log10(s[0]), np.log10(s[-1]), 500)
    k = np.searchsorted(s, xs, side="right")
    cum = np.concatenate([[0], np.cumsum(s)])
    rec_pct = 100 * (n - k) / n
    byte_pct = 100 * (total - cum[k]) / total
    b.plot(xs, byte_pct, color=C_BYTES, label="Share of all plaintext bytes")
    b.plot(xs, rec_pct, color=INK2, label="Share of records")
    for t, pos in ((1332, (0.03, 0.42)), (5083, (0.42, 0.28))):   # labels sit below the record curve
        r, by_ = 100 * (size > t).mean(), 100 * size[size > t].sum() / total
        b.axvline(t, color=AXIS, lw=0.8, zorder=0)
        b.plot([t], [r], "o", color=INK2, ms=4, mec="white", mew=0.8, zorder=4)
        b.plot([t], [by_], "o", color=C_BYTES, ms=4, mec="white", mew=0.8, zorder=4)
        b.annotate(f"Above {t:,} B:\n{r:.1f}% of records\n{by_:.1f}% of bytes", (t, r),
                   xytext=pos, textcoords="axes fraction", ha="left", va="top", fontsize=6.8,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2, shrinkB=3))
    b.set_xscale("log")
    b.set_yscale("log")
    b.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x / 1000:,.0f} KB" if x >= 1000 else f"{x:,.0f} B"))
    b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}%"))
    b.set_ylim(0.002, 130)
    b.set_xlabel("Record size threshold (log scale)")
    b.set_ylabel("Above the threshold (log scale)")
    b.legend(loc="upper right")
    panel_tag(b, "(b) Records and bytes above a size")
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "fig16_record_size_distribution")


def _storage_by_policy(T, out, total, name):
    """Stacked payload + crypto overhead per policy.  total=False: mean bytes per record (KB);
    total=True: total stored bytes for the whole 50,000-record table (MB)."""
    sto = by(read_csv(T / "table_policy_storage.csv"), "policy")
    lad = by(read_csv(T / "table_tier_ladder.csv"), "policy")
    pct = {p: f(sto[p]["pct_records_elevated"]) for p, _, _ in POLICY_ORDER}
    k = 1 if total else 1 / N_RECORDS                     # multiplier applied to the table's totals
    unit = 1e6 if total else 1e3                          # MB or KB on the axis
    b = sto["uniform_baseline"]
    payload = (f(b["total_stored_bytes"]) - f(b["overhead_bytes"])) * k
    base_total = f(b["total_stored_bytes"]) * k
    cl_over = f(lad["classical"]["overhead_mean_bytes"]) * N_RECORDS * k
    rows = [("classical", "Classical (reference)", "classical", cl_over)] + \
           [(p, lab, fam, f(sto[p]["overhead_bytes"]) * k) for p, lab, fam in POLICY_ORDER]
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.6))
    x = np.arange(len(rows))
    C_PAYLOAD = "#d9d8d2"
    ymax = (payload + f(sto["uniform_strong"]["overhead_bytes"]) * k) * 1.12
    for xi, (p, lab, fam, over) in zip(x, rows):
        c = C_CLASSICAL if fam == "classical" else _policy_style(p, fam)[0]
        ax.bar(xi, payload, width=0.68, color=C_PAYLOAD, edgecolor="white", linewidth=1)
        ax.bar(xi, over, bottom=payload, width=0.68, color=c, edgecolor="white", linewidth=1)
        tot = payload + over
        if p in ("classical", "uniform_baseline"):
            lab_txt = f"{tot / unit:.1f}" if total else f"{tot / unit:.2f}"
        elif total:
            lab_txt = f"+{(tot - base_total) / unit:.1f}"
        else:
            lab_txt = f"+{(tot / base_total - 1) * 100:.1f}%"
        ax.text(xi, tot + ymax * 0.01, lab_txt, ha="center", va="bottom", fontsize=6.5, color=INK)
    pcts = {"classical": np.nan, **pct}
    ax.set_xticks(x, [lab if p == "classical" else f"{lab}  ({pcts[p]:.1f}%)" for p, lab, _, _ in rows],
                  rotation=40, ha="right", rotation_mode="anchor", fontsize=7)
    fams = [fam for _, _, fam, _ in rows]
    for k in range(1, len(fams)):
        if fams[k] != fams[k - 1] and not (fams[k - 1] == "classical"):
            ax.axvline(k - 0.5, color=AXIS, lw=0.8, zorder=0)
    ax.axvline(0.5, color=INK2, lw=0.8, zorder=0)
    for fam, title in GROUPS:
        idx = [i for i, f_ in enumerate(fams) if f_ == fam]
        ax.text(np.mean(idx), 1.02, title, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=INK2, fontweight="bold")
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / unit:,.0f}"))
    if total:
        ax.set_ylabel(f"Total stored bytes, {N_RECORDS:,} records (MB)")
        ax.text(0.01, 0.97, "Bar labels: total MB (references); MB added vs uniform Cat. 3 (policies)",
                transform=ax.transAxes, va="top", fontsize=6.8, color=INK2)
        pay_lab = f"Encrypted payload ({payload / unit:,.1f} MB, all policies)"
    else:
        ax.set_ylabel("Mean stored bytes per record (KB)")
        pay_lab = f"Encrypted payload ({payload:,.0f} B, all policies)"
    ax.set_xlabel("Elevation policy  (share of records elevated to Cat. 5)")
    ax.grid(axis="x", visible=False)
    handles = [matplotlib.patches.Patch(color=C_PAYLOAD, label=pay_lab),
               matplotlib.patches.Patch(color=C_CAT3, label="Crypto overhead: Cat. 3"),
               matplotlib.patches.Patch(color=C_SIZE, label="… size-threshold policy"),
               matplotlib.patches.Patch(color=C_FLAG, label="… content-flag policy"),
               matplotlib.patches.Patch(color=C_CAT5, label="… Cat. 5")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=5, fontsize=6.8,
              handlelength=1.0, columnspacing=0.9)
    save(fig, out, name)


# ── size-threshold sweep in isolation (figures 17-19, table_size_sweep) ─────
SWEEP_UP = sorted(SWEEP, key=SWEEP.get)                    # ascending share: 1, 5, 10, 25, 50 %
REPORT_COLORS = {"death": "#1baf7a", "special": "#eda100", "severe": "#e87ba4"}


def _sweep_ticks(ax, x, with_cat5=True, refs_before=()):
    """Tick labels: share elevated over threshold; references (e.g. Cat. 3) can precede the sweep."""
    labs = (list(refs_before) + [f"{SWEEP[t]}%\n{t:,}" for t in SWEEP_UP]
            + (["Uniform\nCat. 5"] if with_cat5 else []))
    ax.set_xticks(x, labs, fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_xlabel("Largest share of records elevated / threshold T (bytes)")


def fig17_size_sweep_latency(T, out):
    lat = by(read_csv(T / "table_policy_latency.csv"), "policy")
    pols = [f"adaptive_{t}" for t in SWEEP_UP] + ["uniform_strong"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9))
    for ax, key, yl, tag in ((axes[0], "e2e", "End-to-end time vs Cat. 3 (%)", "(a) End-to-end latency"),
                             (axes[1], "crypto", "Cryptographic time vs Cat. 3 (%)", "(b) Cryptographic latency")):
        x = np.arange(len(pols))
        full = f(lat["uniform_strong"][f"{key}_delta_pct"])
        for xi, p in zip(x, pols):
            r = lat[p]
            v, lo, hi = f(r[f"{key}_delta_pct"]), f(r[f"{key}_pct_ci_lo"]), f(r[f"{key}_pct_ci_hi"])
            c = C_CAT5 if p == "uniform_strong" else C_SIZE
            ax.bar(xi, v, width=0.62, color=c, edgecolor="white", linewidth=1)
            ax.errorbar([xi], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
            pred = f(r["pct_records_elevated"]) / 100 * full
            if p != "uniform_strong":
                ax.plot([xi - 0.36, xi + 0.36], [pred, pred], color=INK, lw=1.1, ls=(0, (2, 1.5)), zorder=4)
            us = f(r[f"{key}_delta_mean_ms"]) * 1000
            ax.text(xi, max(hi, pred) + full * 0.03, f"+{v:.2f}%\n+{us:.1f} µs", ha="center", va="bottom",
                    fontsize=6.5)
        _sweep_ticks(ax, x)
        ax.set_ylim(0, full * 1.28)
        ax.set_ylabel(yl)
        panel_tag(ax, tag)
    handles = [matplotlib.patches.Patch(color=C_SIZE, label="Size-threshold policy (mean, 95% CI)"),
               matplotlib.patches.Patch(color=C_CAT5, label="Uniform Cat. 5"),
               matplotlib.lines.Line2D([], [], color=INK, lw=1.1, ls=(0, (2, 1.5)),
                                       label="Share elevated × uniform Cat. 5 cost")]
    fig.tight_layout(w_pad=2.5)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    save(fig, out, "fig17_size_sweep_latency")


def fig18_size_sweep_storage(T, out):
    sto = by(read_csv(T / "table_policy_storage.csv"), "policy")
    lad = by(read_csv(T / "table_tier_ladder.csv"), "policy")
    b = sto["uniform_baseline"]
    payload = f(b["total_stored_bytes"]) - f(b["overhead_bytes"])
    base_total = f(b["total_stored_bytes"])
    rows = ([("Classical", f(lad["classical"]["overhead_mean_bytes"]) * N_RECORDS, C_CLASSICAL),
             ("Uniform\nCat. 3", f(b["overhead_bytes"]), C_CAT3)]
            + [("", f(sto[f"adaptive_{t}"]["overhead_bytes"]), C_SIZE) for t in SWEEP_UP]
            + [("Uniform\nCat. 5", f(sto["uniform_strong"]["overhead_bytes"]), C_CAT5)])
    fig, (a, bx) = plt.subplots(1, 2, figsize=(DOUBLE, 3.0), gridspec_kw={"width_ratios": [1.45, 1]})
    C_PAYLOAD = "#d9d8d2"
    x = np.arange(len(rows))
    ymax = (payload + rows[-1][1]) * 1.13
    for xi, (lab, over, c) in zip(x, rows):
        a.bar(xi, payload, width=0.66, color=C_PAYLOAD, edgecolor="white", linewidth=1)
        a.bar(xi, over, bottom=payload, width=0.66, color=c, edgecolor="white", linewidth=1)
        tot = payload + over
        txt = f"{tot / 1e6:.1f}" if xi < 2 or xi == len(rows) - 1 else f"+{(tot - base_total) / 1e6:.1f}"
        if xi == len(rows) - 1:
            txt = f"{tot / 1e6:.1f}"
        a.text(xi, tot + ymax * 0.01, txt, ha="center", va="bottom", fontsize=6.5)
    _sweep_ticks(a, x, refs_before=("Classical", "Uniform\nCat. 3"))
    a.set_ylim(0, ymax)
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:,.0f}"))
    a.set_ylabel(f"Total stored, {N_RECORDS:,} records (MB)")
    a.legend(handles=[matplotlib.patches.Patch(color=C_PAYLOAD, label=f"Encrypted payload ({payload / 1e6:.1f} MB)"),
                      matplotlib.patches.Patch(color=C_SIZE, label="Crypto overhead")],
             loc="upper left", fontsize=6.8)
    panel_tag(a, "(a) Total stored bytes (MB; + = added vs Cat. 3)")
    # (b) share of records vs share of plaintext bytes elevated
    xb = np.arange(len(SWEEP_UP))
    rec = [f(sto[f"adaptive_{t}"]["pct_records_elevated"]) for t in SWEEP_UP]
    byt = [f(sto[f"adaptive_{t}"]["pct_plaintext_elevated"]) for t in SWEEP_UP]
    fp = [f(sto[f"adaptive_{t}"]["delta_footprint_pct"]) for t in SWEEP_UP]
    bx.bar(xb - 0.19, rec, width=0.36, color=C_SIZE, label="Records elevated")
    bx.bar(xb + 0.19, byt, width=0.36, color="#eda100", label="Plaintext bytes elevated")
    for xi, r_, b_, d in zip(xb, rec, byt, fp):
        bx.text(xi - 0.19, r_ + 1.2, f"{r_:.0f}", ha="center", va="bottom", fontsize=6.5)
        bx.text(xi + 0.19, b_ + 1.2, f"{b_:.0f}", ha="center", va="bottom", fontsize=6.5)
        bx.text(xi, -13, f"+{d:.1f}%", ha="center", va="top", fontsize=6.5, color=INK2)
    bx.text(-0.75, -13, "Footprint:", ha="right", va="top", fontsize=6.5, color=INK2)
    bx.set_xticks(xb, [f"{SWEEP[t]}%" for t in SWEEP_UP], fontsize=7)
    bx.set_xlabel("Largest share of records elevated", labelpad=14)
    bx.set_ylim(0, 92)
    bx.set_ylabel("Share elevated to Cat. 5 (%)")
    bx.grid(axis="x", visible=False)
    bx.legend(loc="upper left", fontsize=6.8)
    panel_tag(bx, "(b) Records vs bytes elevated")
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "fig18_size_sweep_storage")


def fig19_size_sweep_coverage(T, out):
    cov = by(read_csv(T / "table_policy_coverage.csv"), "policy")
    cats = [("death", "Death reports"), ("special", "Special-category reports"), ("severe", "Severe-outcome reports")]
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.72, 2.9))
    x = np.arange(len(SWEEP_UP))
    w = 0.25
    for k, (rule, lab) in enumerate(cats):
        v = [f(cov[f"adaptive_{t}"][f"pct_{rule}_protected"]) for t in SWEEP_UP]
        ax.bar(x + (k - 1) * w, v, width=w * 0.95, color=REPORT_COLORS[rule], label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 1) * w, vi + 1, f"{vi:.0f}", ha="center", va="bottom", fontsize=6.2)
    share = [f(cov[f"adaptive_{t}"]["pct_records_elevated"]) for t in SWEEP_UP]
    for xi, s in zip(x, share):
        ax.plot([xi - 1.5 * w, xi + 1.5 * w], [s, s], color=INK, lw=1.1, ls=(0, (2, 1.5)), zorder=4)
    ax.plot([], [], color=INK, lw=1.1, ls=(0, (2, 1.5)), label="Records elevated (= random selection)")
    _sweep_ticks(ax, x, with_cat5=False)
    ax.set_ylim(0, 80)
    ax.set_ylabel("Reports protected at Cat. 5 (%)")
    ax.legend(loc="upper left", fontsize=6.8)
    save(fig, out, "fig19_size_sweep_coverage")


def fig20_policy_impact(T, out):
    """Every policy's added end-to-end latency (a) and storage (b) over uniform Cat. 3, by family."""
    lat = by(read_csv(T / "table_policy_latency.csv"), "policy")
    sto = by(read_csv(T / "table_policy_storage.csv"), "policy")
    base_total = f(sto["uniform_baseline"]["total_stored_bytes"])
    flags = sorted(MAIN_FLAGS, key=lambda r: f(lat[f"flag_{r}"]["pct_records_elevated"]))
    groups = [("Size threshold", [(f"adaptive_{t}", f"Largest {SWEEP[t]}% (> {t:,} B)") for t in SWEEP_UP]),
              ("Content flag", [(f"flag_{r}", FLAG_SHORT[r]) for r in flags]),
              ("Reference", [("uniform_strong", "Uniform Cat. 5")])]    # combined (flag or size) left out
    rows, y, yy, heads = [], [], 0.0, []
    for g, items in groups:
        heads.append((g, yy))
        yy += 0.85
        for p, lab in items:
            rows.append((p, lab, g))
            y.append(yy)
            yy += 1
        yy += 0.25
    y = -np.array(y)
    color = {"Size threshold": C_SIZE, "Content flag": C_FLAG, "Flag or size": C_FLAG, "Reference": C_CAT5}
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.6 + 0.165 * (yy + 1)), sharey=True)
    cat5_e2e = f(lat["uniform_strong"]["e2e_delta_pct"])
    cat5_mb = (f(sto["uniform_strong"]["total_stored_bytes"]) - base_total) / 1e6
    for yi, (p, lab, g) in zip(y, rows):
        r, s = lat[p], sto[p]
        c = color[g]
        hatch = "////" if g == "Flag or size" else None
        v, lo, hi = f(r["e2e_delta_pct"]), f(r["e2e_pct_ci_lo"]), f(r["e2e_pct_ci_hi"])
        a.barh(yi, v, height=0.72, color=c, edgecolor="white" if not hatch else c, hatch=hatch,
               fill=hatch is None, linewidth=0.8)
        a.errorbar([v], [yi], xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
        a.text(hi + cat5_e2e * 0.015, yi, f"+{v:.2f}%  ({f(r['e2e_delta_mean_ms']) * 1000:.1f} µs)",
               va="center", fontsize=6.4)
        mb = (f(s["total_stored_bytes"]) - base_total) / 1e6
        b.barh(yi, mb, height=0.72, color=c, edgecolor="white" if not hatch else c, hatch=hatch,
               fill=hatch is None, linewidth=0.8)
        b.text(mb + cat5_mb * 0.015, yi, f"+{mb:.1f} MB  (+{f(s['delta_footprint_pct']):.2f}%)",
               va="center", fontsize=6.4)
    pct = {p: f(lat[p]["pct_records_elevated"]) for p, _, _ in rows}
    a.set_yticks(y, [f"{lab}  [{pct[p]:.1f}%]" for p, lab, _ in rows], fontsize=7)
    for g, hy in heads:
        a.text(-0.012, -hy - 0.1, g, transform=a.get_yaxis_transform(), ha="right", va="center",
               fontsize=7.2, fontweight="bold", color=INK)
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
        ax.set_ylim(y[-1] - 0.7, 0.3)
        ax.axvline(0, color=AXIS, lw=0.8)
    a.set_xlim(0, cat5_e2e * 1.42)
    b.set_xlim(0, cat5_mb * 1.42)
    base_e2e = f(lat["uniform_baseline"]["e2e_mean"])            # uniform Cat. 3 mean round trip (ms)
    # labels name the Cat. 3 baseline each axis is measured against; latency label aligned left, both padded
    a.set_xlabel(f"Added end-to-end latency\nvs uniform Cat. 3 ({base_e2e:.3f} ms)", labelpad=6)
    b.set_xlabel(f"Added storage\nvs uniform Cat. 3 ({base_total / 1e6:.1f} MB)", labelpad=6)
    b.tick_params(axis="y", length=0)
    panel_tag(a, "(a) Latency")
    panel_tag(b, "(b) Storage")
    fig.tight_layout(w_pad=4.0)
    save(fig, out, "fig20_policy_impact")


def table_size_sweep(T, tables):
    lat = by(read_csv(T / "table_policy_latency.csv"), "policy")
    sto = by(read_csv(T / "table_policy_storage.csv"), "policy")
    cov = by(read_csv(T / "table_policy_coverage.csv"), "policy")
    base_total = f(sto["uniform_baseline"]["total_stored_bytes"])
    pols = ["uniform_baseline"] + [f"adaptive_{t}" for t in SWEEP_UP] + ["uniform_strong"]
    rows = []
    for p in pols:
        L, S, C = lat[p], sto[p], cov[p]
        t = None if not p.startswith("adaptive_") else int(p.split("_")[1])
        rows.append({
            "policy": p, "threshold_bytes": t if t else "",
            "label": sweep_label(t, " ") if t else ("Uniform Cat. 3" if p == "uniform_baseline" else "Uniform Cat. 5"),
            "pct_records_elevated": f(L["pct_records_elevated"]), "pct_bytes_elevated": f(L["pct_plaintext_elevated"]),
            "e2e_delta_pct": f(L["e2e_delta_pct"]), "e2e_pct_ci_lo": f(L.get("e2e_pct_ci_lo")),
            "e2e_pct_ci_hi": f(L.get("e2e_pct_ci_hi")), "e2e_delta_us": f(L["e2e_delta_mean_ms"]) * 1000,
            "crypto_delta_pct": f(L["crypto_delta_pct"]), "crypto_pct_ci_lo": f(L.get("crypto_pct_ci_lo")),
            "crypto_pct_ci_hi": f(L.get("crypto_pct_ci_hi")), "crypto_delta_us": f(L["crypto_delta_mean_ms"]) * 1000,
            "footprint_delta_pct": f(S["delta_footprint_pct"]), "total_stored_mb": f(S["total_stored_bytes"]) / 1e6,
            "mb_added_vs_cat3": (f(S["total_stored_bytes"]) - base_total) / 1e6,
            **{f"pct_{r}_protected": f(C[f"pct_{r}_protected"]) for r in ("death", "special", "severe")}})
    keys = list(rows[0])
    with open(tables / "table_size_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    ci = lambda r, k: ("" if np.isnan(r[f"{k}_pct_ci_lo"]) else
                       f" [{r[f'{k}_pct_ci_lo']:+.2f}, {r[f'{k}_pct_ci_hi']:+.2f}]")
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Policy & \% Rec. & \% Bytes & $\Delta$E2E (95\% CI) & $\Delta$Crypto & $\Delta$Footprint & "
             r"Total (MB) & Death & Special \\", r"\midrule"]
    for r in rows:
        lab = r["label"].replace("%", r"\%").replace(">", r"$>$")
        lines.append(f"{lab} & {r['pct_records_elevated']:.1f} & {r['pct_bytes_elevated']:.1f} & "
                     f"{r['e2e_delta_pct']:+.2f}\\%{ci(r, 'e2e')} & {r['crypto_delta_pct']:+.2f}\\% & "
                     f"{r['footprint_delta_pct']:+.2f}\\% & {r['total_stored_mb']:.1f} & "
                     f"{r['pct_death_protected']:.1f}\\% & {r['pct_special_protected']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "table_size_sweep.tex").write_text("\n".join(lines), encoding="utf-8")
    print("  wrote table_size_sweep")


TABLES = ["table_tier_ladder.csv", "table_tier_ladder.tex", "table_policy_latency.csv", "table_policy_latency.tex",
          "table_policy_latency_flags.tex", "table_policy_latency_paper_thresholds.tex", "table_policy_storage.csv",
          "table_policy_coverage.csv", "table_policy_by_quarter.csv", "table_policy_by_quarter.tex",
          "table_policy_by_quarter_flags.tex", "table_stage_breakdown.csv", "summary.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(ROOT / "results" / "analysis_v2"))
    ap.add_argument("--raw", default=str(ROOT / "results" / "paired_v2"))
    ap.add_argument("--manifest", default=str(ROOT / "Data" / "faers_q25k_manifest.csv"))
    ap.add_argument("--out", default=str(ROOT / "analysis_package"))
    ap.add_argument("--only", default="", help="comma-separated figure numbers, e.g. 4,9")
    a = ap.parse_args()
    T, out = Path(a.analysis), Path(a.out)
    figs, tabs = out / "figures", out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    only = {int(x) for x in a.only.split(",") if x}
    want = (lambda n: not only or n in only)

    summary = json.loads((T / "summary.json").read_text())
    print("loading per-record medians and manifest ...")
    pr = load_per_record(T / "e3_per_record.csv.gz")
    manifest = load_manifest(Path(a.manifest))

    if want(1): fig01_tier_ladder(T, figs)
    if want(2): fig02_stage_breakdown(T, figs)
    if want(3): fig03_primitives(summary, figs)
    if want(4): fig04_deployment(summary, pr, figs)
    if want(5): fig05_penalty_vs_size(pr, figs)
    if want(6): fig06_size_vs_death(manifest, figs)
    if want(7): fig07_cost_vs_elevation(T, figs)
    if want(8): fig08_coverage(T, figs)
    if want(9): fig09_decision_cost(summary, T, figs)
    if want(10): fig10_e5_validation(summary, figs)
    if want(11): fig11_quarters(T, pr, manifest, figs)
    if want(13): fig13_latency_by_policy(T, figs)
    if want(14): fig14_storage_by_policy(T, figs)
    if want(15): fig15_total_storage_by_policy(T, figs)
    if want(16):
        fig16_size_distribution(manifest, figs)
        fig16a_size_histogram(manifest, figs)
    if want(17): fig17_size_sweep_latency(T, figs)
    if want(18): fig18_size_sweep_storage(T, figs)
    if want(19): fig19_size_sweep_coverage(T, figs)
    if want(20): fig20_policy_impact(T, figs)
    if not only or only & {17, 18, 19}: table_size_sweep(T, tabs)
    if want(12):
        print("scanning E3 sentinel rows ...")
        fig12_stability(summary, load_sentinels(Path(a.raw) / "e3_main"), figs)
    for t in TABLES:
        if (T / t).exists():
            shutil.copy2(T / t, tabs / t)
    print("done:", out)


if __name__ == "__main__":
    main()
