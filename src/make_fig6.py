#!/usr/bin/env python3
"""make_fig6.py — Figure 6.

Panels A-C are drawn from the tables written by chromosome21_windows.py
(results/chr21/table4_windows_by_length.csv for Table 4 and panel A,
fig6b_deficit_cpg.csv for panel B and table5_g4_symmetry.csv for Table 5 and
panel C).

    python3 src/make_fig6.py --chr21-results results/chr21 --out figures/Figure6.png

License: MIT
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG, GREEN, ORANGE, BLUE = "#FCFCFB", "#1BAF7A", "#EB6834", "#2A78D6"
GREY, DARK, LIGHT = "#8A8A85", "#4B4B48", "#E3E3E0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#7A7A76", "axes.linewidth": 1.1,
    "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "axes.labelsize": 12, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "savefig.dpi": 200,
})


def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=1.0); ax.set_axisbelow(True)


def title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=13.5, fontweight="bold", pad=12)


def pfmt(p):
    if p < 1e-15:
        return "p < 10⁻¹⁵ (descriptive)"
    if p < 0.001:
        e = int(np.floor(np.log10(p))); return f"p = {p / 10**e:.1f} × 10{str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"
    return f"p = {p:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chr21-results", default="results/chr21")
    ap.add_argument("--out", default="figures/Figure6.png")
    a = ap.parse_args()
    t4 = pd.read_csv(os.path.join(a.chr21_results, "table4_windows_by_length.csv"))
    b = pd.read_csv(os.path.join(a.chr21_results, "fig6b_deficit_cpg.csv")).iloc[0]
    t5 = pd.read_csv(os.path.join(a.chr21_results, "table5_g4_symmetry.csv")).iloc[0]

    fig = plt.figure(figsize=(24.0, 6.4))
    gs = fig.add_gridspec(1, 3, wspace=0.30)

    # A  observed vs predicted on chr21 (Table 4)
    ax = fig.add_subplot(gs[0, 0])
    L = t4.window_nt.tolist(); obs = t4.observed_nmsi.tolist()
    pred = t4.predicted_baseline.tolist(); z = t4.mean_z.tolist()
    xl = np.log10(L)
    ax.fill_between(xl, obs, pred, color=ORANGE, alpha=.15, lw=0)
    ax.plot(xl, pred, "-s", color=BLUE, lw=3, ms=10, mfc="white", mew=2.5)
    ax.plot(xl, obs, "-o", color=ORANGE, lw=3, ms=10, mfc="white", mew=2.5)
    for i in (2, len(L) - 1):
        ax.text(xl[i], (obs[i] + pred[i]) / 2, f"z = {z[i]:.1f}", ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=.25", fc="white", ec=GREY))
    ax.text(xl[-1] + .07, pred[-1], "analytic baseline\n(independent bases)", color=BLUE,
            fontweight="bold", va="center", fontsize=10.5)
    ax.text(xl[-1] + .07, obs[-1] - .01, "real human\nchromosome 21", color=ORANGE,
            fontweight="bold", va="center", fontsize=10.5)
    ax.set_xticks(xl); ax.set_xticklabels([str(v) for v in L]); ax.set_xlim(xl[0] - .1, xl[-1] + .75)
    ax.set_xlabel("window length (nt)"); ax.set_ylabel("nMSI")
    style(ax); title(ax, "A", "Observed and predicted nMSI by window length")

    # B  correlates of the deficit
    ax = fig.add_subplot(gs[0, 1])
    lab = ["GC content", "CpG under-\nrepresentation", "CpG under-\nrepresentation,\nholding GC constant"]
    val = [b.deficit_vs_gc_rho, b.deficit_vs_cpgunder_rho, b.deficit_vs_cpgunder_partial_gc_rho]
    col = [GREY, "#F2A07F", ORANGE]
    ax.bar(range(3), val, .6, color=col)
    for i, v in enumerate(val):
        ax.text(i, v + .012, f"{v:+.3f}", ha="center", fontsize=11.5, fontweight="bold", color=DARK)
    ax.text(2, val[2] + .065, pfmt(b.deficit_vs_cpgunder_partial_gc_p), ha="center",
            fontsize=10.5, style="italic", color=DARK)
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=10.5)
    ax.set_ylim(0, max(.56, max(val) + .12)); ax.set_ylabel("Spearman ρ with the symmetry deficit")
    ax.text(.98, .97, f"n = {int(b.n):,} clean 300-nt windows of chromosome 21",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, style="italic", color=DARK)
    style(ax); title(ax, "B", "Correlates of the symmetry deficit")

    # C  G4 content and symmetry on chr21 (Table 5)
    ax = fig.add_subplot(gs[0, 2])
    lab = ["G4 density\nvs symmetry", "GC content\nvs symmetry", "G4 density\nvs GC",
           "G4 vs symmetry,\nholding GC"]
    val = [t5.g4_vs_symmetry_rho, t5.gc_vs_symmetry_rho, t5.g4_vs_gc_rho,
           t5.g4_vs_symmetry_partial_gc_rho]
    col = [BLUE, GREY, GREY, GREEN]
    ax.bar(range(4), val, .6, color=col); ax.axhline(0, color=DARK, lw=1.4)
    for i, v in enumerate(val):
        ax.text(i, v + (.03 if v >= 0 else -.03), f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=11.5, fontweight="bold", color=DARK)
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=10.5)
    ax.set_ylim(-.55, .98); ax.set_ylabel("Spearman ρ")
    ax.text(.02, .97, f"n = {int(t5.n):,} clean 300-nt windows of chromosome 21",
            transform=ax.transAxes, va="top", fontsize=10, style="italic", color=DARK)
    style(ax); title(ax, "C", "Quadruplex density, GC content and symmetry")

    fig.savefig(a.out, bbox_inches="tight")


if __name__ == "__main__":
    main()
