#!/usr/bin/env python3
"""make_fig4.py -- draw Figure 4 from the CSVs written by g4_runlength_analysis.py.

  A  results/g4/fig4a_runlength_distribution.csv
  B  results/g4/fig4b_classifier_auc.csv          (dinucleotide shuffles)
  C  results/a549_full/per_sequence.csv           (the 23,040 A549 motifs with defined nMSI; a549_full_set.py)

Usage:  python3 src/make_fig4.py [--out figures/Figure4.png]
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "g4")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "Figure4.png"))
    args = ap.parse_args()
    A = pd.read_csv(os.path.join(RES, "fig4a_runlength_distribution.csv"))
    B = pd.read_csv(os.path.join(RES, "fig4b_classifier_auc.csv"))
    C = pd.read_csv(os.path.join(ROOT, "results", "a549_full", "per_sequence.csv"))
    C["d_nMSI"] = C.nMSI - C.mono_mean_nMSI_K100
    C = C[np.isfinite(C.d_nMSI) & np.isfinite(C.d_runs)]

    fig = plt.figure(figsize=(15.8, 11.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.62)

    # A  run-length distribution, 98 consensus-negative sequences vs mono shuffles
    ax = fig.add_subplot(gs[0, 0])
    real, sh = A.real_pct.to_numpy(), A.mono_pct.to_numpy()
    x = np.arange(1, 9); w = 0.38
    ax.bar(x - w / 2, real, w, color=GREEN, label="real")
    ax.bar(x + w / 2, sh, w, color=GREY, alpha=.8, label="matched shuffle")
    ax.set_xticks(x); ax.set_xticklabels(A.run_length.astype(str))
    ax.set_xlabel("length of G-run (nt)"); ax.set_ylabel("% of all G-runs")
    ax.set_ylim(0, 58)
    ax.legend(frameon=False, fontsize=11, loc="upper right", bbox_to_anchor=(1.0, 0.98))
    style(ax); title(ax, "A", "Guanine run-length distribution")

    # B  cross-validated AUC against dinucleotide-preserving shuffles
    ax = fig.add_subplot(gs[0, 1])
    d = B[B.shuffle == "dinuc"].set_index("feature_set")
    rows = [("G-run-length profile", "G-run-length profile", GREEN),
            ("G4Hunter-derived,\nmatched feature count", "G4Hunter-derived (matched feature count)", "#7FD3B3"),
            ("G4Hunter, published\ntwo-feature form", "G4Hunter published two-feature form", GREY),
            ("parity indices\n(nMSI, MSI, S₁)", "parity indices (nMSI, MSI, S1)", ORANGE)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, key, c) in zip(y, rows):
        r = d.loc[key]; v = r.auc_mean_over_repeats
        ax.barh(yi, v - .4, left=.4, height=.62, color=c)
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=DARK, lw=1.4)
        ax.text(max(v, r.ci_hi) + .008, yi, f"{v:.3f}", va="center", fontsize=11.5, fontweight="bold")
    singles = ", ".join(f"{k.split()[0]} {d.loc[k].auc_mean_over_repeats:.2f}"
                        for k in ["nMSI alone", "MSI alone", "S1 alone"])
    ax.text(.70, y[-1] - .02, f"singly: {singles}\nmax upper bound "
            f"{max(d.loc[k].ci_hi for k in ['nMSI alone', 'MSI alone', 'S1 alone']):.2f}",
            fontsize=9.5, color=DARK, style="italic", va="center")
    ax.axvline(.5, color=DARK, ls="--", lw=1.4)
    ax.text(.505, len(rows) - .45, "chance", fontsize=10, style="italic", color=DARK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlim(.4, .9); ax.set_xlabel("cross-validated AUC against dinucleotide-preserving shuffles")
    ax.tick_params(axis="y", length=0)
    style(ax, "x"); title(ax, "B", "Separation from dinucleotide shuffles, by feature set")

    # C  delta nMSI vs delta number of G-runs, all A549 motifs
    sub = gs[1, :].subgridspec(1, 4)
    ax = fig.add_subplot(sub[0, 1:3])
    dn, dr = C.d_nMSI.to_numpy(), C.d_runs.to_numpy()
    rho, p = spearmanr(dn, dr)
    ax.axhline(0, color=DARK, lw=1); ax.axvline(0, color=DARK, lw=1)
    ax.scatter(dr, dn, s=5, color=BLUE, alpha=.18, edgecolor="none", rasterized=True)
    b = np.polyfit(dr, dn, 1); xx = np.linspace(dr.min(), dr.max(), 50)
    ax.plot(xx, np.polyval(b, xx), color=ORANGE, lw=2.4)
    ax.text(.03, .95, f"Spearman ρ = {rho:+.2f}\nn = {len(dn):,} sequences",
            transform=ax.transAxes, va="top", fontsize=11.5, fontweight="bold", color=DARK,
            bbox=dict(facecolor=BG, edgecolor="none", alpha=.85, pad=2))
    ax.set_xlabel("Δ number of G-runs  (real − mean of its shuffles)")
    ax.set_ylabel("Δ nMSI  (real − mean of its shuffles)")
    style(ax, "both"); title(ax, "C", "Change in nMSI against change in G-run number")

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}  (C: rho = {rho:+.3f}, p = {p:.2g}, mean d runs = {dr.mean():+.3f})")


if __name__ == "__main__":
    main()
