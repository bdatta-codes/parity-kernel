#!/usr/bin/env python3
"""make_fig2.py — Figure 2 (the closed-form baseline) from the CSVs written by
src/closed_form.py (run that first). Lines are closed forms computed here;
open marks are the Monte Carlo estimates stored in results/closed_form/.

  A  closed form (words counted without overlap) and closed form + kappa/sqrt(N)
     (sliding window) against N, with Monte Carlo points for both.
  B  sliding-window baseline (closed form at N = L-k+1 plus the fitted kappa of
     Supplementary Table S7) against sequence length for k = 3, 4, 5; the length at which each
     reaches 0.50 is marked (results/closed_form/min_length.csv, column
     sliding_closed_plus_kappa).
  C  z of a fixed nMSI = 0.50 against the length-matched baseline (Table 3:
     mean = closed form + 0.615/sqrt(N), SD simulated).
  D  expected index at 300 nt against strand skew (GC 0.5), mirror and
     reverse-complement pairings, with the S1 large-N ceiling.

License: MIT
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_form as cf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "closed_form")
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


def rd(name):
    return pd.read_csv(os.path.join(RES, name))


uni, nov = rd("kappa_uniform_30_3000.csv"), rd("nonoverlap_uniform.csv")
T3, ML, T4, D = rd("tableS7_word_length.csv"), rd("min_length.csv"), rd("table3_fixed_criterion.csv"), rd("fig2D_skew_300nt.csv")
kap_uni = float(uni.kappa_L.median())   # Eq. 4: median of the per-length values (0.615)

fig = plt.figure(figsize=(14.7, 9.9))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28, left=.06, right=.985, top=.94, bottom=.07)
mk = dict(mfc="white", mew=2.2, ms=9, ls="none", zorder=4)

# ---------------------------------------------------------------- A -------
ax = fig.add_subplot(gs[0, 0]); style(ax)
N = np.unique(np.round(np.geomspace(8, 4500, 90)).astype(int))
c = cf.closed_form(N)
ax.plot(N, c, color=GREEN, lw=3)
Ns = N[N >= 28]                         # the fit range of kappa starts at 30 nt (N = 28)
ax.plot(Ns, cf.closed_form(Ns) + kap_uni / np.sqrt(Ns), color=BLUE, lw=3)
ax.plot(nov.N, nov.sim_mean, "o", color=GREEN, **mk)
ax.plot(uni.N, uni.sim_mean, "s", color=BLUE, **mk)
ax.set_xscale("log"); ax.set_xlim(8, 4500); ax.set_ylim(0.05, 1.0)
ax.set_xlabel("N, number of trinucleotides counted"); ax.set_ylabel("expected nMSI under the null")
ax.text(9, 0.56, f"sliding window\n(+ κ/√N, κ = {kap_uni:.3f})", color=BLUE, fontweight="bold", fontsize=12)
ax.text(160, 0.30, "counted without overlap\n(closed form)", color=GREEN, fontweight="bold", fontsize=12)
ax.text(0.98, 0.06, "open marks = Monte Carlo\nlines = closed form", transform=ax.transAxes,
        ha="right", va="bottom", style="italic", color=DARK)
title(ax, "A", "Expected nMSI against number of words counted")

# ---------------------------------------------------------------- B -------
ax = fig.add_subplot(gs[0, 1]); style(ax, grid="both")
L = np.unique(np.round(np.geomspace(13, 40000, 130)).astype(int))
ax.axvspan(13, 85, color=ORANGE, alpha=.12, lw=0)
ax.axhspan(0, 0.5, color=GREY, alpha=.10, lw=0)
ax.axhline(0.5, color=DARK, ls="--", lw=1.6)
ax.text(36000, 0.53, "baseline = 0.50", ha="right", style="italic", color=DARK)
ax.text(14, 0.93, "G4 motif\nrange", ha="left", color=ORANGE, fontweight="bold")
P3 = rd("tableS7_kappa_per_length.csv"); SH = rd("fig2B_short_sim.csv")
for k, col in ((3, BLUE), (4, ORANGE), (5, GREEN)):
    kap = float(T3[(T3.k == k) & (T3.pairing == "mirror")].kappa.iloc[0])
    Lk = L[L >= 60]                       # closed form + kappa: the 60-6,000 nt fit range and beyond
    y = cf.closed_form(Lk - k + 1, k) + kap / np.sqrt(Lk - k + 1)
    ax.plot(Lk, y, color=col, lw=3, label=f"k = {k}  (κ = {kap:.2f})")
    if k == 3:                            # below the fit range, to reach the 51-nt crossing: extrapolated (dashed)
        Ls = L[(L <= 60) & (L >= 30)]
        ax.plot(Ls, cf.closed_form(Ls - k + 1, k) + kap / np.sqrt(Ls - k + 1), color=col, lw=1.6, ls="--")
    sim = pd.concat([SH[SH.k == k][["L", "sim_mean"]],
                     P3[(P3.k == k) & (P3.pairing == "mirror")][["L", "sim_mean"]]])
    ax.plot(sim.L, sim.sim_mean, "s", color=col, mfc="white", mew=1.6, ms=6, ls="none", zorder=3)
    Lm = int(ML[ML.k == k].sliding_closed_plus_kappa.iloc[0])
    ax.plot([Lm], [0.5], "o", color=col, **mk)
    ax.text(Lm * 1.08, 0.43, f"{Lm:,} nt", color=col, fontweight="bold")
ax.text(0.98, 0.02, "lines = closed form + κ/√N (k = 3 dashed below the 60-nt fit range)\nsquares = sliding-window Monte Carlo; grey = baseline < 0.50",
        transform=ax.transAxes, ha="right", style="italic", color=DARK, fontsize=10)
ax.set_xscale("log"); ax.set_xlim(13, 40000); ax.set_ylim(0, 1.05)
ax.set_xlabel("sequence length (nt)"); ax.set_ylabel("sliding-window baseline of the parity index")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.09), frameon=False, fontsize=11)
title(ax, "B", "Sliding-window baseline by word length")

# ---------------------------------------------------------------- C -------
ax = fig.add_subplot(gs[1, 0]); style(ax)
ax.axvspan(18, 49, color=GREEN, alpha=.12, lw=0)
ax.axhline(0, color=DARK, ls="--", lw=1.4)
ax.axhline(1.645, color=GREEN, ls=":", lw=2)
ax.text(480, 1.85, "z = 1.645  (p = 0.05)", ha="right", color=GREEN)
ax.plot(T4.L, T4.z_sim_sd, "-o", color=ORANGE, lw=3, mfc="white", mew=2.2, ms=9, label="closed form + κ/√N (Table 3)")
ax.plot(T4.L, T4.z_fully_simulated, "s", color=GREY, ms=5, ls="none", zorder=5, label="fully simulated mean")
ax.legend(loc="lower left", frameon=False, fontsize=10)
ax.annotate("experimentally labelled\nsequences of ref. 21\n(18–49 nt)", xy=(32, 0.9), xytext=(16, 2.6),
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=DARK, lw=1.3))
ax.set_xscale("log"); ax.set_xlim(14, 550); ax.set_ylim(-10.5, 4.5)
ax.set_xticks([15, 20, 30, 50, 100, 200, 500]); ax.set_xticklabels(["15", "20", "30", "50", "100", "200", "500"])
ax.minorticks_off()
ax.set_xlabel("sequence length (nt)"); ax.set_ylabel("z of a fixed nMSI = 0.50")
title(ax, "C", "z of the criterion nMSI ≥ 0.50 against length")

# ---------------------------------------------------------------- D -------
ax = fig.add_subplot(gs[1, 1]); style(ax)
ax.fill_between(D["skew"], D.rc_closed, D.mirror_closed, color=ORANGE, alpha=.10, lw=0)
ax.plot(D["skew"], D.mirror_closed, color=BLUE, lw=3)
ax.plot(D["skew"], D.rc_closed, color=ORANGE, lw=3)
ax.plot(np.r_[0, D["skew"]], np.r_[1, D.rc_ceiling], color=ORANGE, ls="--", lw=1.6)
m = D.dropna(subset=["mirror_sim"])
ax.plot(m["skew"], m.mirror_sim, "s", color=BLUE, **mk)
ax.plot(m["skew"], m.rc_sim, "o", color=ORANGE, **mk)
ax.text(0.42, 0.84, "mirror pairing (nMSI)", color=BLUE, fontweight="bold", fontsize=12)
ax.annotate("its ceiling at large N", xy=(0.12, 0.885), xytext=(0.15, 0.97), color=ORANGE, style="italic",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
ax.annotate("reverse complement (S₁)", xy=(0.6, 0.335), xytext=(0.26, 0.27), color=ORANGE, fontweight="bold",
            fontsize=12, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
ax.set_xlim(0, 0.88); ax.set_ylim(0.15, 1.04)
ax.set_xlabel("strand (G versus C) skew, GC = 0.5"); ax.set_ylabel("expected index at 300 nt")
title(ax, "D", "Expected index against strand skew")

out = os.path.join(ROOT, "figures", "Figure2.png")
fig.savefig(out)
print("wrote", out)
