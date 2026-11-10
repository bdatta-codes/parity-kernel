#!/usr/bin/env python3
"""make_figs.py -- Figures 1 and 3.

Figure 1: the two classes of trinucleotide (A), the words created by a
substitution in a GGG tract (B), and the percentage of verified pairs with
no change in nMSI, by mutagenesis convention (C). Figure 3: concordance of
nMSI and G4Hunter by convention, with the concordance expected for nMSI under
the paired mononucleotide null. Reads results/experiment_538_results.csv and
results/pairs_*.csv (experiment_538.py); writes figures/Figure1.png and
figures/Figure3.png.

License: MIT
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

TEAL, ORANGE, VIOLET = "#0A8F86", "#C05A18", "#7A4FB0"
INK, INK2, RULE = "#14181A", "#4B565B", "#C3CDCF"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": INK2, "axes.linewidth": .8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "savefig.dpi": 400, "figure.dpi": 160,
})

CONV = ["G→C\ncentre", "G→A\ncentre", "G→T\ncentre", "G→T\n2 tracts",
        "tract\nablated", "G→T\n3′ edge", "G→T\n5′ edge", "whole\nscramble"]
# every plotted value is read from the mutagenesis results (src/experiment_538.py)
import os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
KEYS = ["C_centre", "A_centre", "T_centre", "two_tract", "tract_ablate", "T_3prime", "T_5prime", "scramble"]
_E = pd.read_csv(os.path.join(RES, "experiment_538_results.csv"))
def _r(conv, index, col):
    return float(_E[(_E.convention == conv) & (_E["index"] == index)][col].iloc[0])
def _pct_zero(conv):
    d = pd.read_csv(os.path.join(RES, f"pairs_{conv}.csv"))
    return round(100 * float(((d.wt_nMSI - d.mut_nMSI) == 0).mean()), 1)
EQUAL = [_pct_zero(k) for k in KEYS]                                   # % of pairs with delta = 0
CONC  = [round(100 * _r(k, "nMSI", "obs"), 1) for k in KEYS]          # ties counted as one half
NULL  = [100 * _r(k, "nMSI", "null") for k in KEYS]                    # concordance expected under the paired mononucleotide null
KIND   = ["centre","centre","centre","centre","other","edge","edge","other"]

# ============================================================ FIGURE 1 ======
fig = plt.figure(figsize=(7.2, 5.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, .92], width_ratios=[1, 1.32],
                      hspace=.30, wspace=.26,
                      left=.075, right=.985, top=.93, bottom=.10)

# --- A: the two classes of trinucleotide -----------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.set_title("A   The two classes of trinucleotide", loc="left", fontweight="bold")
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.add_patch(FancyBboxPatch((.3, 5.6), 9.2, 3.5, boxstyle="round,pad=.12",
                            fc="#EAF3F2", ec=TEAL, lw=1.1))
ax.text(.7, 8.35, "48 words in 24 reverse pairs", fontsize=7.8, fontweight="bold", color=TEAL)
ax.text(.7, 7.35, "GGA  ↔  AGG      TGG  ↔  GGT", fontsize=9, family="monospace")
ax.text(.7, 6.25, "the difference enters the numerator", fontsize=7.4, color=INK2, style="italic")
ax.add_patch(FancyBboxPatch((.3, 1.0), 9.2, 3.5, boxstyle="round,pad=.12",
                            fc="#F5E8EA", ec="#9E3245", lw=1.1))
ax.text(.7, 3.75, "16 words that are their own reverse", fontsize=7.4, fontweight="bold", color="#9E3245")
ax.text(.7, 2.75, "GGG   GTG   GAG   GCG", fontsize=9, family="monospace")
ax.text(.7, 1.65, "excluded from both sums of nMSI", fontsize=7.4, color=INK2, style="italic")

# --- B: the mutation identity ----------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.set_title("B   What a G-tract substitution creates", loc="left", fontweight="bold")
ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 10)
ax.text(.2, 9.1, "position", fontsize=7, color=INK2, fontweight="bold")
ax.text(3.0, 9.1, "GGG becomes", fontsize=7, color=INK2, fontweight="bold")
ax.text(6.6, 9.1, "its reverse", fontsize=7, color=INK2, fontweight="bold")
ax.text(9.4, 9.1, "scored?", fontsize=7, color=INK2, fontweight="bold")
rows = [("centre  G→T", "GTG", "GTG", False), ("centre  G→A", "GAG", "GAG", False),
        ("centre  G→C", "GCG", "GCG", False), ("5′ edge  G→T", "TGG", "GGT", True),
        ("3′ edge  G→T", "GGT", "TGG", True), ("3′ edge  G→A", "GGA", "AGG", True)]
for i, (lab, w, r, seen) in enumerate(rows):
    y = 8.0 - i*1.15
    c = TEAL if seen else "#9E3245"
    ax.add_patch(Rectangle((.05, y-.42), 11.9, 1.06, fc="#F2F6F6" if i % 2 == 0 else "white",
                           ec="none", zorder=0))
    ax.text(.2, y, lab, fontsize=7.6)
    ax.text(3.0, y, w, fontsize=9, family="monospace", color=c, fontweight="bold")
    ax.text(6.6, y, r, fontsize=9, family="monospace", color=c)
    ax.text(9.4, y, "yes" if seen else "no", fontsize=7.6, color=c, fontweight="bold")
ax.text(.2, -.4, "Reversal exchanges the first and third bases of a trinucleotide\n"
                 "and leaves the middle base in place. Words spanning the tract\n"
                 "boundaries, formed with the flanking bases, are not shown.",
        fontsize=7.0, color=INK2, style="italic")

# --- C: fraction unchanged --------------------------------------------------
ax = fig.add_subplot(gs[1, :])
ax.set_title("C   Verified pairs with ΔnMSI = 0, by mutagenesis convention",
             loc="left", fontweight="bold")
x = np.arange(len(CONV))
CMAP = {"centre":"#9E3245","edge":TEAL,"other":"#9AA6AA"}
cols = [CMAP[k] for k in KIND]
b = ax.bar(x, EQUAL, color=cols, width=.62, zorder=3)
ax.bar(x, [0.0]*len(x), color="none")
for xi, v in zip(x, EQUAL):
    ax.text(xi, v+.7, f"{v:.1f}%", ha="center", fontsize=7.4, fontweight="bold")
ax.axhline(0, color=INK2, lw=.8)
ax.set_xticks(x); ax.set_xticklabels(CONV, fontsize=7.2)
ax.set_ylabel("% of pairs with ΔnMSI = 0")
ax.set_ylim(0, 38)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E4EAEA", lw=.7, zorder=0)
ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#9E3245", label="creates a self-reverse trinucleotide (not scored)"),
                   Patch(fc=TEAL,      label="creates one half of a reverse pair (scored)"),
                   Patch(fc="#9AA6AA", label="other design")],
          frameon=False, fontsize=7, loc="upper right", handlelength=1.1, handleheight=.9)
fig.savefig(os.path.join(FIG, "Figure1.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================ FIGURE 3 ======
fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.3))
order = np.argsort(CONC)
labels = [CONV[i].replace("\n", " ") + "" for i in order]
y = np.arange(len(order))
for yi, i in zip(y, order):
    ax.plot([CONC[i], 100], [yi, yi], color=RULE, lw=1.3, zorder=1)
ax.scatter([CONC[i] for i in order], y, s=52, color=TEAL, zorder=3, label="nMSI")
ax.scatter([100]*len(y), y, s=52, color=ORANGE, zorder=3, label="G4Hunter")
ax.scatter([NULL[i] for i in order], y, s=60, marker="|", color=INK2, linewidths=1.6, zorder=4,
           label="nMSI, paired mononucleotide null")
for yi, i in zip(y, order):
    ax.text(CONC[i]-2.2, yi, f"{CONC[i]:.1f}", ha="right", va="center", fontsize=7, zorder=5,
            bbox=dict(fc="white", ec="none", pad=.6))
ax.axvline(50, color="#9E3245", ls=(0, (4, 3)), lw=1.1, zorder=2)
ax.text(50.9, -.62, "50% (fair coin)", fontsize=6.8, color="#9E3245")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.4)
ax.set_xlim(34, 104); ax.set_xlabel("concordance: % of pairs in which the wild type outscores\nits abolished mutant (ties counted as one half)")
ax.set_title("Concordance by mutagenesis convention", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#E4EAEA", lw=.7); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.40, -0.24), ncol=2, handletextpad=.3)

fig.savefig(os.path.join(FIG, "Figure3.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote figures/Figure1.png and figures/Figure3.png")
