#!/usr/bin/env python3
"""si_figs.py -- Supplementary Figures S1-S3.

S1: nMSI over 20,000 shuffles of each promoter quadruplex, with the observed
value and the independent-base mean (results/supplementary_tables/
TableS12_promoter_panel.csv, si_compute.py). S2: change in nMSI over the
verified pairs, by convention (results/pairs_*.csv). S3: the single-
substitution scan by position and substituted base (results/
mutagenesis_scan.csv). Writes figures/FigureS1-S3.png.

License: MIT
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")

TEAL, ORANGE, VIOLET = "#0A8F86", "#C05A18", "#7A4FB0"
CRIM, GREY = "#9E3245", "#9AA6AA"
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
os.makedirs(FIG, exist_ok=True)

# ------------------------------- Figure S1: promoter null panels ---
nulls = json.load(open(os.path.join(RES, "si", "promoter_nulls.json")))
T = pd.read_csv(os.path.join(RES, "supplementary_tables", "TableS12_promoter_panel.csv"))
fig, axes = plt.subplots(3, 5, figsize=(9.6, 5.4))
for ax, (_, r) in zip(axes.ravel(), T.iterrows()):
    v = np.array(nulls[r.gene])
    ax.hist(v, bins=np.linspace(0, 1, 41), color=TEAL, alpha=.75, edgecolor="none")
    ax.axvline(r.nMSI, color=CRIM, lw=1.6)
    ax.axvline(r.iid_mean, color=INK2, lw=1.2, ls="--")
    ax.set_title(f"{r.gene}   $L$ = {r.L}", fontsize=8)
    ax.set_yticks([]); ax.set_xlim(0, 1.03)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.5)
    if r.pct_ceiling > 5:
        ax.text(.02, .92, f"{r.pct_ceiling:.0f}% at 1.000", transform=ax.transAxes,
                fontsize=6.5, color=CRIM, fontweight="bold", va="top")
lg = axes.ravel()[-1]; lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
lg.plot([0, .18], [.86, .86], color=CRIM, lw=1.6)
lg.text(.24, .86, "observed nMSI", color=CRIM, fontsize=8, fontweight="bold", va="center")
lg.plot([0, .18], [.60, .60], color=INK2, lw=1.2, ls="--")
lg.text(.24, .60, "independent bases,\nsame composition", color=INK2, fontsize=8, fontweight="bold", va="center")
lg.add_patch(plt.Rectangle((0, .27), .18, .1, color=TEAL, alpha=.75, lw=0))
lg.text(.24, .32, "20,000 composition-\nmatched shuffles", color=TEAL, fontsize=8, fontweight="bold", va="center",
        linespacing=1.4)
fig.suptitle("", y=1)
fig.supxlabel("nMSI", fontsize=8.5, y=0.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "FigureS1.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ------------------------------------- S2: Delta nMSI by convention --------
FILES = [("pairs_C_centre.csv", "G→C, centre"), ("pairs_A_centre.csv", "G→A, centre"),
         ("pairs_T_centre.csv", "G→T, centre"), ("pairs_two_tract.csv", "G→T, two tracts"),
         ("pairs_tract_ablate.csv", "tract ablated"), ("pairs_T_3prime.csv", "G→T, 3′ edge"),
         ("pairs_T_5prime.csv", "G→T, 5′ edge"), ("pairs_scramble.csv", "whole scramble")]
fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.4), sharex=True)
for ax, (f, lab) in zip(axes.ravel(), FILES):
    d = pd.read_csv(os.path.join(RES, f))
    dn = (d.wt_nMSI - d.mut_nMSI).values
    z0 = np.isclose(dn, 0)                       # ties, as counted in the text
    zero = 100 * float(z0.mean())
    ax.hist(dn[~z0], bins=np.linspace(-.6, .8, 57), color=TEAL, alpha=.8, edgecolor="none")   # non-zero changes only
    n0 = int(z0.sum())
    ax.bar([0], [n0], width=.025, color=CRIM)
    ax.axvline(0, color=INK2, lw=.8)
    ax.set_title(f"{lab}\n{zero:.1f}% exactly zero", fontsize=8, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
axes[1, 0].set_xlabel("nMSI(wild type) − nMSI(abolished mutant)", fontsize=8)
axes[0, 0].set_ylabel("pairs"); axes[1, 0].set_ylabel("pairs")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "FigureS2.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ------------------------------------------- S3: single-substitution scan --
S = pd.read_csv(os.path.join(RES, "mutagenesis_scan.csv"))   # delta = mutant - wild type
order = ["5' edge", "interior", "3' edge"]
lab = {"5' edge": "5′ edge", "interior": "tract interior", "3' edge": "3′ edge"}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1, 1.25]})
x = np.arange(3); w = .26
for i, (cat, col) in enumerate([("raise", TEAL), ("zero", GREY), ("lower", ORANGE)]):
    vals = []
    for p in order:
        d = S[S.position == p].delta
        vals.append(100 * float((d > 0).mean() if cat == "raise" else
                                (d == 0).mean() if cat == "zero" else (d < 0).mean()))
    ax.bar(x + (i - 1) * w, vals, w, color=col,
           label={"raise": "raises nMSI", "zero": "no change", "lower": "lowers nMSI"}[cat])
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([lab[p] for p in order])
ax.set_ylabel("% of substitutions"); ax.set_ylim(0, 72)
ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper center")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("A   By position within the guanine tract", loc="left", fontsize=9, fontweight="bold")

piv = S.pivot_table(index="position", columns="base", values="delta", aggfunc="mean").loc[order]
bases = list(piv.columns)
for i, b in enumerate(bases):
    ax2.bar(x + (i - (len(bases) - 1) / 2) * .26, piv[b].values, .26,
            color=[TEAL, ORANGE, VIOLET][i % 3], label=f"G→{b}")
ax2.axhline(0, color=INK2, lw=.8)
ax2.set_xticks(x); ax2.set_xticklabels([lab[p] for p in order])
ax2.set_ylabel("mean change in nMSI (mutant − wild type)")
ax2.legend(frameon=False, fontsize=7.5, ncol=3)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_title("B   By substituted base", loc="left", fontsize=9, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "FigureS3.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote SI figures")
