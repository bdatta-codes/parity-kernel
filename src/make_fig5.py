#!/usr/bin/env python3
"""
make_fig5.py — Figure 5, both panels from CSV.

Panel A is drawn by make_fig5a.render_panel_a() from
results/cds/fig5a_auc_by_orf_length_default.csv. Panel B is drawn from results/cds/fig5b_codon_structure.csv
(coding_sequence_analysis.py / coding_extras.fig5b_codon_structure): the mean
frame-0 index of the real coding sequences, after whole-codon permutation,
after shuffling bases within each codon position and after a mononucleotide
shuffle. The depression is measured from the mononucleotide-shuffled value;
the part recovered by the within-position shuffle is due to codon identity
beyond per-position composition, and the part that remains is carried by
per-position composition (what Fickett's statistic reads).

    python3 src/make_fig5.py --fig5a results/cds/fig5a_auc_by_orf_length_default.csv \\
        --fig5b results/cds/fig5b_codon_structure.csv --out figures/Figure5.png

License: MIT
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_fig5a import render_panel_a, BG, INK, GRID, MUTED  # noqa: E402

ORANGE, BLUE, GREY = "#EB6834", "#2A78D6", "#8A8A85"
W_B, H = 1393, 1214          # panel-B size in px at 300 dpi


def render_panel_b(csv, tmp):
    r = pd.read_csv(csv).iloc[0]
    real, perm = r.mean_real, r.mean_codon_permuted
    pos, mono = r.mean_within_position_shuffled, r.mean_mononucleotide_shuffled

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(W_B / 300, H / 300), dpi=300)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    vals = [real, perm, pos, mono]
    cols = [ORANGE, ORANGE, BLUE, GREY]
    x = [0, 1, 2, 3]
    ax.bar(x, vals, 0.62, color=cols, zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#4B4B48")
    # total depression: real -> mononucleotide-shuffled
    xa = -0.52
    ax.annotate("", xy=(xa, real), xytext=(xa, mono),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text(xa + 0.08, mono + 0.035, "mononucleotide-shuffled − real",
            fontsize=8, fontweight="bold", va="center", ha="left", color=INK)
    # split at the position-shuffled bar
    xb = 2.42
    ax.plot([2.31, 3.31], [mono, mono], color=INK, lw=0.8, ls=(0, (2, 2)))
    ax.plot([0.31, 2.31], [real, real], color=INK, lw=0.8, ls=(0, (2, 2)))
    ax.annotate("", xy=(xb, real), xytext=(xb, pos),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2))
    ax.annotate("", xy=(xb, pos), xytext=(xb, mono),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
    ax.text(1.62, 0.715,
            "mononucleotide-shuffled\n− position-shuffled",
            ha="right", va="center", fontsize=7.5, color="#52514E", fontweight="bold")
    ax.text(1.62, 0.615,
            "position-shuffled\n− real",
            ha="right", va="center", fontsize=7.5, color=BLUE, fontweight="bold")
    fig.text(0.115, 0.905, f"n = {int(r.n_sequences):,} coding sequences, "
             f"{int(r.n_shuffles_per_sequence)} shuffles each",
             ha="left", va="center", fontsize=7, style="italic", color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(["real\nCDS", "codon\nshuffled", "position\nshuffled",
                        "mononucleotide\nshuffled"])
    ax.set_xlim(-0.75, 3.45)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("frame-0 symmetry index", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    fig.text(0.115, 0.958, "B   Frame-0 index of CDS and its shuffles",
             fontweight="bold", fontsize=10.5, color=INK, va="center")
    fig.subplots_adjust(left=0.115, right=0.985, top=0.865, bottom=0.135)
    fig.savefig(tmp, dpi=300, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig5a", default="results/cds/fig5a_auc_by_orf_length_default.csv")
    ap.add_argument("--fig5b", default="results/cds/fig5b_codon_structure.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from PIL import Image
    ta, tb = a.out + ".panelA.png", a.out + ".panelB.png"
    render_panel_a(a.fig5a, ta)
    render_panel_b(a.fig5b, tb)
    pa, pb = Image.open(ta).convert("RGB"), Image.open(tb).convert("RGB")
    out = Image.new("RGB", (pa.width + pb.width, max(pa.height, pb.height)), BG)
    out.paste(pa, (0, 0)); out.paste(pb, (pa.width, 0))
    out.save(a.out, dpi=(300, 300))
    os.remove(ta); os.remove(tb)


if __name__ == "__main__":
    main()
