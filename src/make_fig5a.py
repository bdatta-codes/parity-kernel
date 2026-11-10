#!/usr/bin/env python3
"""
make_fig5a.py — Figure 5A from results/cds/fig5a_auc_by_orf_length_default.csv.

Plots, for each ORF-length bin, the mean AUC over repeated draws of
length-matched non-coding windows (coding_sequence_analysis.py) with the
2.5-97.5% band over draws, for the frame-resolved index, the frame-free index
and Fickett's position asymmetry. Bins are placed at their geometric-mean ORF
length. The complete Figure 5 is drawn by make_fig5.py, which uses
render_panel_a() from this file; run alone, this script draws panel A only:

    python3 src/make_fig5a.py results/cds/fig5a_auc_by_orf_length_default.csv \\
        --out figures/Figure5A.png

License: MIT
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BG, INK, GRID, MUTED = "#FCFCFB", "#0B0B0B", "#E3E3E1", "#6F6E6B"
SERIES = [  # column stem, label, colour, marker, bold label
    ("fickett", "Fickett 1982\nposition asymmetry", "#52514E", "^", True),
    ("frame_resolved", "frame-resolved\nsymmetry", "#1BAF7A", "o", True),
    ("frame_free", "frame-free\nsymmetry", "#2A78D6", "s", False),
]
BINS = ["60-100", "100-150", "150-250", "250-400", "400-800", "800-+"]


def geo(b):
    lo, hi = b.split("-")
    if hi == "+":                     # open top bin: placed at 800 x sqrt(2)
        return float(lo) * math.sqrt(2)
    return math.sqrt(float(lo) * float(hi))


def render_panel_a(csv, tmp, split=1500):
    """Draw panel A from the Figure 5A CSV and save it to `tmp` (PNG,
    `split` x 1214 px at 300 dpi). Used by main() and by make_fig5.py."""
    df = pd.read_csv(csv).set_index("orf_codon_bin").loc[BINS]
    x = [geo(b) for b in BINS]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(split / 300, 1214 / 300), dpi=300)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for stem, label, col, mk, bold in SERIES:
        m = df[f"auc_{stem}"].values
        ax.fill_between(x, df[f"auc_{stem}_lo"], df[f"auc_{stem}_hi"],
                        color=col, alpha=0.13, lw=0)
        ax.plot(x, m, color=col, lw=2.0, marker=mk, ms=6.5, mfc=BG, mew=1.8,
                zorder=3)
        ax.annotate(label, (x[-1], m[-1]), xytext=(9, {"fickett": 8, "frame_resolved": -17}.get(stem, 0)),
                    textcoords="offset points", color=col if stem != "fickett" else INK,
                    fontsize=8.5, fontweight="bold" if bold else "normal",
                    va="center", annotation_clip=False)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{round(v, -1):.0f}" if not b.endswith("+") else "≥800" for v, b in zip(x, BINS)])
    ax.minorticks_off()
    ax.set_xlim(x[0] / 1.35, x[-1] * 2.6)
    ax.set_ylim(0.48, 1.02)
    ax.set_xlabel("ORF length (codons, bin geometric mean)", color=MUTED)
    ax.set_ylabel("AUC, real coding vs real non-coding", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    fig.text(0.115, 0.958, "A   Coding vs non-coding AUC, by ORF length",
             fontweight="bold", fontsize=10.5, color=INK, va="center")
    fig.subplots_adjust(left=0.115, right=0.80, top=0.865, bottom=0.135)
    fig.savefig(tmp, dpi=300, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    render_panel_a(a.csv, a.out)


if __name__ == "__main__":
    main()
