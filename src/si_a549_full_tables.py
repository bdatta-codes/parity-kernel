#!/usr/bin/env python3
"""si_a549_full_tables.py -- Supplementary Tables S14 and S21 from results/a549_full/.

S21  the A549 analyses on all motifs, on the distinct sequences, on the first 300
     (the A549 G-rich set) and on the remainder (a549_full_set.py)
S14  the mutagenesis experiment (A) and the placement-matched test (B) on every
     eligible A549 motif (a549_full_mutagenesis.py)
Writes results/supplementary_tables/TableS21_a549_all_motifs.csv, TableS14a_*.csv, TableS14b_*.csv
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "a549_full")
OUT = os.path.join(ROOT, "results", "supplementary_tables")

SUBSETS = ["all motifs", "distinct sequences", "first 300 (A549 G-rich set)", "all but the first 300"]


def fp(p):
    if p == 0:
        return "<1e-300"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def table_s21():
    s = pd.read_csv(os.path.join(IN, "summary.csv"))
    w = s.pivot_table(index="quantity", columns="subset", values="value", sort=False)[SUBSETS]
    n = s.groupby("subset", sort=False).n.first()[SUBSETS]
    def v(q): return w.loc[q]
    rows = []
    def add(label, vals, fmt):
        rows.append([label] + [fmt(x) for x in vals])
    add("Sequences (n)", n.values, lambda x: f"{int(x):,}")
    add("Interior G≥3 runs with identical flanks (%)", v("% identical flanks"), lambda x: f"{x:.1f}")
    add("Share of nMSI numerator from boundary words (%)", v("% of nMSI numerator from boundary words"), lambda x: f"{x:.1f}")
    add("Δ number of G-runs, real − shuffles", v("mean delta number of G-runs (real - mono shuffles)"), lambda x: f"{x:+.2f}")
    add("Spearman ρ, Δ nMSI vs Δ G-runs", v("Spearman rho(d nMSI, d runs)"), lambda x: f"{x:+.3f}")
    for kind, lab in (("mono", "mononucleotide"), ("dinuc", "dinucleotide")):
        for name in ("nMSI", "MSI"):
            q = f"pooled AUC real vs own {kind} shuffles, {name}"
            vals = [f"{a:.3f} ({b:.3f}–{c:.3f})" for a, b, c in
                    zip(v(q), v(q + ", 95% CI low"), v(q + ", 95% CI high"))]
            rows.append([f"AUC real vs own {lab} shuffles, {name} (95% interval)"] + vals)
            add(f"  own {lab} shuffles outscored, {name} (%, ties one half)",
                v(f"% of own {kind} shuffles beaten, ties one half, {name}"), lambda x: f"{x:.1f}")
            add(f"  paired Wilcoxon p, {name}", v(f"paired Wilcoxon p (real vs {kind} shuffle mean), {name}"), fp)
    add("Spearman ρ, MSI vs G4Hunter", v("rho MSI vs G4Hunter"), lambda x: f"{x:+.3f}")
    add("Spearman ρ, nMSI vs G4Hunter", v("rho nMSI vs G4Hunter"), lambda x: f"{x:+.3f}")
    add("  nMSI mean of 1,000 mononucleotide shuffles vs G4Hunter", v("rho mean mono-shuffled nMSI vs WT G4Hunter"), lambda x: f"{x:+.3f}")
    add("  nMSI mean of 200 dinucleotide shuffles vs G4Hunter", v("rho mean dinuc-shuffled nMSI vs WT G4Hunter"), lambda x: f"{x:+.3f}")
    add("  ratio of shuffle-mean ρ to real ρ, mono / dinuc (%; see Supplementary Table S22)",
        [f"{100 * a:.0f} / {100 * b:.0f}" for a, b in zip(v("share of association kept by mono shuffles"),
                                                          v("share of association kept by dinuc shuffles"))], str)
    T = pd.DataFrame(rows, columns=["Quantity"] + ["All motifs", "Distinct sequences", "First 300 (chr1)", "Remainder"])
    T.to_csv(os.path.join(OUT, "TableS21_a549_all_motifs.csv"), index=False)
    return T


LAB = {"T_centre": "G→T centre", "T_3prime": "G→T 3′ edge", "T_5prime": "G→T 5′ edge",
       "A_centre": "G→A centre", "C_centre": "G→C centre", "two_tract": "two tracts (G→T centre)",
       "tract_ablate": "first tract ablated", "scramble": "whole-sequence scramble"}


def table_s14():
    m = pd.read_csv(os.path.join(IN, "mutagenesis.csv"))
    rows = []
    for conv, g in m.groupby("convention", sort=False):
        a = g.set_index("index")
        n_, mo, di = a.loc["nMSI"], a.loc["MSI"], a.loc["nMSI/dinuc"]
        rows.append([LAB[conv], f"{int(n_.n):,}", f"{100 * n_.obs:.1f} ({100 * n_.obs_lo:.1f}–{100 * n_.obs_hi:.1f})",
                     f"{n_.pct_ties:.1f}", fp(n_.p_fair), f"{100 * n_.null:.1f}", fp(n_.p_corrected),
                     f"{100 * di.null:.1f}", fp(di.p_corrected), f"{100 * mo.obs:.1f}",
                     f"{100 * a.loc['G4Hunter (control)'].obs:.1f}"])
    A = pd.DataFrame(rows, columns=["Convention", "Pairs", "nMSI, % (95% CI)", "Ties %",
                                    "p fair coin", "Mono null %", "p mono", "Dinuc null %", "p dinuc",
                                    "MSI %", "G4H %"])
    A.to_csv(os.path.join(OUT, "TableS14a_a549_all_mutagenesis.csv"), index=False)
    p = pd.read_csv(os.path.join(IN, "matched_placement.csv"))
    ci = lambda r, k: f"{r[k + '_mean_q']:.3f} ({r[k + '_lo']:.3f}–{r[k + '_hi']:.3f})"
    B = pd.DataFrame([[r.design, f"{int(r.wild_types):,}", ci(r, "nMSI"), ci(r, "MSI"), ci(r, "G4Hunter")]
                      for _, r in p.iterrows()],
                     columns=["Design", "Wild types", "nMSI", "MSI", "G4Hunter"])
    B.to_csv(os.path.join(OUT, "TableS14b_a549_all_matched_placement.csv"), index=False)
    return A, B


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(table_s21().to_string(index=False))
        if os.path.exists(os.path.join(IN, "matched_placement.csv")):
            A, B = table_s14(); print(A.to_string(index=False)); print(B.to_string(index=False))
