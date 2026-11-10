#!/usr/bin/env python3
"""si_additional_tables.py - Supplementary Tables S4, S5, S9, S15-S17, S20, S23, S25 and S26 from results/additional/.

S15 placement null (placement_null.py); S20 kernel decomposition
(kernel_decomposition.py); S25 coding sequence at quadruplex lengths and S4
GC-matched chromosome 21 windows (length_matched.py); S26 planted features
(planted_control.py); S5 decomposition of the chromosome 21 deficit
(chr21_markov_decomposition.py); S9 consensus-positive wild types
(consensus_positive_subset.py); S16 placement-null robustness
(placement_null_variants.py); S17 matched placement (matched_placement.py); S23 experimentally labelled
sequences (experimental_labels.py).
Writes the corresponding results/supplementary_tables/TableS*.csv.

License: MIT
"""
import os
import pandas as pd
REN = {"G→T, two tracts only": "G→T, two tracts", "first tract replaced by thymine": "First tract ablated"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "results", "additional")
T = os.path.join(ROOT, "results", "supplementary_tables")
ci = lambda r, k: f"{r[k + '_mean_q']:.3f} ({r[k + '_lo']:.3f}–{r[k + '_hi']:.3f})"

d = pd.read_csv(os.path.join(A, "placement_null.csv"))
pd.DataFrame([{"Convention": REN.get(r.convention, r.convention), "Pairs": r.pairs, "Pairs with a consensus-positive null mutant": r.pairs_with_null,
               "nMSI": ci(r, "nMSI"), "MSI": ci(r, "MSI"), "G4Hunter": ci(r, "G4Hunter")}
              for _, r in d.iterrows()]).to_csv(os.path.join(T, "TableS15_placement_null.csv"), index=False)

d = pd.read_csv(os.path.join(A, "kernel_decomposition.csv"))
key = [("98 consensus-negative reference", "mono"), ("98 consensus-negative reference", "dinuc"),
       ("538 mutagenesis wild types", "dinuc")]
rows = []
for f, g in d.groupby("feature_set", sort=False):
    row = {"Feature set": f, "Features": int(g.n_features.iloc[0])}
    for s, sh in key:
        r = g[(g.set == s) & (g.shuffle == sh)].iloc[0]
        row[f"{s.split()[0]} vs {sh}"] = f"{r.auc:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})"
    fc = os.path.join(A, "kernel_decomposition_control.csv")
    if os.path.exists(fc):
        c = pd.read_csv(fc); r = c[c.feature_set == f].iloc[0]
        row["chr21 control vs dinuc"] = f"{r.auc:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})"
    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(T, "TableS20_kernel_decomposition.csv"), index=False)

d = pd.read_csv(os.path.join(A, "length_matched_coding.csv"))
f = lambda r, k: f"{r['auc_' + k]:.3f} ({r['auc_' + k + '_lo']:.3f}–{r['auc_' + k + '_hi']:.3f})"
pd.DataFrame([{"Length (codons / nt)": f"{int(r.codons)} / {int(r.nt)}", "Coding sequences": int(r.n_coding),
               "Frame-resolved": f(r, "frame_resolved"), "Frame-free": f(r, "frame_free"), "Fickett": f(r, "fickett")}
              for _, r in d.iterrows()]).to_csv(os.path.join(T, "TableS25_coding_fragments.csv"), index=False)

d = pd.read_csv(os.path.join(A, "length_matched_chr21.csv"))
pd.DataFrame([{"Motif": r.label, "Windows with motif": int(r.n_with), "Matched per class": int(r.n_matched_each),
               "nMSI": f(r, "nMSI"), "Deficit": f(r, "deficit"), "|G4Hunter|": f(r, "G4Hunter")}
              for _, r in d.iterrows()]).to_csv(os.path.join(T, "TableS4_chr21_gc_matched.csv"), index=False)

d = pd.read_csv(os.path.join(A, "planted_control.csv")).round(3)
d.to_csv(os.path.join(T, "TableS26_planted_features.csv"), index=False)

d = pd.read_csv(os.path.join(A, "chr21_markov_decomposition.csv")).round({"weighted_mean": 4, "share_of_deficit": 3})
d.to_csv(os.path.join(T, "TableS5_chr21_deficit_decomposition.csv"), index=False)
LAB = {"T_centre": "G→T, tract centre", "A_centre": "G→A, tract centre", "C_centre": "G→C, tract centre",
       "two_tract": "G→T, two tracts", "tract_ablate": "First tract ablated", "T_3prime": "G→T, 3′ edge",
       "T_5prime": "G→T, 5′ edge", "scramble": "Whole-sequence scramble"}


def fp(p):
    if p >= 0.001:
        return f"{p:.3f}" if p < 0.995 else "1.000" if p > 0.9995 else f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m} × 10{str(int(e)).translate(sup)}"


f = os.path.join(A, "consensus_positive_subset.csv")
if os.path.exists(f):
    d = pd.read_csv(f)
    pd.DataFrame([{"Convention": LAB[r.convention], "n": int(r.n), "Δ = 0 (%)": f"{r.delta0_pct:.1f}",
                   "Concord. (%)": f"{r.concord_pct:.1f}", "p, fair": fp(r.p_fair),
                   "Null, mono": f"{r.null_nMSI_mono:.3f}", "p, mono": fp(r.p_nMSI_mono),
                   "Null, dinuc": f"{r.null_nMSI_di:.3f}", "p, dinuc": fp(r.p_nMSI_di)} for _, r in d.iterrows()]
                 ).to_csv(os.path.join(T, "TableS9_consensus_positive_subset.csv"), index=False)
f = os.path.join(A, "placement_null_variants.csv")
if os.path.exists(f):
    d = pd.read_csv(f)
    def ci(r, nm):
        return f"{r[nm + '_mean_q']:.3f}" + ("" if pd.isna(r.get(nm + "_lo")) else f" ({r[nm + '_lo']:.3f}–{r[nm + '_hi']:.3f})")
    pd.DataFrame([{"Variant": r.variant, "Convention": REN.get(r.convention, r.convention), "Pairs": int(r.pairs),
                   "nMSI": ci(r, "nMSI"), "MSI": ci(r, "MSI"), "G4Hunter": ci(r, "G4Hunter")}
                  for _, r in d.iterrows()]).to_csv(os.path.join(T, "TableS16_placement_null_variants.csv"), index=False)
f = os.path.join(A, "matched_placement.csv")
if os.path.exists(f):
    d = pd.read_csv(f)
    pd.DataFrame([{"Design": r.design, "Wild types": int(r.wild_types),
                   "of which A549": int(r["n_A549 G-rich"]), "nMSI": ci(r, "nMSI"), "MSI": ci(r, "MSI"), "G4Hunter": ci(r, "G4Hunter")}
                  for _, r in d.iterrows()]).to_csv(os.path.join(T, "TableS17_matched_placement.csv"), index=False)
f = os.path.join(A, "experimental_labels_auc.csv")
if os.path.exists(f):
    d = pd.read_csv(f)
    NM = {"MSI": "MSI", "nMSI": "nMSI", "nMSI_z": "nMSI z-score against own mononucleotide shuffles",
          "nMSI_shuffle_mean": "Mean nMSI of own mononucleotide shuffles (composition only)", "G4Hunter": "G4Hunter",
          "Quadparser": "Quadparser consensus", "G_fraction": "Guanine fraction", "G3_tracts": "Number of G≥3 tracts"}
    rows = []
    for _, r in d.iterrows():
        if r.score in NM:
            rows.append({"Score": NM[r.score], "AUC (95% interval)": f"{r.auc:.3f} ({r.lo:.3f}–{r.hi:.3f})"})
        else:
            rows.append({"Score": r.score, "AUC (95% interval)": f"{r.auc:.3f}"})
    pd.DataFrame(rows).to_csv(os.path.join(T, "TableS23_experimental_labels.csv"), index=False)
print("written S4, S5, S9, S15-S17, S20, S23, S25, S26")
