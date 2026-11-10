#!/usr/bin/env python3
"""si_composition_and_matching_tables.py -- Supplementary Tables S11, S18 and S22.

S11  the 22 experimentally labelled sequences: length, guanine fraction, nMSI and its
     composition-matched baseline (mean nMSI of 2,000 mononucleotide shuffles of the same
     sequence; results/additional/experimental_labels.csv) against the nMSI >= 0.50 criterion
S22  how much of the index-propensity correlation composition reproduces
     (results/g4/composition_share.csv; composition_share.py)
S18  the placement-matched test matched also on the length of the hit tracts, and
     additionally on the bases flanking them
     (results/additional/ and results/a549_full/matched_placement_length.csv)
(S1, the 14 wild-type/mutant pairs of ref. 21, is written by ref21_pairs.py.)
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); TAB = os.path.join(RES, "supplementary_tables")


def table_s11():
    d = pd.read_csv(os.path.join(RES, "additional", "experimental_labels.csv"))
    T = pd.DataFrame({"Sequence": d.name.str.replace("_", " "),
                      "Folds": d.folds.map({1: "yes", 0: "no"}), "Length (nt)": d.length,
                      "G fraction": d.G_fraction.map("{:.2f}".format), "nMSI": d.nMSI.map("{:.3f}".format),
                      "Composition baseline": d.nMSI_shuffle_mean.map("{:.3f}".format),
                      "Baseline ≥ 0.50": (d.nMSI_shuffle_mean >= 0.5).map({True: "yes", False: "no"}),
                      "nMSI ≥ 0.50": (d.nMSI >= 0.5).map({True: "yes", False: "no"}),
                      "z vs own shuffles": d.nMSI_z.map("{:+.2f}".format)})
    T.to_csv(os.path.join(TAB, "TableS11_labelled_composition_baselines.csv"), index=False)
    return T


def fp(p):
    return "p < 1e-60" if p < 1e-60 else (f"p = {p:.1e}" if p < 1e-3 else f"p = {p:.3f}")


def table_s22():
    d = pd.read_csv(os.path.join(RES, "g4", "composition_share.csv"))
    f = lambda m, lo, hi: f"{m:+.3f} ({lo:+.3f} to {hi:+.3f})"
    T = pd.DataFrame({"Set": [f"{a} (n = {b:,})" for a, b in zip(d.set, d.n)], "Index": d["index"],
                      "ρ, real": d.rho_real.map("{:+.3f}".format),
                      "ρ, one mono shuffle (95% range)": [f(*r) for r in d[["rho_single_mono", "rho_single_mono_lo", "rho_single_mono_hi"]].values],
                      "ρ, one dinuc shuffle (95% range)": [f(*r) for r in d[["rho_single_dinuc", "rho_single_dinuc_lo", "rho_single_dinuc_hi"]].values],
                      "Share, mono (95% bootstrap CI; %)": [f"{100*a:.0f} ({100*b:.0f}–{100*c:.0f})" for a, b, c in d[["share_single_mono", "share_single_mono_boot_lo", "share_single_mono_boot_hi"]].values],
                      "Share, dinuc (95% bootstrap CI; %)": [f"{100*a:.0f} ({100*b:.0f}–{100*c:.0f})" for a, b, c in d[["share_single_dinuc", "share_single_dinuc_boot_lo", "share_single_dinuc_boot_hi"]].values],
                      "Dispersion ratio, mono / dinuc": [f"{a:.2f} / {b:.2f}" for a, b in d[["dispersion_ratio_mono", "dispersion_ratio_dinuc"]].values],
                      "Variance-matched share, mono / dinuc (%)": [f"{100*a:.0f} / {100*b:.0f}" for a, b in d[["share_variance_matched_mono", "share_variance_matched_dinuc"]].values],
                      "Partial ρ | mono mean (p)": [f"{a:+.3f} ({fp(p)})" for a, p in d[["partial_rho_mono", "partial_p_mono"]].values],
                      "Partial ρ | dinuc mean (p)": [f"{a:+.3f} ({fp(p)})" for a, p in d[["partial_rho_dinuc", "partial_p_dinuc"]].values]})
    T.to_csv(os.path.join(TAB, "TableS22_composition_share.csv"), index=False)
    return T


def table_s18():
    parts = [pd.read_csv(os.path.join(RES, "additional", "matched_placement_length.csv")),
             pd.read_csv(os.path.join(RES, "a549_full", "matched_placement_length.csv"))]
    d = pd.concat(parts)
    ci = lambda r, k: f"{r[k + '_mean_q']:.3f} ({r[k + '_lo']:.3f}–{r[k + '_hi']:.3f})"
    T = pd.DataFrame([[r.set, r.design, f"{r.mean_hit_length_abolishing:.2f} / {r.mean_hit_length_retaining:.2f}",
                       f"{int(r.wild_types):,}", ci(r, "nMSI"), ci(r, "MSI"), ci(r, "G4Hunter"),
                       f"{int(r.wild_types_flank):,}", ci(r, "nMSI_flank"), ci(r, "MSI_flank")]
                      for _, r in d.iterrows()],
                     columns=["Set", "Design", "Hit tract length, abolishing / retaining (nt)",
                              "Wild types (length-matched)", "nMSI", "MSI", "G4Hunter (equal by construction)",
                              "Wild types (length- and flank-matched)", "nMSI, flank-matched", "MSI, flank-matched"])
    T.to_csv(os.path.join(TAB, "TableS18_matched_placement_length.csv"), index=False)
    return T


if __name__ == "__main__":
    with pd.option_context("display.width", 250, "display.max_colwidth", 50):
        for f in (table_s11, table_s22, table_s18):
            try:
                print(f().to_string(index=False))
            except FileNotFoundError as e:
                print("missing:", e)
