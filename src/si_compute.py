#!/usr/bin/env python3
"""si_compute.py — Supplementary Tables S2, S8, S12, S13, S19 and S24 and the numerical checks of
Supplementary Table S6. Reads results/ and data/, writes
results/supplementary_tables/ and results/si/ (inputs for si_figs.py).
Run from anywhere: paths are resolved relative to the repository.

Each section draws from its own seeded generator, so the tables are
independent of one another and of the order in which they are computed.

License: MIT
"""
import itertools, json, os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity
import experiment_538 as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
TAB = os.path.join(RES, "supplementary_tables")
SI = os.path.join(RES, "si")
os.makedirs(TAB, exist_ok=True); os.makedirs(SI, exist_ok=True)
NSHUF = 20000          # shuffles per sequence for Tables S12 and S24
out = {}

PROM = [("c-MYC", "TGGGGAGGGTGGGGAGGGTGGGGAAGG"), ("VEGF", "GGGGCGGGCCGGGGGCGGGG"),
        ("BCL2", "GAAGGAGGGCAGGAGGGCTCTGGGTGGGTCTGT"),
        ("HIF-1α", "GCGCGGGGAGGGGAGAGGGGGCGGGAGCGCG"),
        ("KRAS", "AGGGCGGTGTGGGAAGAGGGAAGAGGGGGAGG"), ("BCR", "GGGCAGGGTGTGGGGAAACAGGG"),
        ("c-KIT", "AGGGAGGGCGCTGGGAGGAGGG"),
        ("MET", "CAGGGCAGCGCGCGTGTGGGAAGGGGCGGAGGGAG"),
        ("CARD11", "GGTGGGGTGGGAGTGGGGACAGGGCCGGGGGTGGGGGTGCTTGGGGACA"),
        ("ACC1", "GCTGGGCGGGCGGGCGGGCGGGTGG"), ("HRAS", "CGGGGCGGGGCGGGGGCGGGGGCG"),
        ("ATG7", "GGGGCTGGGGTCCCTTGGGGAACTGTATTGGG"),
        ("MDM2", "GTTCGTGGCTGGGGGCTCGGGGCGCGGGGCGCGGGGCATGGGGC"),
        ("ARID1A", "GGGAGGGGGCGGGGAGGG")]



def shuffles(s, n, rng):
    a = np.array(list(s))
    return ["".join(rng.permutation(a)) for _ in range(n)]


def comp(s):
    L = len(s)
    return tuple(s.count(b) / L for b in "ACGT")


def p_greater_exact(A, B):
    """P(A > B) + 0.5 P(A = B) over all pairs of the two samples, exactly."""
    Bs = np.sort(B)
    lt = np.searchsorted(Bs, A, side="left")
    le = np.searchsorted(Bs, A, side="right")
    return float((lt + 0.5 * (le - lt)).sum() / (len(A) * len(B)))


# --------------------------------------------- Table S12: promoter panel ----
rng6 = np.random.default_rng(20260926)
rng6b = np.random.default_rng(20261128)   # independent-base simulation (own generator)
import closed_form as CF
rows, null_store = [], {}
for g, s in PROM:
    nm = parity.msi_scores(s)[1]
    sh = np.array([parity.msi_scores(x)[1] for x in shuffles(s, NSHUF, rng6)])
    null_store[g] = sh
    N = len(s) - 2
    # analytic mean at the sequence's own composition: closed form plus the derived
    # large-N overlap constant kappa_inf at that composition (main text, Table 2)
    kinf = CF.kappa_inf(3, comp(s))[0]
    base = parity.nmsi_baseline_exact(N, *comp(s)) + kinf / np.sqrt(N)
    # independent bases at the same composition, simulated (mean and SD)
    iid = CF.simulate(len(s), 3, "mirror", comp(s), NSHUF, rng6b)
    iid = iid[np.isfinite(iid)]
    rows.append(dict(gene=g, L=len(s), GC=round(sum(s.count(b) for b in "GC") / len(s), 3),
                     nMSI=round(nm, 4), kappa_inf=round(kinf, 3), analytic_mean=round(base, 4),
                     iid_mean=round(float(iid.mean()), 4), iid_sd=round(float(iid.std(ddof=1)), 4),
                     emp_mean=round(sh.mean(), 4), emp_sd=round(sh.std(ddof=1), 4),
                     skew=round(float(skew(sh)), 3), exkurt=round(float(kurtosis(sh)), 2),
                     pct_ceiling=round(100 * float(np.mean(sh >= 0.999999)), 1),
                     z_iid=round(float((nm - iid.mean()) / iid.std(ddof=1)), 2),
                     z_emp=round(float((nm - sh.mean()) / sh.std(ddof=1)), 2)))
T6 = pd.DataFrame(rows)
T6.to_csv(os.path.join(TAB, "TableS12_promoter_panel.csv"), index=False)
out.update(median_skew=float(np.median(T6["skew"])),
           iid_mean_above_empirical=int((T6.iid_mean > T6.emp_mean).sum()),
           iid_sd_above_empirical=int((T6.iid_sd > T6.emp_sd).sum()),
           analytic_minus_iid_mean_range=[float((T6.analytic_mean - T6.iid_mean).min()),
                                          float((T6.analytic_mean - T6.iid_mean).max())])
print(T6.to_string(index=False))
print("median skew %.3f | iid mean > empirical in %d/14 | iid SD > empirical in %d/14 | analytic - iid %s"
      % (out["median_skew"], out["iid_mean_above_empirical"], out["iid_sd_above_empirical"],
         out["analytic_minus_iid_mean_range"]))

# --------------------------------------- Table S24: five published pairs ----
rng7 = np.random.default_rng(20260927)
SEQ = {"MYC WT": "CGGGAGGGGCGCTTATGGGGAGGGTGGGGAGGGTGGGGAAGGTGGGGA",
       "MYC MUT": "TAGGATCCGCGCTTATAAAGAGCGTTAAGAGCGTGAACAAGGTGAGGA",
       "MUT CORE": "CGGGAGGGGCGCTTATGGAGAGGATCCGGAGCGTGGAGAAGGTGGGGA",
       "MUT MIN": "CGCGAGGAGCGCTTATGGAGAGGATCCGGAGCGTGGAGAAGGTGAGGA",
       "GPX4 G4": "AAAACAGGGGGGGGGGGGGGGGGACCGACGTG",
       "mutGPX4": "TTTACAGGTGTGTGTGTGTGTGTGAACCGACGTG",
       "CLIC4 G4": "CTTCCAAACGGGGAGGGAGGGCGCGGACGG",
       "mutCLIC4": "CTTCCAAACGGAGAGAGAGCGCGGACGG"}
cache2 = {k: np.array([parity.msi_scores(x) for x in shuffles(v, NSHUF, rng7)])
          for k, v in SEQ.items()}                     # columns: MSI, nMSI
cache = {k: v[:, 1] for k, v in cache2.items()}
prs = [("MYC WT", "MYC MUT"), ("MYC WT", "MUT CORE"), ("MYC WT", "MUT MIN"),
       ("GPX4 G4", "mutGPX4"), ("CLIC4 G4", "mutCLIC4")]
rows = []
for a, b in prs:
    A, B = cache[a], cache[b]
    na, nb = parity.msi_scores(SEQ[a])[1], parity.msi_scores(SEQ[b])[1]
    rows.append(dict(pair=f"{a} / {b}", wt_nMSI=round(na, 3), mut_nMSI=round(nb, 3),
                     wt_higher="yes" if na > nb else "no",
                     p_null=round(p_greater_exact(A, B), 3),
                     p_null_MSI=round(p_greater_exact(cache2[a][:, 0], cache2[b][:, 0]), 3),
                     z_wt=round(float((na - A.mean()) / A.std(ddof=1)), 2),
                     z_mut=round(float((nb - B.mean()) / B.std(ddof=1)), 2)))
T7 = pd.DataFrame(rows)
T7.to_csv(os.path.join(TAB, "TableS24_published_pairs.csv"), index=False)
p = T7.p_null.values
pb = np.zeros(15); pb[0] = 1.0
for _ in range(14):                     # 12 of 14 against a null of mean(p)
    pb[1:] = pb[1:] * (1 - p.mean()) + pb[:-1] * p.mean(); pb[0] *= (1 - p.mean())
out.update(p0_range=[float(p.min()), float(p.max())], p0_mean=round(float(p.mean()), 3),
           p_12_of_14=float(pb[12:].sum()),
           z_wt_mean=round(float(T7.z_wt.mean()), 2), z_mut_mean=round(float(T7.z_mut.mean()), 2),
           p0_MSI_range=[float(T7.p_null_MSI.min()), float(T7.p_null_MSI.max())],
           p0_MSI_mean=round(float(T7.p_null_MSI.mean()), 3))
print(T7.to_string(index=False))
print("null range", out["p0_range"], "mean", out["p0_mean"], "| 12/14 tail %.4f" % out["p_12_of_14"],
      "| mean z", out["z_wt_mean"], out["z_mut_mean"])

# --------------------------------- Table S13: all conventions x measures ----
NAME = {"T_centre": "G→T, tract centre", "A_centre": "G→A, tract centre",
        "C_centre": "G→C, tract centre", "two_tract": "G→T, two tracts only",
        "tract_ablate": "first tract replaced by thymine", "T_3prime": "G→T, 3′ edge",
        "T_5prime": "G→T, 5′ edge", "scramble": "whole-sequence scramble"}
R = pd.read_csv(os.path.join(RES, "experiment_538_results.csv"))
R["Convention"] = R["convention"].map(NAME)
T3 = R[["Convention", "index", "n", "obs", "null", "p_fair", "p_corrected", "mean_delta"]]
T3.columns = ["Convention", "Index", "n", "Concordance", "Measured null",
              "p (fair coin)", "p (corrected)", "mean Δ index"]
T3.to_csv(os.path.join(TAB, "TableS13_all_conventions.csv"), index=False)

# ------------------------------- Table S8: verification of abolition -------
rows = []
for k in NAME:
    d = pd.read_csv(os.path.join(RES, f"pairs_{k}.csv"))
    dn = d.wt_nMSI - d.mut_nMSI
    rows.append(dict(Convention=NAME[k], n=len(d),
                     mean_wt_G4H=round(d.wt_g4.mean(), 3), mean_mut_G4H=round(d.mut_g4.mean(), 3),
                     mean_drop=round((d.wt_g4 - d.mut_g4).mean(), 3),
                     pct_zero_delta=round(100 * float((dn == 0).mean()), 1),
                     mean_delta_nMSI=round(float(dn.mean()), 4),
                     # paired null, mean of P(null WT higher) + P(tie)/2 (Supplementary Methods S8)
                     mean_null_mono=round(float((d.g_nMSI_mono + d.t_nMSI_mono / 2).mean()), 3),
                     mean_null_dinuc=round(float((d.g_nMSI_di + d.t_nMSI_di / 2).mean()), 3)))
T4 = pd.DataFrame(rows)
T4.to_csv(os.path.join(TAB, "TableS8_verification.csv"), index=False)

# ------------------------------ Table S19: single-substitution scan ---------
S = pd.read_csv(os.path.join(RES, "mutagenesis_scan.csv"))   # delta = mutant - wild type
POS = {"5' edge": "5′ edge", "interior": "tract interior", "3' edge": "3′ edge"}
g = S.groupby(["position", "base"])["delta"]
T5 = g.agg(n="size", pct_raise=lambda x: round(100 * float((x > 0).mean()), 1),
           pct_zero=lambda x: round(100 * float((x == 0).mean()), 1),
           pct_lower=lambda x: round(100 * float((x < 0).mean()), 1),
           mean_delta=lambda x: round(float(x.mean()), 4)).reset_index()
T5["position"] = pd.Categorical(T5.position, ["3' edge", "5' edge", "interior"])
T5 = T5.sort_values(["position", "base"])
T5["position"] = T5.position.map(POS); T5["base"] = "G→" + T5.base.astype(str)
T5.to_csv(os.path.join(TAB, "TableS19_substitution_scan.csv"), index=False)
out.update(scan_sequences=int(S.seq.nunique()), scan_total=int(len(S)),
           scan_pct_raise=round(100 * float((S.delta > 0).mean()), 1),
           scan_pct_zero=round(100 * float((S.delta == 0).mean()), 1),
           scan_pct_lower=round(100 * float((S.delta < 0).mean()), 1),
           scan_pct_lower_by_position={POS[p]: round(100 * float((d.delta < 0).mean()), 1)
                                       for p, d in S.groupby("position")})

# ----------------------------- Table S2: composition of the collections ----
ref = pd.read_excel(os.path.join(DATA, "298_sequences_with_MSI_nMSI.xlsx"))
refseq = [s.upper() for s in ref["WT_Sequence (5'-3')"] if isinstance(s, str)]
a549 = [l.split("\t")[1].strip().upper() for l in open(os.path.join(DATA, "a549_grich_300.txt")) if l.strip()]
prom = [s for _, s in PROM]
qp = re.compile(r"G{3,}\w{1,7}G{3,}\w{1,7}G{3,}\w{1,7}G{3,}")


def summarise(name, seqs):
    L = np.array([len(s) for s in seqs])
    gc = np.array([sum(s.count(b) for b in "GC") / len(s) for s in seqs])
    tr = np.array([len(re.findall(r"G{3,}", s)) for s in seqs])
    return dict(Collection=name, n=len(seqs), median_length=int(np.median(L)),
                length_range=f"{L.min()}–{L.max()}", median_GC=round(float(np.median(gc)), 3),
                median_G_tracts=float(np.median(tr)),
                pct_Quadparser_positive=round(100 * float(np.mean([bool(qp.search(s)) for s in seqs])), 1),
                # the mutagenesis set: ACGT-only sequences of >= 16 nt with >= 4 tracts
                n_with_4plus_tracts=int(sum(1 for s, t in zip(seqs, tr)
                                            if t >= 4 and set(s) <= set("ACGT") and len(s) >= 16)))


T2 = pd.DataFrame([summarise("G4Hunter reference set", refseq),
                   summarise("A549 G-rich set", a549),
                   summarise("Oncogene-promoter panel", prom)])
T2.to_csv(os.path.join(TAB, "TableS2_datasets.csv"), index=False)
print(T2.to_string(index=False))

# ------------------------- Table S6: implementation and identity checks ----
# (a) the reference implementation against the stored MSI/nMSI of the 298
#     reference sequences and their 298 stored shuffles (1,192 values)
diffs = []
for sc, mc, nc in (("WT_Sequence (5'-3')", "WT_MSI", "WT_nMSI"),
                   ("Shuffled_Sequence (5'-3')", "Shuffled_MSI", "Shuffled_nMSI")):
    for s, m, n in zip(ref[sc], ref[mc], ref[nc]):
        a, b = parity.msi_scores(str(s).upper())
        if a is not None:
            diffs += [abs(a - m), abs(b - n)]
out.update(stored_values_compared=len(diffs), stored_max_abs_diff=float(max(diffs)))
# (b) Eq. 2 against enumeration, in exact rational arithmetic
from fractions import Fraction
from math import comb
eq2_identical = all(Fraction(sum(comb(m, x) * abs(2 * x - m) for x in range(m + 1)), 2 ** m)
                    == Fraction(m * comb(m - 1, (m - 1) // 2), 2 ** (m - 1)) for m in range(1, 56))
out["eq2_identical_m_le_55"] = bool(eq2_identical)
out["binom_mad_float_max_rel_err"] = max(
    abs(parity.binom_mad(m) - float(Fraction(m * comb(m - 1, (m - 1) // 2), 2 ** (m - 1))))
    / float(Fraction(m * comb(m - 1, (m - 1) // 2), 2 ** (m - 1))) for m in range(1, 56))
# (b2) the remaining rows of Table S6
out["g4h_telomere_GGGTTAGGGTTAGGGTTAGGG"] = round(parity.g4hunter("GGGTTAGGGTTAGGGTTAGGG"), 3)
out["g4h_cMYC_Pu27"] = round(parity.g4hunter("TGGGGAGGGTGGGGAGGGTGGGGAAGG"), 3)
_g = np.array([parity.g4hunter(str(x)) for x in ref["WT_Sequence (5'-3')"]])
out["g4h_pearson_vs_stored_298"] = float(np.corrcoef(_g, ref["WT_G4H Score"].astype(float))[0, 1])
out["quadparser_positive_of_298"] = int(sum(E.is_canonical(str(x)) for x in ref["WT_Sequence (5'-3')"]))
# the exact tail used for every test (three outcomes), with all probabilities 1/2 and no ties
out["exact_tail_12_of_14_all_half_no_ties"] = float(E.three_outcome_sf(12, [0.5] * 14, [0.0] * 14))
# (b2b) P(w) = P(rev w) under i.i.d. bases: GC 0.10-0.90 (step 0.05), G-C skew -0.9..0.9 (step 0.1)
_w = 0.0
for gc in np.arange(0.10, 0.901, 0.05):
    for sk in np.arange(-0.9, 0.901, 0.1):
        p4 = np.array([(1 - gc) / 2, gc * (1 - sk) / 2, gc * (1 + sk) / 2, (1 - gc) / 2])
        for a_, b_, c_ in itertools.product(range(4), repeat=3):
            _w = max(_w, abs(p4[a_] * p4[b_] * p4[c_] - p4[c_] * p4[b_] * p4[a_]))
out["iid_max_abs_P_w_minus_P_revw"] = _w
# (b3) convention spreads quoted in the Results (seven designed conventions)
_K = ["T_centre", "A_centre", "C_centre", "two_tract", "tract_ablate", "T_3prime", "T_5prime"]
_P = {k: pd.read_csv(os.path.join(RES, f"pairs_{k}.csv")).set_index("i") for k in _K}
def _conc(x, mode):
    w, t = (x > 1e-12), np.isclose(x, 0)
    return 100 * (w.mean() + 0.5 * t.mean() if mode == "half" else w.mean() if mode == "discordant"
                  else w.sum() / (len(x) - t.sum()))
for mode in ("half", "discordant", "excluded"):
    v = [_conc((_P[k].wt_nMSI - _P[k].mut_nMSI).values, mode) for k in _K]
    out[f"spread_ties_{mode}"] = [round(min(v), 1), round(max(v), 1), round(max(v) - min(v), 1)]
_common = sorted(set.intersection(*[set(d.index) for d in _P.values()]))
v = [_conc((_P[k].loc[_common].wt_nMSI - _P[k].loc[_common].mut_nMSI).values, "half") for k in _K]
out["spread_common_sequences"] = [len(_common), round(min(v), 1), round(max(v), 1), round(max(v) - min(v), 1)]
# null at which 12 of 14 stops being significant at 0.05
from scipy.optimize import brentq
from scipy.stats import binom
out["null_threshold_12_of_14"] = float(brentq(lambda p: binom.sf(11, 14, p) - 0.05, 0.5, 0.9))
# (c) reversibility: P(w) = P(rev w) under reversible but not general chains
rngr = np.random.default_rng(20260928)


def rev_check(n=400, reversible=True):
    worst = 0.0
    for _ in range(n):
        if reversible:
            J = rngr.random((4, 4)); J = J + J.T; J /= J.sum()
            pi = J.sum(1); T = J / pi[:, None]
        else:
            T = rngr.random((4, 4)); T /= T.sum(1, keepdims=True)
            w, v = np.linalg.eig(T.T); pi = np.real(v[:, np.argmin(abs(w - 1))]); pi /= pi.sum()
        for a, b, c in itertools.product(range(4), repeat=3):
            worst = max(worst, abs(pi[a] * T[a, b] * T[b, c] - pi[c] * T[c, b] * T[b, a]))
    return worst


out["rev_worst_reversible"] = rev_check(400, True)
out["rev_worst_unconstrained"] = rev_check(400, False)
print("Table S6 extra checks:", {k: out[k] for k in ("binom_mad_float_max_rel_err", "g4h_telomere_GGGTTAGGGTTAGGGTTAGGG",
      "g4h_cMYC_Pu27", "g4h_pearson_vs_stored_298", "quadparser_positive_of_298", "exact_tail_12_of_14_all_half_no_ties", "spread_ties_half", "spread_ties_discordant", "spread_ties_excluded", "spread_common_sequences", "null_threshold_12_of_14", "iid_max_abs_P_w_minus_P_revw")})
print("stored values: max |diff| %.2e over %d | Eq.2 identical: %s | reversible %.1e, unconstrained %.2e"
      % (out["stored_max_abs_diff"], out["stored_values_compared"], eq2_identical,
         out["rev_worst_reversible"], out["rev_worst_unconstrained"]))

json.dump(out, open(os.path.join(SI, "si_summary.json"), "w"), indent=1, default=float)
json.dump({k: v.tolist() for k, v in null_store.items()}, open(os.path.join(SI, "promoter_nulls.json"), "w"))
print("DONE")
