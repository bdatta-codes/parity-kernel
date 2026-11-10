#!/usr/bin/env python3
"""quoted_counts.py - counts and simple tests quoted in the text that are not
written by another script: the four-tract/more-tract split of the placement
null; tract counts and even-length tracts; wild types failing the consensus;
lengths against the 51-nt minimum; chr21 GC percentiles and the
prediction-GC correlation; the sign tests of the true frame against each
shifted frame (from results/cds/frame_values.csv); the k = 3 minimum length at reference
levels 0.40-0.60 (sliding-window baseline, closed form plus kappa); and the
tie rate of the G->T centre convention against the same number of
substitutions placed at random (seed 20261118). Writes results/si/quoted_counts.json.

License: MIT
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
seqs, src = E.load()
out = {}
nt = np.array([len(E.tracts(s)) for s in seqs])
tl = [b - a for s in seqs for a, b in E.tracts(s)]
out["tracts_total"] = len(tl); out["tracts_even_length"] = int(sum(t % 2 == 0 for t in tl))
neg = [i for i, s in enumerate(seqs) if not E.is_canonical(s)]
out["wild_types_failing_consensus"] = len(neg)
out["pairs_from_failing_wild_types"] = int(sum(pd.read_csv(os.path.join(RES, f"pairs_{k}.csv")).i.isin(neg).sum()
                                               for k in ["T_centre", "A_centre", "C_centre", "two_tract", "tract_ablate", "T_3prime", "T_5prime", "scramble"]))
L = np.array([len(s) for s in seqs])
out["fraction_shorter_than_51nt"] = float((L < 51).mean()); out["median_length"] = float(np.median(L))
pp = pd.read_csv(os.path.join(RES, "additional", "placement_null_per_pair.csv"))
d = pd.read_csv(os.path.join(RES, "pairs_T_centre.csv"))
tested = set(pp[pp.convention == "G→T, tract centre"].i)
four = d.i[nt[d.i] == 4]; more = d.i[nt[d.i] > 4]
out["T_centre_pairs_four_tracts"] = [int(four.isin(tested).sum()), int(len(four))]
out["T_centre_pairs_more_tracts"] = [int(more.isin(tested).sum()), int(len(more))]
out["T_centre_tested_by_source"] = {s_: int(sum(1 for i in tested if src[i] == s_)) for s_ in sorted(set(src))}
w = pd.read_csv(os.path.join(RES, "chr21", "windows300.csv.gz"))
out["chr21_gc_p5_p95"] = [float(w.gc.quantile(.05)), float(w.gc.quantile(.95))]
out["chr21_rho_prediction_gc"] = float(stats.spearmanr(w.predicted, w.gc)[0])
fv = pd.read_csv(os.path.join(RES, "cds", "frame_values.csv"))
for j, col in ((1, "frame1"), (2, "frame2")):
    w_, l_ = int((fv.true_frame < fv[col]).sum()), int((fv.true_frame > fv[col]).sum())
    out[f"frame_true_below_{col}"] = dict(n_below=w_, n_above=l_,
                                          sign_test_p_two_sided=float(stats.binomtest(w_, w_ + l_, 0.5).pvalue),
                                          mean_difference=float((fv.true_frame - fv[col]).mean()),
                                          wilcoxon_p=float(stats.wilcoxon(fv.true_frame, fv[col]).pvalue))
import closed_form as CF
T3 = pd.read_csv(os.path.join(RES, "closed_form", "tableS7_word_length.csv"))
kap = float(T3[(T3.k == 3) & (T3.pairing == "mirror")].kappa.iloc[0])
out["min_length_k3_by_reference_level"] = {f"{lv:.2f}": int(CF.min_length(lambda L: CF.closed_form(L - 2, 3) + kap / np.sqrt(L - 2), 3, 20000, lv))
                                           for lv in (0.40, 0.45, 0.50, 0.55, 0.60)}
# tie-rate control: as many substitutions as the G->T centre convention makes,
# placed at random outside the tracts (G->T at a guanine, otherwise a random
# new base) or anywhere (random new base); 20 draws per verified pair
rng = np.random.default_rng(20261118)
dT = pd.read_csv(os.path.join(RES, "pairs_T_centre.csv"))
tie = {"real": [], "outside_tracts": [], "anywhere": []}
for i in dT.i:
    s_ = seqs[i]; tr = E.tracts(s_); k = len(tr); intr = {p for a, b in tr for p in range(a, b)}
    w0 = E.idx_msi(E.enc(s_))[1]
    m = E.mutate(s_, "T_centre", np.random.default_rng(0)); tie["real"].append(float(np.isclose(E.idx_msi(E.enc(m))[1], w0)))
    for key, pool in (("outside_tracts", [p for p in range(len(s_)) if p not in intr]), ("anywhere", list(range(len(s_))))):
        if len(pool) < k:
            continue
        t = []
        for _ in range(20):
            y = list(s_)
            for p in rng.choice(pool, k, replace=False):
                y[p] = "T" if (key == "outside_tracts" and y[p] == "G") else str(rng.choice([c for c in "ACGT" if c != y[p]]))
            t.append(np.isclose(E.idx_msi(E.enc("".join(y)))[1], w0))
        tie[key].append(float(np.mean(t)))
out["tie_rate_T_centre_vs_random_placement_pct"] = {k: [round(100 * float(np.mean(v)), 1), len(v)] for k, v in tie.items()}
# base-matched version of the same control: G->T only, at guanines outside the
# tracts (pairs with at least as many such guanines as tracts), and at guanines
# anywhere outside the substituted positions (own generator)
rng2 = np.random.default_rng(20261127)
tie2 = {"G_to_T_outside_tracts": [], "G_to_T_any_guanine": []}
for i in dT.i:
    s_ = seqs[i]; tr = E.tracts(s_); k = len(tr); intr = {p for a, b in tr for p in range(a, b)}
    w0 = E.idx_msi(E.enc(s_))[1]
    for key, pool in (("G_to_T_outside_tracts", [p for p, c in enumerate(s_) if c == "G" and p not in intr]),
                      ("G_to_T_any_guanine", [p for p, c in enumerate(s_) if c == "G"])):
        if len(pool) < k:
            continue
        t = []
        for _ in range(20):
            y = list(s_)
            for p in rng2.choice(pool, k, replace=False):
                y[p] = "T"
            t.append(np.isclose(E.idx_msi(E.enc("".join(y)))[1], w0))
        tie2[key].append(float(np.mean(t)))
# the true frame lowest against one third (frames are not exchangeable, so this
# shows only that a minimum-of-three rule beats a random pick; frame +2 beats it)
n_low = int((fv[["true_frame", "frame1", "frame2"]].values.argmin(1) == 0).sum())
out["frame_true_lowest_vs_one_third"] = dict(n=n_low, of=len(fv), binomial_p_one_sided=float(
    stats.binomtest(n_low, len(fv), 1 / 3, alternative="greater").pvalue))
# exact duplicate wild types among the 538, and mutants dropped because their nMSI
# is undefined (every trinucleotide self-reverse), per convention
out["n_distinct_wild_types_538"] = len(set(seqs))
und = {}
for conv in E.CONV:
    if conv == "scramble":
        continue
    c = 0
    for s_ in seqs:
        m = E.mutate(s_, conv, np.random.default_rng(0))
        if not E.is_canonical(m) and E.g4hunter(E.enc(m)) < E.g4hunter(E.enc(s_)) and not np.isfinite(E.idx_msi(E.enc(m))[1]):
            c += 1
    und[conv] = c
out["mutants_dropped_undefined_nMSI"] = und
out["tie_rate_T_centre_base_matched_control_pct"] = {k: [round(100 * float(np.mean(v)), 1), len(v)] for k, v in tie2.items()}
json.dump(out, open(os.path.join(RES, "si", "quoted_counts.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
