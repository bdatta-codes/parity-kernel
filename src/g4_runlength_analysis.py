#!/usr/bin/env python3
"""g4_runlength_analysis.py -- guanine-run architecture versus the parity indices.

Reproduces every number in the Results subsections "The Index Reads the Number
of Guanine-Tract Boundaries" and "Three Effects Account for the Apparent Success
of the Descriptor", and writes the data behind Figure 4 (drawn by make_fig4.py).

Inputs
  data/298_sequences_with_MSI_nMSI.xlsx   G4Hunter reference set (Bedrat et al.)
  data/a549_grich_300.txt                 A549 G-rich set (tab-separated, seq in col 2)
The four reference sequences containing U or N are excluded (n = 294) except in
the rows explicitly labelled n = 298, which use all sequences of the set.

Definitions (those the manuscript leaves implicit are marked *)
  G-run                maximal run of >= 1 G (Fig 4A, 4C, classifier profile)
  *A549 flank count    maximal runs of >= 3 G with a base on both sides
                       (838 such runs; 308 have identical flanking bases)
  *boundary words      the six trinucleotides HGG / GGH (H != G), i.e. every word
                       spanning the edge of a run of >= 2 G; their share of the
                       nMSI numerator sum|f(w) - f(rev w)| pooled over the set
  *GGG count           GGG words counted without overlap, i.e. Python
                       str.count("GGG") = sum over G-runs of floor(length/3)
                       (the overlapping count gives 4.02 vs 3.97, p = 0.8)
  consensus-negative   fails Quadparser G3+N1-7G3+N1-7G3+N1-7G3+ (98 of 294)
  *matched shuffle     mononucleotide permutation unless stated (dinucleotide =
                       Altschul-Erickson, src/shuffles.py)
  *classifier          L2 logistic regression (C = 1, standardised features,
                       class_weight = balanced) on the 98 consensus-negative
                       sequences vs 20 shuffles of each; 5-fold cross-validation
                       with folds grouped by source sequence, repeated 10 times
                       with a fresh random group-to-fold assignment; AUC of the
                       pooled out-of-fold scores, mean over repeats; 95% interval
                       = 2.5-97.5 percentiles of 2,000 bootstrap resamples of
                       source sequences (repeat-averaged out-of-fold scores).
  *feature sets        run-length profile (11): counts of G-runs of length
                         1..7 and >= 8 divided by length, longest run, number of
                         runs, mean run length
                       G4Hunter-derived (11): fraction of bases with each
                         per-base G4Hunter score -4..+4, mean score, maximum
                         25-nt window score
                       G4Hunter published two-feature form (2): mean score and
                         maximum 25-nt window score (the published window)
                       parity (3): nMSI, MSI, S1 (reverse-complement index);
                         also each alone
  *shuffled correlation Spearman rho of the per-sequence mean index over K
                       shuffles with the wild-type G4Hunter score
  *"% of shuffles beaten"  strictly greater (ties count 0); the tie-half
                       version is also written
  pooled AUC           Mann-Whitney AUC of the 300 real values against all
                       300 x 100 shuffled values (ties one half)
  partial Spearman     Spearman correlation of the residuals of rank(x),
                       rank(y) after linear regression on rank(z)
  skew                 (G - C) / (G + C)  (the sets are G-rich, so equal to |.|)

Usage:  PYTHONPATH=src python3 src/g4_runlength_analysis.py [--quick]
Outputs: results/g4/*.csv
"""
from __future__ import annotations
import argparse, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from parity import msi_scores
from shuffles import mono_shuffle, dinuc_shuffle
from experiment_538 import enc, idx_msi, g4hunter, QP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "g4")
SEED = 20261025
G = 2  # integer code of G in enc()

# ------------------------------------------------------------------ data --
def load_reference():
    d = pd.read_excel(os.path.join(ROOT, "data", "298_sequences_with_MSI_nMSI.xlsx"))
    d["seq"] = d["WT_Sequence (5'-3')"].astype(str).str.upper().str.strip()
    d["acgt"] = d["seq"].str.fullmatch("[ACGT]+")
    return d


def load_a549():
    with open(os.path.join(ROOT, "data", "a549_grich_300.txt")) as fh:
        return [l.rstrip("\n").split("\t")[1].strip().upper() for l in fh if l.strip()]

# ------------------------------------------------------------- features --
_RC = np.array([(3 - c) * 16 + (3 - b) * 4 + (3 - a)
                for a in range(4) for b in range(4) for c in range(4)])
_RCI = np.array([i for i in range(64) if i < _RC[i]]); _RCJ = _RC[_RCI]


def s1_index(x):
    """Reverse-complement trinucleotide index S1 (32 pairs, overlapping words)."""
    f = np.bincount(x[:-2].astype(np.int64) * 16 + x[1:-1] * 4 + x[2:], minlength=64)
    den = (f[_RCI] + f[_RCJ]).sum()
    return 1 - np.abs(f[_RCI] - f[_RCJ]).sum() / den if den else np.nan


def g_runs(x, kmin=1):
    g = np.r_[0, (x == G).astype(np.int8), 0]
    dg = np.diff(g)
    L = np.flatnonzero(dg == -1) - np.flatnonzero(dg == 1)
    return L[L >= kmin]


def n_ggg(x):
    """GGG words counted without overlap (as str.count('GGG')): sum floor(run/3)."""
    return int((g_runs(x, 3) // 3).sum())


def n_ggg_overlapping(x):
    g = (x == G)
    return int(np.sum(g[:-2] & g[1:-1] & g[2:]))


def runlength_profile(x):
    L = g_runs(x); n = x.size
    h = np.bincount(np.minimum(L, 8), minlength=9)[1:9] / n
    return np.r_[h, L.max() if L.size else 0, L.size, L.mean() if L.size else 0.0]


def g4h_perbase(x):
    brk = np.flatnonzero(np.diff(x)) + 1
    st, en = np.r_[0, brk], np.r_[brk, x.size]
    v = np.minimum(en - st, 4) * np.where(x[st] == G, 1, np.where(x[st] == 1, -1, 0))
    return np.repeat(v, en - st)


def g4h_window_max(pb, w=25):
    if pb.size <= w:
        return pb.mean()
    c = np.r_[0, np.cumsum(pb)]
    return ((c[w:] - c[:-w]) / w).max()


def g4h_derived(x):
    pb = g4h_perbase(x)
    hist = np.bincount(pb + 4, minlength=9) / pb.size
    return np.r_[hist, pb.mean(), g4h_window_max(pb)]


def g4h_two(x):
    pb = g4h_perbase(x)
    return np.r_[pb.mean(), g4h_window_max(pb)]


def parity3(x):
    m, n = idx_msi(x)
    return np.r_[n, m, s1_index(x)]


FEATURES = {
    "G-run-length profile": runlength_profile,
    "G4Hunter-derived (matched feature count)": g4h_derived,
    "G4Hunter published two-feature form": g4h_two,
    "parity indices (nMSI, MSI, S1)": parity3,
    "nMSI + MSI": lambda x: np.r_[idx_msi(x)],
    "nMSI alone": lambda x: np.r_[idx_msi(x)[1]],
    "MSI alone": lambda x: np.r_[idx_msi(x)[0]],
    "S1 alone": lambda x: np.r_[s1_index(x)],
}

# ----------------------------------------------------------- statistics --
def partial_spearman(x, y, z):
    xr, yr, zr = (stats.rankdata(v) for v in (x, y, z))
    D = np.c_[np.ones_like(zr), zr]
    res = lambda a: a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
    r, p = stats.spearmanr(res(xr), res(yr))   # as chromosome21_windows.partial_spearman
    return float(r), float(p)


def frac_beaten(real, shuf, tie=0.0):
    """Mean over sequences of P(real > shuffle); ties count `tie` (0 = strictly beaten)."""
    return float(np.mean([(np.sum(s < r) + tie * np.sum(s == r)) / s.size
                          for r, s in zip(real, shuf)]))


def pooled_auc(real, shuf):
    s = np.concatenate(shuf)
    y = np.r_[np.ones(len(real)), np.zeros(s.size)]
    return float(roc_auc_score(y, np.r_[real, s]))


def pooled_auc_ci(real, shuf, B=1000, seed=20261103):
    """95% interval for pooled_auc, resampling sequences (each with its own
    shuffles); its own generator, so no other result depends on it."""
    r = np.random.default_rng(seed); n = len(real); v = []
    for _ in range(B):
        ix = r.integers(0, n, n)
        v.append(pooled_auc(real[ix], [shuf[i] for i in ix]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

# ------------------------------------------------------------- sections --
rows = []


def rec(section, quantity, value, n=None, note=''):
    rows.append(dict(section=section, quantity=quantity, value=value, n=n, note=note))


def runlength_section(ref, rng, K=20):
    neg = ref[ref.acgt & ~ref.seq.apply(lambda s: bool(QP.search(s)))]
    seqs = [enc(s) for s in neg.seq]
    print(f"  consensus-negative ACGT sequences: {len(seqs)}")
    mono = [[mono_shuffle(x, rng) for _ in range(K)] for x in seqs]
    dinu = [[dinuc_shuffle(x, rng) for _ in range(K)] for x in seqs]
    # GGG count
    real_ggg = np.array([n_ggg(x) for x in seqs], float)
    shuf_ggg = np.array([np.mean([n_ggg(y) for y in S]) for S in mono])
    p = stats.wilcoxon(real_ggg, shuf_ggg).pvalue
    sec = "run-length (98 consensus-negative)"
    rec(sec, 'n sequences', len(seqs))
    rec(sec, 'mean GGG count, real', real_ggg.mean())
    rec(sec, 'mean GGG count, mono shuffles', shuf_ggg.mean())
    rec(sec, 'Wilcoxon signed-rank p (GGG real vs shuffle mean)', p)
    sd = np.array([np.mean([n_ggg(y) for y in S]) for S in dinu])
    rec(sec, 'mean GGG count, dinuc shuffles (supplementary)', sd.mean())
    ro = np.array([n_ggg_overlapping(x) for x in seqs], float)
    so = np.array([np.mean([n_ggg_overlapping(y) for y in S]) for S in mono])
    rec(sec, 'mean overlapping GGG count real / mono (alternative definition)', ro.mean(), note=f'shuffles {so.mean():.3f}, Wilcoxon p = {stats.wilcoxon(ro, so).pvalue:.2g}')
    # run-length distribution (runs of >= 1 G), pooled over sequences
    bins = np.arange(1, 9)
    hr = np.bincount(np.minimum(np.concatenate([g_runs(x) for x in seqs]), 8), minlength=9)[1:]
    hs = np.bincount(np.minimum(np.concatenate([g_runs(y) for S in mono for y in S]), 8), minlength=9)[1:]
    hd = np.bincount(np.minimum(np.concatenate([g_runs(y) for S in dinu for y in S]), 8), minlength=9)[1:]
    dist = pd.DataFrame({"run_length": [str(b) for b in bins[:-1]] + ["8+"],
                         "real_count": hr, "mono_count": hs, "dinuc_count": hd,
                         "real_pct": 100 * hr / hr.sum(), "mono_pct": 100 * hs / hs.sum(),
                         "dinuc_pct": 100 * hd / hd.sum(),
                         "real_per_seq": hr / len(seqs), "mono_per_seq": hs / (len(seqs) * K)})
    dist["real_over_mono_per_seq"] = dist.real_per_seq / dist.mono_per_seq
    dist.to_csv(os.path.join(OUT, "fig4a_runlength_distribution.csv"), index=False)
    rec(sec, '% of G-runs of length 3, real', dist.real_pct[2])
    rec(sec, '% of G-runs of length 3, mono shuffles', dist.mono_pct[2])
    rec(sec, 'real/shuffle runs of length 7 per sequence', dist.real_over_mono_per_seq[6])
    return seqs, mono, dinu


def classifier_section(seqs, mono, dinu, rng, repeats=10, nboot=2000):
    out = []
    for shuf_name, SH in [("mono", mono), ("dinuc", dinu)]:
        y = np.r_[np.ones(len(seqs)), np.zeros(sum(len(S) for S in SH))]
        grp = np.r_[np.arange(len(seqs)), np.concatenate([[i] * len(S) for i, S in enumerate(SH)])]
        allseq = list(seqs) + [z for S in SH for z in S]
        for fname, fn in FEATURES.items():
            X = np.array([fn(z) for z in allseq], float)
            X = np.nan_to_num(X, nan=np.nanmean(X))
            oof_sum = np.zeros(y.size); aucs = []
            for r in range(repeats):
                fold_of_group = rng.permutation(np.arange(len(seqs)) % 5)
                fold = fold_of_group[grp]
                oof = np.zeros(y.size)
                for k in range(5):
                    tr, te = fold != k, fold == k
                    clf = make_pipeline(StandardScaler(), LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=2000))
                    clf.fit(X[tr], y[tr])
                    oof[te] = clf.decision_function(X[te])
                aucs.append(roc_auc_score(y, oof)); oof_sum += oof
            oof_mean = oof_sum / repeats
            # bootstrap over source sequences
            idx_by_g = [np.flatnonzero(grp == g) for g in range(len(seqs))]
            bs = []
            for _ in range(nboot):
                gs = rng.integers(0, len(seqs), len(seqs))
                ii = np.concatenate([idx_by_g[g] for g in gs])
                bs.append(roc_auc_score(y[ii], oof_mean[ii]))
            out.append(dict(shuffle=shuf_name, feature_set=fname, n_features=X.shape[1],
                            auc_mean_over_repeats=np.mean(aucs), auc_sd_over_repeats=np.std(aucs),
                            auc_repeat_averaged=roc_auc_score(y, oof_mean),
                            ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
            print(f"    {shuf_name:5s} {fname:42s} AUC {np.mean(aucs):.3f} "
                  f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "fig4b_classifier_auc.csv"), index=False)
    sec = "classifier (98 consensus-negative, 20 shuffles each)"
    get = lambda s, f: df[(df.shuffle == s) & (df.feature_set == f)].iloc[0]
    rec(sec, 'AUC run-length profile vs mono', get('mono', 'G-run-length profile').auc_mean_over_repeats)
    rec(sec, 'AUC run-length profile vs dinuc', get('dinuc', 'G-run-length profile').auc_mean_over_repeats)
    rec(sec, 'AUC G4Hunter-derived (11 features) vs dinuc', get('dinuc', 'G4Hunter-derived (matched feature count)').auc_mean_over_repeats)
    rec(sec, 'AUC G4Hunter two-feature vs dinuc', get('dinuc', 'G4Hunter published two-feature form').auc_mean_over_repeats)
    for f in ["parity indices (nMSI, MSI, S1)", "nMSI + MSI", "nMSI alone", "MSI alone", "S1 alone"]:
        r = get("dinuc", f)
        rec(sec, f'AUC {f} vs dinuc', r.auc_mean_over_repeats)
        rec(sec, f'95% upper bound {f} vs dinuc', r.ci_hi)
    return df


def a549_section(a549, rng, K=100):
    sec = "A549 G-rich (n = 300)"
    # flanks of interior runs of >= 3 G
    tot = same = 0
    for s in a549:
        for m in re.finditer("G{3,}", s):
            if m.start() > 0 and m.end() < len(s):
                tot += 1; same += s[m.start() - 1] == s[m.end()]
    rec(sec, 'interior G>=3 runs', tot)
    rec(sec, '... with identical flanking bases', same)
    rec(sec, '% identical flanks', 100 * same / tot)
    # share of the nMSI numerator from boundary words HGG / GGH
    from experiment_538 import _PI, _PJ
    bnd = np.array([a * 16 + G * 4 + G for a in (0, 1, 3)] + [G * 16 + G * 4 + b for b in (0, 1, 3)])
    num = numb = 0.0
    for s in a549:
        x = enc(s).astype(np.int64)
        f = np.bincount(x[:-2] * 16 + x[1:-1] * 4 + x[2:], minlength=64)
        d = np.abs(f[_PI] - f[_PJ]); num += d.sum()
        numb += d[np.isin(_PI, bnd) | np.isin(_PJ, bnd)].sum()
    rec(sec, '% of nMSI numerator from boundary words', 100 * numb / num)

    seqs = [enc(s) for s in a549]
    real = np.array([idx_msi(x) for x in seqs])            # (MSI, nMSI)
    mono = [[mono_shuffle(x, rng) for _ in range(K)] for x in seqs]
    dinu = [[dinuc_shuffle(x, rng) for _ in range(K)] for x in seqs]
    I_m = [np.array([idx_msi(y) for y in S]) for S in mono]
    I_d = [np.array([idx_msi(y) for y in S]) for S in dinu]
    nr = np.array([g_runs(x).size for x in seqs]); mr = np.array([g_runs(x).mean() for x in seqs])
    ns = np.array([np.mean([g_runs(y).size for y in S]) for S in mono])
    ms = np.array([np.mean([g_runs(y).mean() for y in S]) for S in mono])
    dn = real[:, 1] - np.array([np.nanmean(I[:, 1]) for I in I_m])
    dm = real[:, 0] - np.array([np.nanmean(I[:, 0]) for I in I_m])
    pd.DataFrame(dict(seq=a549, length=[len(s) for s in a549], nMSI=real[:, 1], MSI=real[:, 0],
                      n_runs=nr, mean_run=mr, d_nMSI=dn, d_MSI=dm, d_runs=nr - ns,
                      d_mean_run=mr - ms)).to_csv(os.path.join(OUT, "a549_grich300_deltas.csv"), index=False)
    rec(sec, 'mean delta number of G-runs (real - shuffles)', np.mean(nr - ns))
    rec(sec, 'mean delta mean G-run length', np.mean(mr - ms))
    r, p = stats.spearmanr(dn, nr - ns)
    rec(sec, 'Spearman rho(d nMSI, d runs)', r); rec(sec, 'p', p)
    rec(sec, 'Spearman rho(d nMSI, d mean run length)', stats.spearmanr(dn, mr - ms)[0])
    for j, name in [(1, "nMSI"), (0, "MSI")]:
        sh = [I[:, j] for I in I_m]
        rec(sec, f'pooled AUC real vs own mono shuffles, {name}', pooled_auc(real[:, j], sh))
        rec(sec, f'% of own shuffles strictly beaten, {name}', 100 * frac_beaten(real[:, j], sh))
        rec(sec, f'% of own shuffles beaten, ties one half, {name}', 100 * frac_beaten(real[:, j], sh, 0.5))
        rec(sec, f'paired Wilcoxon p (real vs shuffle mean), {name}', stats.wilcoxon(real[:, j], [s.mean() for s in sh]).pvalue)
    for j, name in [(1, "nMSI"), (0, "MSI")]:
        sh = [I[:, j] for I in I_d]
        rec(sec, f'pooled AUC real vs own dinuc shuffles, {name}', pooled_auc(real[:, j], sh))
        lo, hi = pooled_auc_ci(real[:, j], sh)
        rec(sec, f'pooled AUC real vs own dinuc shuffles, {name}, 95% CI low', lo)
        rec(sec, f'pooled AUC real vs own dinuc shuffles, {name}, 95% CI high', hi)
        rec(sec, f'% of own dinuc shuffles strictly beaten, {name}', 100 * frac_beaten(real[:, j], sh))
        rec(sec, f'% of own dinuc shuffles beaten, ties one half, {name}', 100 * frac_beaten(real[:, j], sh, 0.5))
        rec(sec, f'paired Wilcoxon p (real vs dinuc shuffle mean), {name}', stats.wilcoxon(real[:, j], [np.nanmean(x) for x in sh]).pvalue)
    for j, name in [(1, "nMSI"), (0, "MSI")]:
        sh = [I[:, j] for I in I_m]
        lo, hi = pooled_auc_ci(real[:, j], sh)
        rec(sec, f'pooled AUC real vs own mono shuffles, {name}, 95% CI low', lo)
        rec(sec, f'pooled AUC real vs own mono shuffles, {name}, 95% CI high', hi)


def three_effects(ref, a549, rng, Km=1000, Kd=200):
    sec = "three effects"
    # all 298 sequences (index computed from the strings, any letters)
    full = ref.copy()
    sc = np.array([msi_scores(s) for s in full.seq], float)
    g298 = full["WT_G4H Score"].to_numpy(float)
    sk298 = np.array([(s.count("G") - s.count("C")) / (s.count("G") + s.count("C")) for s in full.seq])
    gc298 = np.array([(s.count("G") + s.count("C")) / len(s) for s in full.seq])
    rec(sec, 'rho nMSI vs G4Hunter, reference', stats.spearmanr(sc[:, 1], g298)[0], 298)
    rec(sec, 'rho MSI vs G4Hunter, reference', stats.spearmanr(sc[:, 0], g298)[0], 298)
    rec(sec, 'rho MSI vs GC, reference', stats.spearmanr(sc[:, 0], gc298)[0], 298)
    rec(sec, 'rho nMSI vs GC, reference', stats.spearmanr(sc[:, 1], gc298)[0], 298)
    r, p = stats.spearmanr(sc[:, 1], sk298)
    rec(sec, 'rho nMSI vs G-C skew, reference', r, 298); rec(sec, 'p', p, 298)
    rec(sec, 'partial rho nMSI~G4Hunter | skew', partial_spearman(sc[:, 1], g298, sk298)[0], 298)
    rec(sec, 'partial rho nMSI~G4Hunter | GC', partial_spearman(sc[:, 1], g298, gc298)[0], 298)
    # string-level mono shuffles on all 298 sequences
    ms298 = np.array([np.mean([msi_scores("".join(rng.permutation(list(s))))[1]
                               for _ in range(200)]) for s in full.seq])
    rec(sec, 'rho mean mono-shuffled nMSI vs WT G4Hunter, reference', stats.spearmanr(ms298, g298)[0], 298, '200 shuffles, string permutation')

    # n = 294 (and A549 n = 300)
    sub = ref[ref.acgt]
    sets = {"reference": ([enc(s) for s in sub.seq], sub["WT_G4H Score"].to_numpy(float)),
            "A549": ([enc(s) for s in a549], None)}
    per = []
    for name, (seqs, gh) in sets.items():
        n = len(seqs)
        if gh is None:
            gh = np.array([g4hunter(x) for x in seqs])
        wt = np.array([idx_msi(x) for x in seqs])
        mm = np.array([np.nanmean([idx_msi(mono_shuffle(x, rng))[1] for _ in range(Km)]) for x in seqs])
        dd = np.array([np.nanmean([idx_msi(dinuc_shuffle(x, rng))[1] for _ in range(Kd)]) for x in seqs])
        r_wt = stats.spearmanr(wt[:, 1], gh)[0]
        rec(sec, f'rho nMSI vs G4Hunter, {name}', r_wt, n)
        if name == "reference":
            rec(sec, 'rho MSI vs G4Hunter, reference', stats.spearmanr(wt[:, 0], gh)[0], n)
        else:
            rec(sec, 'rho MSI vs G4Hunter, A549', stats.spearmanr(wt[:, 0], gh)[0], n)
        r_m = stats.spearmanr(mm, gh)[0]; r_d = stats.spearmanr(dd, gh)[0]
        rec(sec, f'rho mean mono-shuffled nMSI vs WT G4Hunter, {name}', r_m, n, f'{Km} shuffles')
        rec(sec, f'rho mean dinuc-shuffled nMSI vs WT G4Hunter, {name}', r_d, n, f'{Kd} shuffles')
        rec(sec, f'share of association kept by mono shuffles, {name}', r_m / r_wt, n)
        rec(sec, f'share of association kept by dinuc shuffles, {name}', r_d / r_wt, n)
        if name == "reference":
            sq = list(sub.seq)
            gc = np.array([(s.count("G") + s.count("C")) / len(s) for s in sq])
            sk = np.array([(s.count("G") - s.count("C")) / (s.count("G") + s.count("C")) for s in sq])
            rec(sec, 'rho MSI vs GC, reference', stats.spearmanr(wt[:, 0], gc)[0], n)
            rec(sec, 'rho nMSI vs GC, reference', stats.spearmanr(wt[:, 1], gc)[0], n)
            rec(sec, 'p (MSI vs GC)', stats.spearmanr(wt[:, 0], gc)[1], n)
            rec(sec, 'p (nMSI vs GC)', stats.spearmanr(wt[:, 1], gc)[1], n)
            r, p = stats.spearmanr(wt[:, 1], sk)
            rec(sec, 'rho nMSI vs G-C skew, reference', r, n)
            rec(sec, 'p', p, n)
            rec(sec, 'partial rho nMSI~G4Hunter | skew', partial_spearman(wt[:, 1], gh, sk)[0], n)
            rec(sec, 'partial rho nMSI~G4Hunter | GC', partial_spearman(wt[:, 1], gh, gc)[0], n)
        per.append(pd.DataFrame(dict(set=name, nMSI=wt[:, 1], MSI=wt[:, 0], G4Hunter=gh,
                                     mono_mean_nMSI=mm, dinuc_mean_nMSI=dd)))
    pd.concat(per).to_csv(os.path.join(OUT, "three_effects_per_sequence.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer shuffles/repeats (smoke test)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    ref, a549 = load_reference(), load_a549()
    rng = np.random.default_rng(SEED)
    print("[1/4] run-length section"); seqs, mono, dinu = runlength_section(ref, rng)
    print("[2/4] classifier"); classifier_section(seqs, mono, dinu, rng,
                                                 repeats=2 if a.quick else 10, nboot=200 if a.quick else 2000)
    print("[3/4] A549"); a549_section(a549, rng, K=100)
    print("[4/4] three effects"); three_effects(ref, a549, rng, Km=100 if a.quick else 1000,
                                                Kd=50 if a.quick else 200)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summary_runlength_three_effects.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.max_colwidth", 60):
        print(df[["section", "quantity", "n", "value"]].to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
