#!/usr/bin/env python3
"""kernel_decomposition.py - where the separating signal lives, relative to what
nMSI reads.

nMSI depends on the trinucleotide count vector only through the 24 signed
differences f(w) - f(rev w) of the mirror pairs and the total of the non-self-
reverse words. The complementary information, which its numerator discards,
is the 16 self-reverse word counts and the 24 pair sums f(w) + f(rev w). If the
kernel account is right, a classifier given the discarded part should separate
real quadruplex-forming sequence from its shuffles, and one given the signed
differences should not.

Data and protocol are those of Figure 4B: the 98 consensus-negative reference
sequences with their 20 mononucleotide and 20 dinucleotide shuffles each,
regenerated exactly by g4_runlength_analysis.runlength_section (same seed);
L2 logistic regression (C = 1, balanced weights) on standardised features,
five folds grouped by source sequence, ten repeats, AUC averaged over repeats,
2,000-resample bootstrap over source sequences. The same is repeated on the
538 wild types of the mutagenesis experiment against 20 dinucleotide shuffles
each. Counts are per word counted (divided by N, all 64 words), so the
non-self-reverse fraction is the normalised denominator of nMSI; the set
"signed pair differences + non-self-reverse fraction" carries everything nMSI
is computed from (though a linear model cannot form |d|). Seeds: shuffles as
Figure 4B; folds and bootstrap 20261108. Writes
results/additional/kernel_decomposition.csv. With --chr21 FASTA, the same
classifiers are also run on a non-quadruplex control: chr21 fragments matched
to each of the 538 wild types in length and GC (within 0.03), with no
permissive quadruplex motif ((G2+ N1-7)3 G2+) and no two G3+ tracts within
7 nt on either strand, against 20 dinucleotide shuffles
each (seeds 20261116, 20261117); writes kernel_decomposition_control.csv.

License: MIT
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g4_runlength_analysis as R
import experiment_538 as E
from shuffles import dinuc_shuffle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "additional"); os.makedirs(OUT, exist_ok=True)
SELF = np.array([i for i in range(64) if i // 16 == i % 4])
PI, PJ = E._PI, E._PJ


def counts(x):
    x = np.asarray(x, np.int64)
    f = np.bincount(x[:-2] * 16 + x[1:-1] * 4 + x[2:], minlength=64).astype(float)
    return f / max(f.sum(), 1.0)


FEATURES = {
    "nMSI alone": lambda x: np.r_[E.idx_msi(x)[1]],
    "non-self-reverse fraction (1)": lambda x: (lambda f: np.r_[f[PI].sum() + f[PJ].sum()])(counts(x)),
    "signed pair differences (24)": lambda x: (lambda f: f[PI] - f[PJ])(counts(x)),
    "signed pair differences + non-self-reverse fraction (25)": lambda x: (lambda f: np.r_[f[PI] - f[PJ], f[PI].sum() + f[PJ].sum()])(counts(x)),
    "absolute pair differences (24)": lambda x: (lambda f: np.abs(f[PI] - f[PJ]))(counts(x)),
    "self-reverse word counts (16)": lambda x: counts(x)[SELF],
    "pair sums (24)": lambda x: (lambda f: f[PI] + f[PJ])(counts(x)),
    "self-reverse counts + pair sums (40)": lambda x: (lambda f: np.r_[f[SELF], f[PI] + f[PJ]])(counts(x)),
    "all 64 word counts": counts,
}


def classify(seqs, SH, rng, repeats=10, nboot=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(seqs)), np.zeros(sum(len(S) for S in SH))]
    grp = np.r_[np.arange(len(seqs)), np.concatenate([[i] * len(S) for i, S in enumerate(SH)])]
    allseq = list(seqs) + [z for S in SH for z in S]
    res = []
    for fname, fn in FEATURES.items():
        X = np.array([fn(z) for z in allseq], float)
        X = np.nan_to_num(X, nan=np.nanmean(X))
        oof_sum = np.zeros(y.size); aucs = []
        for r in range(repeats):
            fold = rng.permutation(np.arange(len(seqs)) % 5)[grp]
            oof = np.zeros(y.size)
            for k in range(5):
                tr, te = fold != k, fold == k
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
                clf.fit(X[tr], y[tr]); oof[te] = clf.decision_function(X[te])
            aucs.append(roc_auc_score(y, oof)); oof_sum += oof
        oof_mean = oof_sum / repeats
        idx_by_g = [np.flatnonzero(grp == g) for g in range(len(seqs))]
        bs = []
        for _ in range(nboot):
            gs = rng.integers(0, len(seqs), len(seqs))
            ii = np.concatenate([idx_by_g[g] for g in gs]); bs.append(roc_auc_score(y[ii], oof_mean[ii]))
        res.append(dict(feature_set=fname, n_features=X.shape[1], auc=float(np.mean(aucs)),
                        ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5))))
        print(f"    {fname:40s} AUC {np.mean(aucs):.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]", flush=True)
    return res


def control_fragments(seqs538, chr21, rng):
    """chr21 fragments matched to each wild type in length and GC (within 0.03),
    with no permissive quadruplex motif ((G2+ N1-7)3 G2+) and no two G3+ tracts
    within 7 nt (or the complements) on
    either strand; random strand."""
    import re, gzip
    op = gzip.open if chr21.endswith(".gz") else open
    s = "".join(l.strip() for l in op(chr21, "rt") if not l.startswith(">")).upper()
    comp = str.maketrans("ACGT", "TGCA"); out = []
    for w in seqs538:
        L = len(w); gc = (w.count("G") + w.count("C")) / L
        for _ in range(200000):
            p = int(rng.integers(0, len(s) - L)); f = s[p:p + L]
            if set(f) - set("ACGT"):
                continue
            if rng.random() < 0.5:
                f = f[::-1].translate(comp)
            if abs((f.count("G") + f.count("C")) / L - gc) > 0.03:
                continue
            if (re.search("G{3,}.{1,7}G{3,}", f) or re.search("C{3,}.{1,7}C{3,}", f)
                    or re.search("(?:G{2,}.{1,7}){3}G{2,}", f) or re.search("(?:C{2,}.{1,7}){3}C{2,}", f)):
                continue
            out.append(f); break
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--chr21", help="chr21 FASTA; adds the non-quadruplex control set")
    a = ap.parse_args()
    ref = R.load_reference()
    rng0 = np.random.default_rng(R.SEED)                 # identical shuffles to Figure 4B
    seqs, mono, dinu = R.runlength_section(ref, rng0)
    rng = np.random.default_rng(20261108)
    rows = []
    for name, SH in (("mono", mono), ("dinuc", dinu)):
        print(f"[98 consensus-negative, {name}]")
        for r in classify(seqs, SH, rng):
            rows.append(dict(set="98 consensus-negative reference", shuffle=name, **r))
    print("[538 mutagenesis wild types, dinuc]")
    s538, _ = E.load(); X538 = [E.enc(s) for s in s538]
    rs = np.random.default_rng(20261109)
    SH = [[dinuc_shuffle(x, rs) for _ in range(20)] for x in X538]
    for r in classify(X538, SH, rng):
        rows.append(dict(set="538 mutagenesis wild types", shuffle="dinuc", **r))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "kernel_decomposition.csv"), index=False)
    if a.chr21:
        print("[chr21 control fragments matched to the 538 wild types, dinuc]")
        rc = np.random.default_rng(20261116)
        F = control_fragments(s538, a.chr21, rc); XF = [E.enc(f) for f in F]
        SHF = [[dinuc_shuffle(x, rc) for _ in range(20)] for x in XF]
        crow = [dict(set=f"chr21 control fragments (n = {len(F)})", shuffle="dinuc", **r)
                for r in classify(XF, SHF, np.random.default_rng(20261117))]
        pd.DataFrame(crow).to_csv(os.path.join(OUT, "kernel_decomposition_control.csv"), index=False)


if __name__ == "__main__":
    main()
