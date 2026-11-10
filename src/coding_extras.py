#!/usr/bin/env python3
"""
coding_extras.py
================

The coding-sequence results of the paper that are not Figure 5A, computed from
the same sets that coding_sequence_analysis.build_sets_genbank() builds
(1,076 coding sequences, 2,952 non-coding pieces):

  fig5b_codon_structure      Figure 5B: the frame-0 index of real CDS, after
                             whole-codon permutation, after shuffling bases
                             within each codon position, after a mononucleotide
                             shuffle; codon-permutation invariance; and the
                             fraction of the depression recovered.
  frame_identification       the share of coding sequences whose true frame gives
                             the lowest of the three frame values, by ORF length.
  fickett_vs_symmetry        Spearman rho between Fickett's statistic and the
                             frame-0 index, and cross-validated logistic AUCs
                             (Fickett alone / symmetry alone / both).

DEFINITIONS (as stated in the Methods)
--------------------------------------
Figure 5B
  * Sequences: all 1,076 coding sequences; the codon-permutation identity is
    checked on all 1,076.
  * Each control value is the mean over sequences of the mean over
    N_SHUF_5B (=20) independent shuffles of that sequence.
  * Total depression = mononucleotide-shuffled value minus real value, i.e.
    the reference is the SAME sequences with all arrangement destroyed but
    composition held exactly (the right-hand bar of Figure 5B). The controls are
    then nested (codon permutation > within-position > mononucleotide).
    The alternative reference (frame-0 index of exact-length non-coding windows,
    one window per CDS, drawn as in Figure 5A) is reported alongside.
  * recovered fraction = (within-position - real) / (mono - real). The
    within-position shuffle destroys codon identity but keeps the per-position
    base composition, so the RECOVERED fraction is the part of the depression
    due to codon identity beyond per-position composition; the REMAINING
    fraction (mono - within) / (mono - real) is carried by per-position
    composition, which is what Fickett's statistic reads.
  * 95% interval: bootstrap over sequences (2,000 resamples).

Frame identification
  * frame k value = nmsi_frame0(s[k:]), k = 0, 1, 2; the counts of each frame
    giving the lowest value are reported per Figure 5A ORF-length bin (bins with
    < 10 sequences skipped, as in Figure 5A). The frames are not exchangeable
    (frame +2 triplets hold the first two positions of every codon), so one
    third is not a chance level; sign tests of frame 0 against each shifted
    frame are in quoted_counts.py.

Fickett vs symmetry
  * 660 coding sequences = a random subset (seeded) of the 1,076; each paired
    with one exact-length non-coding window (_window_matched_negatives).
    The Methods state that the 660 are a seeded random subset; the full 1,076
    and the 660 shortest are reported as sensitivity rows.
  * rho = Spearman between Fickett's statistic and the frame-0 index over all
    1,320 sequences pooled across the two classes (so it includes the
    between-class separation).
  * Symmetry features = frame-0 nMSI and frame-free nMSI (primary); the three
    frame values plus frame-free, and the minimum over frames plus frame-free,
    as alternatives.
  * Logistic regression (sklearn, L2, C=1) on standardised features (scaler
    fitted within each training fold), five-fold stratified CV with shuffling
    (seeded). AUC = mean of the five fold AUCs; SD = SD of the fold AUCs
    (ddof=1); gain = mean fold AUC(both) - mean fold AUC(Fickett); the SD of
    the per-fold gain is also given.

License: MIT
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coding_comparators import fickett_asymmetry  # noqa: E402
import coding_sequence_analysis as csa  # noqa: E402

N_SHUF_5B = 20
N_BOOT = 2000

SEED_5B = 5002
SEED_FRAME = 5003
SEED_LOGIT = 5004


def mono_shuffle_str(s: str, rng: np.random.Generator) -> str:
    a = np.frombuffer(s.encode(), dtype=np.uint8).copy()
    rng.shuffle(a)
    return a.tobytes().decode()


# --------------------------------------------------------------------------- #
# Figure 5B
# --------------------------------------------------------------------------- #
def fig5b_codon_structure(coding, noncoding, seed=SEED_5B, n_shuf=N_SHUF_5B,
                          n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(coding)
    real = np.empty(n); perm = np.empty(n); pos = np.empty(n); mono = np.empty(n)
    maxd = np.zeros(n)
    for i, s in enumerate(coding):
        real[i] = csa.nmsi_frame0(s)
        p = [csa.nmsi_frame0(csa.permute_codons(s, rng)) for _ in range(n_shuf)]
        maxd[i] = np.max(np.abs(np.array(p) - real[i]))
        perm[i] = np.mean(p)
        pos[i] = np.mean([csa.nmsi_frame0(csa.shuffle_within_codon_position(s, rng))
                          for _ in range(n_shuf)])
        mono[i] = np.mean([csa.nmsi_frame0(mono_shuffle_str(s, rng))
                           for _ in range(n_shuf)])
    neg = csa._window_matched_negatives([len(s) for s in coding], noncoding, rng)
    nc = np.array([csa.nmsi_frame0(x) for x in neg])

    def rec(idx, top):
        r, w, t = real[idx].mean(), pos[idx].mean(), top[idx].mean()
        return (w - r) / (t - r)

    allidx = np.arange(n)
    boot_m, boot_n = [], []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        boot_m.append(rec(b, mono)); boot_n.append(rec(b, nc))
    rm, rn = rec(allidx, mono), rec(allidx, nc)
    row = dict(
        n_sequences=n, n_shuffles_per_sequence=n_shuf,
        mean_real=real.mean(), mean_codon_permuted=perm.mean(),
        mean_within_position_shuffled=pos.mean(),
        mean_mononucleotide_shuffled=mono.mean(),
        mean_noncoding_window=nc.mean(),
        max_abs_delta_codon_permutation_all=float(maxd.max()),
        # reference = mononucleotide shuffle of the same sequences (primary)
        recovered_by_within_position_frac=rm,
        recovered_by_within_position_lo=float(np.percentile(boot_m, 2.5)),
        recovered_by_within_position_hi=float(np.percentile(boot_m, 97.5)),
        codon_identity_beyond_position_frac=rm,
        per_position_composition_frac=1 - rm,
        # reference = exact-length non-coding windows (alternative)
        recovered_vs_noncoding_frac=rn,
        recovered_vs_noncoding_lo=float(np.percentile(boot_n, 2.5)),
        recovered_vs_noncoding_hi=float(np.percentile(boot_n, 97.5)),
    )
    return row


# --------------------------------------------------------------------------- #
# Frame identification
# --------------------------------------------------------------------------- #
def frame_values(s):
    return [csa.nmsi_frame0(s[k:]) for k in range(3)]


def frame_identification(coding, bins=None):
    """Which of the three frames gives the lowest frame-resolved index. Frame k
    reads triplets starting at codon position k + 1, so frame +2 triplets hold
    the third position of one codon and the first two of the next. The three
    frames are not exchangeable, so one third is not a chance level; the counts
    of each frame being lowest are reported, with sign tests of the true frame
    against each shifted frame computed in quoted_counts.py from frame_values.csv."""
    bins = bins or csa.DEFAULT_BINS
    F = np.array([frame_values(s) for s in coding])
    C = np.array([len(s) // 3 for s in coding])
    am = np.argmin(F, axis=1)
    tied = np.isclose(np.sort(F, 1)[:, 0], np.sort(F, 1)[:, 1])
    rows = []
    def row(lab, m):
        return dict(orf_codon_bin=lab, n_coding=int(m.sum()),
                    n_true_frame_lowest=int((am[m] == 0).sum()), n_frame1_lowest=int((am[m] == 1).sum()),
                    n_frame2_lowest=int((am[m] == 2).sum()), n_lowest_tied=int(tied[m].sum()),
                    pct_true_frame_lowest=round(100 * (am[m] == 0).mean(), 1),
                    pct_frame1_lowest=round(100 * (am[m] == 1).mean(), 1),
                    pct_frame2_lowest=round(100 * (am[m] == 2).mean(), 1),
                    mean_true=round(float(F[m, 0].mean()), 4), mean_frame1=round(float(F[m, 1].mean()), 4),
                    mean_frame2=round(float(F[m, 2].mean()), 4))
    for lo, hi in bins:
        m = (C >= lo) & (C < hi)
        if m.sum() < 10:
            continue
        rows.append(row(f"{lo}-{hi if hi < 100000 else '+'}", m))
    rows.append(row("all", np.ones(len(coding), bool)))
    return rows, F


# --------------------------------------------------------------------------- #
# Fickett vs symmetry
# --------------------------------------------------------------------------- #
def _features(s):
    f = frame_values(s)
    return dict(fickett=fickett_asymmetry(s), f0=f[0], f1=f[1], f2=f[2],
                fmin=min(f), framefree=csa.nmsi_framefree(s))


FEATURE_SETS = {
    "fickett": ["fickett"],
    "sym_f0_framefree": ["f0", "framefree"],
    "both_f0_framefree": ["fickett", "f0", "framefree"],
    "sym_3frames_framefree": ["f0", "f1", "f2", "framefree"],
    "both_3frames_framefree": ["fickett", "f0", "f1", "f2", "framefree"],
    "sym_fmin_framefree": ["fmin", "framefree"],
    "both_fmin_framefree": ["fickett", "fmin", "framefree"],
    "sym_f0_only": ["f0"],
    "both_f0_only": ["fickett", "f0"],
}


def _cv_fold_aucs(X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = []
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(X[tr], y[tr])
        out.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(out)


def fickett_vs_symmetry(coding, noncoding, n=660, seed=SEED_LOGIT):
    """Returns a list of rows: one per (selection, feature set)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    C = np.array([len(s) for s in coding])
    selections = {
        f"random{n}": np.sort(rng.choice(len(coding), n, replace=False)),
        f"shortest{n}": np.argsort(C, kind="stable")[:n],
        f"all{len(coding)}": np.arange(len(coding)),
    }
    rows = []
    for sel_name, idx in selections.items():
        pos = [coding[i] for i in idx]
        neg = csa._window_matched_negatives([len(s) for s in pos], noncoding, rng)
        feats = [_features(s) for s in pos + neg]
        y = np.array([1] * len(pos) + [0] * len(neg))
        fk = np.array([f["fickett"] for f in feats])
        f0 = np.array([f["f0"] for f in feats])
        rho = spearmanr(fk, f0).correlation
        rho_ff = spearmanr(fk, [f["framefree"] for f in feats]).correlation
        aucs = {}
        for name, cols in FEATURE_SETS.items():
            X = np.array([[f[c] for c in cols] for f in feats])
            aucs[name] = _cv_fold_aucs(X, y, seed)
        for name, a in aucs.items():
            row = dict(selection=sel_name, n_coding=len(pos), n_noncoding=len(neg),
                       spearman_fickett_vs_frame0=round(rho, 3),
                       spearman_fickett_vs_framefree=round(rho_ff, 3),
                       feature_set=name, features="+".join(FEATURE_SETS[name]),
                       auc_cv_mean=round(a.mean(), 4), auc_fold_sd=round(a.std(ddof=1), 4))
            if name.startswith("both"):
                g = a - aucs["fickett"]
                row["gain_over_fickett"] = round(g.mean(), 4)
                row["gain_fold_sd"] = round(g.std(ddof=1), 4)
            rows.append(row)
    return rows


