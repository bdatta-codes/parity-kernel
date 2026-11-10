#!/usr/bin/env python3
"""composition_share.py -- how much of the index-propensity correlation composition reproduces.

The ratio rho(mean of K shuffles, G4Hunter) / rho(real index, G4Hunter) compares a
noise-free composition expectation with a single noisy value and so overstates the
share (it can exceed one). Two like-for-like measures are reported instead, for the
294 ACGT reference sequences and the 23,040 A549 motifs, for nMSI and MSI:

  single-shuffle rho   Spearman rho of ONE shuffle of each sequence with the wild-type
                       G4Hunter score, over 200 replicate draws (mean and 2.5-97.5%
                       range), as a share of the real rho: what a sequence of the same
                       composition (mono) or dinucleotide composition (dinuc) shows
  bootstrap interval   of the share, resampling sequences with their shuffles (200
                       resamples of 20 replicate draws; seed 20261130)
  variance-matched     the share recomputed after rescaling each shuffle's deviation
                       from its sequence's shuffle mean to the dispersion of the real
                       index about that mean (a shuffle is less dispersed, which
                       inflates the plain share)
  partial rho          rho(real index, G4Hunter | composition expectation), the
                       composition expectation being the mean index over the shuffles
                       of the same sequence (nMSI: the 1,000 mononucleotide / 200
                       dinucleotide means of g4_runlength_analysis.py and
                       a549_full_set.py; MSI: 200 / 100 shuffles drawn here)

Usage:  python3 src/composition_share.py
Writes results/g4/composition_share.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from experiment_538 import enc, idx_msi
from shuffles import mono_shuffle, dinuc_shuffle
from g4_runlength_analysis import load_reference, partial_spearman

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_REP = 200


def run(name, seqs, gh, nmsi_mono_mean, nmsi_dinuc_mean, rng):
    X = [enc(s) for s in seqs]
    real = np.array([idx_msi(x) for x in X])                    # MSI, nMSI
    msi_mm = np.array([np.nanmean([idx_msi(mono_shuffle(x, rng))[0] for _ in range(200)]) for x in X])
    msi_dm = np.array([np.nanmean([idx_msi(dinuc_shuffle(x, rng))[0] for _ in range(100)]) for x in X])
    rows = []
    for j, idx, mm, dm in ((1, "nMSI", nmsi_mono_mean, nmsi_dinuc_mean), (0, "MSI", msi_mm, msi_dm)):
        r_real = stats.spearmanr(real[:, j], gh)[0]
        row = dict(set=name, n=len(seqs), index=idx, rho_real=r_real,
                   rho_mean_mono=stats.spearmanr(mm, gh)[0], rho_mean_dinuc=stats.spearmanr(dm, gh)[0])
        for kind, f in (("mono", mono_shuffle), ("dinuc", dinuc_shuffle)):
            V = np.array([[idx_msi(f(x, rng))[j] for x in X] for _ in range(R_REP)])   # (R_REP, n)
            reps = np.array([stats.spearmanr(v[np.isfinite(v)], gh[np.isfinite(v)])[0] for v in V])
            share = reps.mean() / r_real
            # bootstrap over sequences (resampling each sequence with its shuffles),
            # 200 resamples, 20 replicate shuffles each
            rb = np.random.default_rng(20261130); bs = []
            for _ in range(200):
                ix = rb.integers(0, len(X), len(X)); rr = stats.spearmanr(real[ix, j], gh[ix])[0]
                sub = V[rb.choice(R_REP, 20, replace=False)][:, ix]
                bs.append(np.mean([stats.spearmanr(np.nan_to_num(v, nan=np.nanmean(v)), gh[ix])[0] for v in sub]) / rr)
            # variance-matched surrogate: a shuffle is less dispersed about the composition
            # expectation than the real index is, so rescale each shuffle's deviation from
            # its sequence mean to the real index's dispersion about that mean
            mu = np.nanmean(V, 0); s_within = np.sqrt(np.nanmean(np.nanvar(V, 0)))
            s_real = np.nanstd(real[:, j] - mu); Vm = mu + (V - mu) * (s_real / s_within)
            reps_m = np.array([stats.spearmanr(v[np.isfinite(v)], gh[np.isfinite(v)])[0] for v in Vm])
            row.update({f"rho_single_{kind}": reps.mean(), f"rho_single_{kind}_lo": np.percentile(reps, 2.5),
                        f"rho_single_{kind}_hi": np.percentile(reps, 97.5),
                        f"share_single_{kind}": share,
                        f"share_single_{kind}_boot_lo": np.percentile(bs, 2.5),
                        f"share_single_{kind}_boot_hi": np.percentile(bs, 97.5),
                        f"dispersion_ratio_{kind}": s_real / s_within,
                        f"share_variance_matched_{kind}": reps_m.mean() / r_real})
        for kind, m in (("mono", mm), ("dinuc", dm)):
            pr, pp = partial_spearman(real[:, j], gh, m)
            row.update({f"partial_rho_{kind}": pr, f"partial_p_{kind}": pp})
        rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return rows


def main():
    rng = np.random.default_rng(20261031)
    ref = load_reference(); sub = ref[ref.acgt]
    te = pd.read_csv(os.path.join(ROOT, "results", "g4", "three_effects_per_sequence.csv"))
    tr = te[te.set == "reference"]
    rows = run("G4Hunter reference set", list(sub.seq), sub["WT_G4H Score"].to_numpy(float),
               tr.mono_mean_nMSI.to_numpy(), tr.dinuc_mean_nMSI.to_numpy(), rng)
    a = pd.read_csv(os.path.join(ROOT, "results", "a549_full", "per_sequence.csv"))
    rows += run("A549 motifs", list(a.seq), a.G4Hunter.to_numpy(float),
                a.mono_mean_nMSI_K1000.to_numpy(), a.dinuc_mean_nMSI_K200.to_numpy(), rng)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "g4", "composition_share.csv"), index=False)


if __name__ == "__main__":
    main()
