#!/usr/bin/env python3
"""a549_full_set.py -- the A549 analyses on every guanine-strand motif of the peak table.

The A549 G-rich set of the main analyses (n = 300) is the first 300 guanine-strand
canonical motifs of the A549 G4P-ChIP-seq peak table, all on chromosome 1.  This
script repeats every A549 analysis on all 23,045 such motifs (data/a549_g4p_all_motifs.txt,
written by src/derive_a549_set.py --all), genome-wide, with the same definitions,
shuffle counts and conventions as src/g4_runlength_analysis.py:

  boundary analysis   interior G>=3 runs with identical flanks; share of the nMSI
                      numerator from the six boundary words HGG / GGH
  run number          real minus mean of 100 mononucleotide shuffles: number of
                      G-runs, mean run length, nMSI; Spearman rho of the deltas
  real vs shuffles    pooled AUC of real against own shuffles (100 mono, 100 dinuc),
                      % of own shuffles beaten (strict and ties one half), paired
                      Wilcoxon; 95% interval by resampling sequences (B = 200)
  composition         Spearman rho of nMSI and MSI with G4Hunter; of the mean of
                      1,000 mononucleotide and 200 dinucleotide shuffles with the
                      wild-type G4Hunter; share kept

Five motifs that are a single uninterrupted G-run have no defined nMSI (every
word is self-reverse) and are set aside, leaving 23,040.
Every quantity is also given for the distinct sequences (first occurrence
kept) and, for comparison, for the first 300 (identical to data/a549_grich_300.txt,
recomputed here with this script's generator, so values differ from the main
analysis only by shuffle noise).

Usage:  python3 src/a549_full_set.py [--quick]
Outputs: results/a549_full/summary.csv, results/a549_full/per_sequence.csv
"""
from __future__ import annotations
import argparse, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats

from shuffles import mono_shuffle, dinuc_shuffle
from experiment_538 import enc, idx_msi, g4hunter, _PI, _PJ
from g4_runlength_analysis import g_runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "a549_full")
G = 2
SEED = 20261026


def load():
    d = pd.read_csv(os.path.join(ROOT, "data", "a549_g4p_all_motifs.txt"), sep="\t", comment=None)
    d.columns = ["index", "peak", "chr", "seq"]
    # five motifs are a single uninterrupted G-run (15-18 G); every trinucleotide in them
    # is self-reverse, so nMSI is undefined (0/0) and they are set aside
    ok = d.seq.apply(lambda s: np.isfinite(idx_msi(enc(s))[1]))
    print(f"{(~ok).sum()} motifs with undefined nMSI set aside: {sorted(set(d.seq[~ok]))}")
    return d[ok].reset_index(drop=True)


def fast_pooled_auc(real, S_sorted):
    lo = np.searchsorted(S_sorted, real, "left"); hi = np.searchsorted(S_sorted, real, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (real.size * S_sorted.size))


def auc_ci(real, shuf, B, seed=20261103):
    r = np.random.default_rng(seed); n = real.size; v = []
    for _ in range(B):
        ix = r.integers(0, n, n)
        v.append(fast_pooled_auc(real[ix], np.sort(shuf[ix].ravel())))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def compute(seqs, rng, K=100, Km=1000, Kd=200):
    """Per-sequence quantities; shuffles drawn once and reused by every summary."""
    n = len(seqs); X = [enc(s) for s in seqs]
    real = np.array([idx_msi(x) for x in X])                       # MSI, nMSI
    gh = np.array([g4hunter(x) for x in X])
    Im = np.full((n, K, 2), np.nan); Id = np.full((n, K, 2), np.nan)
    ns = np.zeros(n); ms = np.zeros(n); mm = np.zeros(n); dd = np.zeros(n)
    t0 = time.time()
    for i, x in enumerate(X):
        ys = [mono_shuffle(x, rng) for _ in range(K)]
        Im[i] = [idx_msi(y) for y in ys]
        ns[i] = np.mean([g_runs(y).size for y in ys]); ms[i] = np.mean([g_runs(y).mean() for y in ys])
        Id[i] = [idx_msi(dinuc_shuffle(x, rng)) for _ in range(K)]
        mm[i] = np.nanmean([idx_msi(mono_shuffle(x, rng))[1] for _ in range(Km)])
        dd[i] = np.nanmean([idx_msi(dinuc_shuffle(x, rng))[1] for _ in range(Kd)])
        if i % 2000 == 0:
            print(f"  {i}/{n}  {time.time() - t0:.0f} s", flush=True)
    nr = np.array([g_runs(x).size for x in X]); mr = np.array([g_runs(x).mean() for x in X])
    # boundary analysis, per sequence (pooled later)
    bnd = np.array([a * 16 + G * 4 + G for a in (0, 1, 3)] + [G * 16 + G * 4 + b for b in (0, 1, 3)])
    tot = np.zeros(n, int); same = np.zeros(n, int); num = np.zeros(n); numb = np.zeros(n)
    for i, s in enumerate(seqs):
        for m in re.finditer("G{3,}", s):
            if m.start() > 0 and m.end() < len(s):
                tot[i] += 1; same[i] += s[m.start() - 1] == s[m.end()]
        x = X[i].astype(np.int64)
        f = np.bincount(x[:-2] * 16 + x[1:-1] * 4 + x[2:], minlength=64)
        dlt = np.abs(f[_PI] - f[_PJ]); num[i] = dlt.sum()
        numb[i] = dlt[np.isin(_PI, bnd) | np.isin(_PJ, bnd)].sum()
    return dict(real=real, gh=gh, Im=Im, Id=Id, nr=nr, mr=mr, ns=ns, ms=ms, mm=mm, dd=dd,
                tot=tot, same=same, num=num, numb=numb)


def summarise(D, ix, label, B):
    out = []
    def rec(q, v): out.append(dict(subset=label, n=len(ix), quantity=q, value=v))
    real, gh = D["real"][ix], D["gh"][ix]
    rec("interior G>=3 runs", D["tot"][ix].sum())
    rec("% identical flanks", 100 * D["same"][ix].sum() / D["tot"][ix].sum())
    rec("% of nMSI numerator from boundary words", 100 * D["numb"][ix].sum() / D["num"][ix].sum())
    Im, Id = D["Im"][ix], D["Id"][ix]
    dn = real[:, 1] - np.nanmean(Im[:, :, 1], axis=1)
    drun = D["nr"][ix] - D["ns"][ix]; dmr = D["mr"][ix] - D["ms"][ix]
    rec("mean delta number of G-runs (real - mono shuffles)", drun.mean())
    rec("mean delta mean G-run length", dmr.mean())
    r, p = stats.spearmanr(dn, drun)
    rec("Spearman rho(d nMSI, d runs)", r); rec("p, rho(d nMSI, d runs)", p)
    rec("Spearman rho(d nMSI, d mean run length)", stats.spearmanr(dn, dmr)[0])
    for kind, I in (("mono", Im), ("dinuc", Id)):
        for j, name in ((1, "nMSI"), (0, "MSI")):
            sh = I[:, :, j]; rv = real[:, j]
            ok = np.isfinite(sh)
            S = np.sort(sh[ok])
            rec(f"pooled AUC real vs own {kind} shuffles, {name}", fast_pooled_auc(rv, S))
            lo, hi = auc_ci(rv, sh, B)
            rec(f"pooled AUC real vs own {kind} shuffles, {name}, 95% CI low", lo)
            rec(f"pooled AUC real vs own {kind} shuffles, {name}, 95% CI high", hi)
            below = np.nansum(sh < rv[:, None], 1); eq = np.nansum(sh == rv[:, None], 1); m = ok.sum(1)
            rec(f"% of own {kind} shuffles strictly beaten, {name}", 100 * np.mean(below / m))
            rec(f"% of own {kind} shuffles beaten, ties one half, {name}", 100 * np.mean((below + 0.5 * eq) / m))
            rec(f"paired Wilcoxon p (real vs {kind} shuffle mean), {name}",
                stats.wilcoxon(rv, np.nanmean(sh, 1)).pvalue)
    r_n = stats.spearmanr(real[:, 1], gh)[0]; r_m = stats.spearmanr(real[:, 0], gh)[0]
    rm = stats.spearmanr(D["mm"][ix], gh)[0]; rd = stats.spearmanr(D["dd"][ix], gh)[0]
    rec("rho nMSI vs G4Hunter", r_n); rec("rho MSI vs G4Hunter", r_m)
    rec("rho mean mono-shuffled nMSI vs WT G4Hunter", rm)
    rec("rho mean dinuc-shuffled nMSI vs WT G4Hunter", rd)
    rec("share of association kept by mono shuffles", rm / r_n)
    rec("share of association kept by dinuc shuffles", rd / r_n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = load()
    if a.quick:
        d = d.iloc[:600]
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    D = compute(list(d.seq), rng, K=100, Km=100 if a.quick else 1000, Kd=50 if a.quick else 200)
    n = len(d)
    first = ~d.seq.duplicated().to_numpy()
    B = 50 if a.quick else 200
    rows = (summarise(D, np.arange(n), "all motifs", B)
            + summarise(D, np.flatnonzero(first), "distinct sequences", B)
            + summarise(D, np.arange(300), "first 300 (A549 G-rich set)", B)
            + summarise(D, np.arange(300, n), "all but the first 300", B))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    real = D["real"]
    pd.DataFrame(dict(index=d["index"], peak=d.peak, chr=d.chr, seq=d.seq, length=d.seq.str.len(),
                      nMSI=real[:, 1], MSI=real[:, 0], G4Hunter=D["gh"],
                      mono_mean_nMSI_K100=np.nanmean(D["Im"][:, :, 1], 1),
                      dinuc_mean_nMSI_K100=np.nanmean(D["Id"][:, :, 1], 1),
                      mono_mean_nMSI_K1000=D["mm"], dinuc_mean_nMSI_K200=D["dd"],
                      n_runs=D["nr"], mean_run=D["mr"], d_runs=D["nr"] - D["ns"], d_mean_run=D["mr"] - D["ms"])
                 ).to_csv(os.path.join(OUT, "per_sequence.csv"), index=False)
    w = s.pivot_table(index="quantity", columns="subset", values="value", sort=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.float_format", "{:.4g}".format):
        print(w.to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
