#!/usr/bin/env python3
"""a549_full_mutagenesis.py -- the mutagenesis experiment and the matched-placement
test repeated on every guanine-strand A549 motif (genome-wide replication).

Sequences: data/a549_g4p_all_motifs.txt (23,045 guanine-strand canonical motifs of
the A549 G4P-ChIP-seq peak table; src/derive_a549_set.py --all), filtered exactly as
in experiment_538.load(): ACGT only, >= 16 nt, >= 4 maximal G>=3 runs.

Mutagenesis: the eight conventions of experiment_538 (experiment_538.mutate), with
the same verification (mutant fails the Quadparser consensus and its G4Hunter score
falls) and the same paired nulls (experiment_538.paired_null), at 200 mononucleotide
and 100 dinucleotide paired draws per pair instead of 1,000 and 400 (run time; the
exact tail aggregates over thousands of pairs). The unpaired null of the main
experiment, reported there for comparison, is not computed here. Reported per convention:
pairs, concordance (wild type above mutant, ties one half) with bootstrap interval,
share of exact nMSI ties, the fair-coin sign test, the exact three-outcome tail under
each paired null, and G4Hunter on the same pairs.

Matched placement: matched_placement.DESIGNS on every wild type that satisfies the
consensus, as in matched_placement.main().

Usage:  python3 src/a549_full_mutagenesis.py [--quick] [--procs 2]
Outputs: results/a549_full/mutagenesis.csv, results/a549_full/matched_placement.csv,
         results/a549_full/pairs.csv.gz
"""
from __future__ import annotations
import argparse, itertools, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy import stats

import experiment_538 as E
import placement_null as P
from matched_placement import DESIGNS, q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "a549_full")
K_MONO, K_DI = 200, 100


def load(quick=False):
    d = pd.read_csv(os.path.join(ROOT, "data", "a549_g4p_all_motifs.txt"), sep="\t")
    d.columns = ["index", "peak", "chr", "seq"]
    keep = d.seq.apply(lambda s: set(s) <= set("ACGT") and len(s) >= 16 and len(E.tracts(s)) >= 4)
    d = d[keep].reset_index(drop=True)
    return d.iloc[:400] if quick else d


def work(args):
    """One chunk of wild types: every convention, verification and paired nulls."""
    chunk, seed = args
    rng = np.random.default_rng(seed)                    # mutate (scramble)
    pr = np.random.default_rng(seed + 1)                 # paired mononucleotide null
    pdr = np.random.default_rng(seed + 2)                # paired dinucleotide null
    rows = []
    for i, s in chunk:
        x = E.enc(s); w = E.idx_msi(x); wg = E.g4hunter(x)
        for conv in E.CONV:
            m = E.mutate(s, conv, rng)
            if len(m) != len(s) or E.is_canonical(m):
                continue
            xm = E.enc(m); gm = E.g4hunter(xm)
            if gm >= wg:
                continue
            mm, mn = E.idx_msi(xm)
            if not np.isfinite(mn) or not np.isfinite(w[1]):
                continue
            a = E.paired_null(x, xm, K_MONO, "mono", pr)
            b = E.paired_null(x, xm, K_DI, "dinuc", pdr)
            rows.append(dict(i=i, convention=conv, wt_MSI=w[0], wt_nMSI=w[1], mut_MSI=mm, mut_nMSI=mn,
                             wt_g4=wg, mut_g4=gm,
                             g_nMSI_mono=a["nMSI"][0], t_nMSI_mono=a["nMSI"][1],
                             g_MSI_mono=a["MSI"][0], t_MSI_mono=a["MSI"][1],
                             g_nMSI_di=b["nMSI"][0], t_nMSI_di=b["nMSI"][1]))
    return rows


def summarise_mutagenesis(R, rng):
    out = []
    for conv in E.CONV:
        r = R[R.convention == conv]
        if r.empty:
            continue
        for lab, wc, mc, null in (("nMSI", "wt_nMSI", "mut_nMSI", "nMSI_mono"),
                                  ("MSI", "wt_MSI", "mut_MSI", "MSI_mono"),
                                  ("nMSI/dinuc", "wt_nMSI", "mut_nMSI", "nMSI_di")):
            d = r[wc].values - r[mc].values
            win, tie = d > 1e-12, np.isclose(d, 0); lose = ~win & ~tie
            k = float(win.sum() + 0.5 * tie.sum())
            g, t = r[f"g_{null}"].values, r[f"t_{null}"].values
            score = win + 0.5 * tie
            bs = [rng.choice(score, score.size).mean() for _ in range(2000)]
            out.append(dict(convention=conv, index=lab, n=len(d), concordant=k, obs=k / len(d),
                            obs_lo=np.percentile(bs, 2.5), obs_hi=np.percentile(bs, 97.5),
                            ties=int(tie.sum()), pct_ties=100 * tie.mean(),
                            null=float(np.mean(g + t / 2)), null_tie=float(np.mean(t)),
                            p_fair=stats.binomtest(int(win.sum()), int(win.sum() + lose.sum()), 0.5,
                                                   alternative="greater").pvalue,
                            p_corrected=E.three_outcome_sf(k, g, t), mean_delta=d.mean()))
        dg = r.wt_g4.values - r.mut_g4.values
        out.append(dict(convention=conv, index="G4Hunter (control)", n=len(dg),
                        concordant=float((dg > 0).sum()), obs=(dg > 0).mean(), pct_ties=100 * np.mean(dg == 0),
                        p_fair=stats.binomtest(int((dg > 0).sum()), len(dg), 0.5, alternative="greater").pvalue,
                        mean_delta=dg.mean()))
    return pd.DataFrame(out)


def matched(seqs):
    rng = np.random.default_rng(20261115)
    rows = []
    for k, base, where in DESIGNS:
        res = []
        pos = (lambda a, b: (a + b) // 2) if where == "centre" else ((lambda a, b: a) if where == "5' edge" else (lambda a, b: b - 1))
        for s in seqs:
            if not E.is_canonical(s):
                continue
            tr = E.tracts(s); ab, ok = [], []
            for comb in itertools.combinations(range(len(tr)), k):
                y = list(s)
                for j in comb:
                    y[pos(*tr[j])] = base
                y = "".join(y); sc = P.scores(y)
                (ok if E.is_canonical(y) else ab).append(sc)
            if ab and ok:
                ab, ok = np.array(ab), np.array(ok)
                res.append({nm: np.nanmean([q(ok[:, j], r) for r in ab[:, j]])
                            for j, nm in enumerate(("nMSI", "MSI", "G4Hunter"))})
        R = pd.DataFrame(res)
        row = dict(design=f"G→{base}, {where}, {k} tract{'s' if k > 1 else ''}",
                   consensus_positive=int(sum(E.is_canonical(s) for s in seqs)), wild_types=len(R))
        for nm in ("nMSI", "MSI", "G4Hunter"):
            v = R[nm].dropna().to_numpy()
            bs = [np.mean(rng.choice(v, v.size)) for _ in range(2000)]
            row.update({f"{nm}_mean_q": float(v.mean()), f"{nm}_lo": float(np.percentile(bs, 2.5)),
                        f"{nm}_hi": float(np.percentile(bs, 97.5))})
        rows.append(row); print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--only", choices=["mutagenesis", "matched"], default=None)
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute mutagenesis.csv from the stored pairs.csv.gz without new shuffles")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = load(a.quick); seqs = list(d.seq)
    if a.summary_only:
        R = pd.read_csv(os.path.join(OUT, "pairs.csv.gz"))
        S = summarise_mutagenesis(R, np.random.default_rng(20262201))
        S.to_csv(os.path.join(OUT, "mutagenesis.csv"), index=False); print(S.to_string(index=False)); return
    print(f"{len(d)} wild types with >= 4 G>=3 runs, >= 16 nt", flush=True)
    t0 = time.time()
    if a.only != "matched":
        items = list(enumerate(seqs)); C = 250
        chunks = [(items[j:j + C], 20261200 + 3 * (j // C)) for j in range(0, len(items), C)]
        rows = []
        with Pool(a.procs) as pool:
            for n_done, r in enumerate(pool.imap(work, chunks), 1):
                rows += r
                print(f"  chunk {n_done}/{len(chunks)}  {time.time() - t0:.0f} s", flush=True)
        R = pd.DataFrame(rows)
        R.insert(1, "peak", d.peak.values[R.i.values]); R.insert(2, "chr", d.chr.values[R.i.values])
        R.to_csv(os.path.join(OUT, "pairs.csv.gz"), index=False)
        S = summarise_mutagenesis(R, np.random.default_rng(20262201))   # not a chunk seed
        S.to_csv(os.path.join(OUT, "mutagenesis.csv"), index=False)
        with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format):
            print(S.to_string(index=False))
    if a.only != "mutagenesis":
        M = matched(seqs)
        M.to_csv(os.path.join(OUT, "matched_placement.csv"), index=False)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
