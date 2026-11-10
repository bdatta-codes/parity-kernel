#!/usr/bin/env python3
"""chr21_markov_decomposition.py - split the chromosome 21 symmetry deficit into
a part that a reversible background can produce and a part that requires
non-reversibility.

All 133,601 clean, non-overlapping 300-nt windows (results/chr21/
windows300.csv.gz) are grouped by GC content into bins of 0.02. Within each bin
three backgrounds are fitted to the pooled sequence of all its windows (so that
the fitted parameters carry no sampling noise from any single window; fitting
a chain to a single 300-nt window biases the non-reversible
component by about 0.03 even on i.i.d. sequence):
  iid   independent bases at the bin's pooled composition;
  rev   the reversible first-order chain from the symmetrised pooled
        dinucleotide counts, J = (N + N^T)/2, T = J / rowsums;
  full  the unconstrained first-order chain from the pooled dinucleotide
        counts, started from its stationary distribution.
From each, 1,000 sequences of 300 nt are simulated and the mean nMSI taken
(E_iid, E_rev, E_full); O is the mean observed nMSI of the bin's windows. The
deficit E_iid - O splits exactly into
  E_iid - E_rev    reversible first-order memory,
  E_rev - E_full   first-order non-reversibility,
  E_full - O       what a first-order chain does not capture (higher order).
The share of the non-reversible part due to CpG/GpC alone is estimated by
re-simulating the full chain with only the CpG and GpC counts replaced by
their mean (E_cpg); E_cpg - E_full is reported as cpg_gpc_asymmetry. Because
this attribution depends on the order in which the asymmetry is removed, the
reverse order is also reported: the reversible chain with only the CpG and
GpC counts restored to their observed, unequal values (E_rev_cpg);
E_rev - E_rev_cpg is reported as cpg_gpc_asymmetry_from_reversible. The last
component is labelled higher_order but is the residual beyond a bin-pooled
first-order chain, so it also absorbs between-window variation in
first-order parameters at fixed GC.
Components are averaged over bins weighted by window counts (bins with at
least 200 windows, covering nearly all windows). Seed 20261111. Needs the
chr21 FASTA:

    python3 src/chr21_markov_decomposition.py --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz

For another chromosome, first run chromosome21_windows.py --out results/chrNN on
its FASTA, then pass --windows results/chrNN/windows300.csv.gz here, together with
--outdir results/chrNN --tag chrNN so this chromosome's outputs do not overwrite
chromosome 21's.

Writes <outdir>/<tag>_markov_decomposition.csv (summary) and <tag>_markov_bins.csv
(per GC bin); by default <outdir> is results/additional and <tag> is chr21.

License: MIT
"""
import argparse, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chromosome21_windows as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "additional"); os.makedirs(OUT, exist_ok=True)
REV = np.array([(i % 4) * 16 + ((i // 4) % 4) * 4 + i // 16 for i in range(64)])
NP = (np.arange(64) // 16) != (np.arange(64) % 4)
LUT = np.full(256, -1, np.int64)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i


def nmsi_batch(X):
    t = X[:, :-2] * 16 + X[:, 1:-1] * 4 + X[:, 2:]
    C = np.zeros((X.shape[0], 64))
    np.add.at(C, (np.repeat(np.arange(X.shape[0]), t.shape[1]), t.ravel()), 1)
    Cr = C[:, REV]
    return 1 - (np.abs(C - Cr)[:, NP].sum(1) / 2) / ((C + Cr)[:, NP].sum(1) / 2)


def sim_chain(T, pi, n, L, rng):
    X = np.empty((n, L), np.int64); X[:, 0] = rng.choice(4, n, p=pi)
    cum = np.cumsum(T, 1)
    for t in range(1, L):
        u = rng.random(n); X[:, t] = np.minimum((u[:, None] > cum[X[:, t - 1]]).sum(1), 3)
    return X


def stationary(T):
    w, v = np.linalg.eig(T.T); p = np.real(v[:, np.argmin(abs(w - 1))]); return p / p.sum()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--chr21", required=True)
    ap.add_argument("--sims", type=int, default=1000); ap.add_argument("--min-windows", type=int, default=200)
    ap.add_argument("--windows", default=None,
                    help="windows300.csv.gz for this chromosome (from chromosome21_windows.py --out); "
                         "default results/chr21/windows300.csv.gz")
    ap.add_argument("--outdir", default=None,
                    help="directory for the two output CSVs; default results/additional")
    ap.add_argument("--tag", default="chr21",
                    help="filename prefix for the outputs; default chr21 "
                         "(use e.g. chr22 on another chromosome so chr21's files are not overwritten)")
    a = ap.parse_args()
    outdir = a.outdir or OUT; os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(20261111)
    seq = np.frombuffer(W.read_single_fasta(a.chr21).upper().encode(), np.uint8)
    w = pd.read_csv(a.windows or os.path.join(ROOT, "results", "chr21", "windows300.csv.gz"))
    w["bin"] = np.floor(w.gc / 0.02 + 1e-9).astype(int)   # GC = k/300 exactly on a bin edge goes to the upper bin
    rows = []
    for b, g in w.groupby("bin"):
        if len(g) < a.min_windows:
            continue
        N = np.zeros((4, 4)); comp = np.zeros(4)
        for s in g.start.values:
            x = LUT[seq[s:s + 300]]
            np.add.at(N, (x[:-1], x[1:]), 1); comp += np.bincount(x, minlength=4)
        comp /= comp.sum()
        J = (N + N.T) / 2; Tr = J / J.sum(1, keepdims=True); pr = J.sum(1) / J.sum()
        Tf = N / N.sum(1, keepdims=True); pf = stationary(Tf)
        e_iid = np.nanmean(nmsi_batch(rng.choice(4, (a.sims, 300), p=comp)))
        e_rev = np.nanmean(nmsi_batch(sim_chain(Tr, pr, a.sims, 300, rng)))
        e_full = np.nanmean(nmsi_batch(sim_chain(Tf, pf, a.sims, 300, rng)))
        # the full chain with only the CpG/GpC asymmetry removed (both set to their mean)
        Nc = N.copy(); Nc[1, 2] = Nc[2, 1] = (N[1, 2] + N[2, 1]) / 2
        Tc = Nc / Nc.sum(1, keepdims=True); pc = stationary(Tc)
        e_cpg_sym = np.nanmean(nmsi_batch(sim_chain(Tc, pc, a.sims, 300, rng)))
        Ja = J.copy(); Ja[1, 2] = N[1, 2]; Ja[2, 1] = N[2, 1]
        Ta = Ja / Ja.sum(1, keepdims=True); pa = stationary(Ta)
        e_rev_cpg = np.nanmean(nmsi_batch(sim_chain(Ta, pa, a.sims, 300, rng)))
        cpg_oe = N[1, 2] / N.sum() / (comp[1] * comp[2])
        rows.append(dict(gc_bin=round(b * 0.02, 2), n_windows=len(g), observed=g.nmsi.mean(),
                         e_iid=e_iid, e_rev=e_rev, e_full=e_full, e_full_cpg_symmetrised=e_cpg_sym, e_rev_cpg_asymmetric=e_rev_cpg, cpg_obs_exp=cpg_oe,
                         gpc_obs_exp=N[2, 1] / N.sum() / (comp[1] * comp[2])))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d["deficit"] = d.e_iid - d.observed
    d["memory"] = d.e_iid - d.e_rev
    d["nonreversibility"] = d.e_rev - d.e_full
    d["higher_order"] = d.e_full - d.observed
    d["cpg_gpc_asymmetry"] = d.e_full_cpg_symmetrised - d.e_full   # removed last (from the full chain)
    d["cpg_gpc_asymmetry_from_reversible"] = d.e_rev - d.e_rev_cpg_asymmetric   # added first (to the reversible chain)
    d.to_csv(os.path.join(outdir, a.tag + "_markov_bins.csv"), index=False)
    wt = d.n_windows / d.n_windows.sum()
    out = [dict(component=c, weighted_mean=float((d[c] * wt).sum()),
                share_of_deficit=float((d[c] * wt).sum() / (d.deficit * wt).sum()))
           for c in ("deficit", "memory", "nonreversibility", "cpg_gpc_asymmetry", "cpg_gpc_asymmetry_from_reversible", "higher_order")]
    o = pd.DataFrame(out); o["windows_covered"] = int(d.n_windows.sum())
    o.to_csv(os.path.join(outdir, a.tag + "_markov_decomposition.csv"), index=False)
    print(o.to_string(index=False))


if __name__ == "__main__":
    main()
