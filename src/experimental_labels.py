#!/usr/bin/env python3
"""experimental_labels.py - the indices against experimentally labelled folding.

The only sequences in this study whose folding was established experimentally
rather than by a sequence rule are the evaluation set of the preprint that
introduced MSI and nMSI (Arya & Datta 2026, bioRxiv 10.64898/2026.07.05.736592):
the fourteen oncogene-promoter quadruplexes, MYC WT, GPX4 and CLIC4 (folding by
CD, NMR or G4-ChIP-qPCR) and five non-folding mutants, MYC MUT, MYC MUT CORE,
MYC MUT MIN (CD) and mutGPX4, mutCLIC4 (G4-ChIP-negative). The KRAS swap
sequence of that set is not reproduced here. That leaves 17 folding and 5
non-folding sequences (data/promoter_panel.fasta, data/published_pairs.fasta).

For each sequence: MSI, nMSI, the nMSI z-score against 2,000 mononucleotide
shuffles of itself, the mean nMSI of those shuffles (what composition alone
predicts), G4Hunter, the Quadparser consensus, the guanine fraction and the
number of G>=3 tracts. Each is used as a classifier of folding: AUC with a
2,000-resample stratified bootstrap interval, and, for nMSI, the performance
of the fixed criterion nMSI >= 0.50. Seed 20261119. Writes
results/additional/experimental_labels.csv (per sequence) and
experimental_labels_auc.csv (summary).

License: MIT
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "additional"); os.makedirs(OUT, exist_ok=True)
NEG = {"MYC_MUT", "MYC_MUT_CORE", "MYC_MUT_MIN", "mutGPX4", "mutCLIC4"}


def read_fasta(p):
    out, name = [], None
    for l in open(p):
        l = l.strip()
        if l.startswith(">"):
            name = l[1:]
        elif l:
            out.append((name, l.upper()))
    return out


def main():
    rng = np.random.default_rng(20261119)
    S = read_fasta(os.path.join(ROOT, "data", "promoter_panel.fasta")) + \
        read_fasta(os.path.join(ROOT, "data", "published_pairs.fasta"))
    rows = []
    for name, s in S:
        x = E.enc(s); msi, nm = E.idx_msi(x)
        sh = np.array([E.idx_msi(rng.permutation(x))[1] for _ in range(2000)]); sh = sh[np.isfinite(sh)]
        rows.append(dict(name=name, folds=int(name not in NEG), length=len(s), MSI=msi, nMSI=nm,
                         nMSI_z=(nm - sh.mean()) / sh.std(), nMSI_shuffle_mean=sh.mean(),
                         G4Hunter=E.g4hunter(x), Quadparser=int(E.is_canonical(s)),
                         G_fraction=s.count("G") / len(s), G3_tracts=len(E.tracts(s))))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "experimental_labels.csv"), index=False)
    y = d.folds.to_numpy(); pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    summ = []
    for f in ("MSI", "nMSI", "nMSI_z", "nMSI_shuffle_mean", "G4Hunter", "Quadparser", "G_fraction", "G3_tracts"):
        v = d[f].to_numpy(float); auc = roc_auc_score(y, v); bs = []
        for _ in range(2000):
            ii = np.r_[rng.choice(pos, pos.size), rng.choice(neg, neg.size)]
            bs.append(roc_auc_score(y[ii], v[ii]))
        summ.append(dict(score=f, auc=auc, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    s = pd.DataFrame(summ)
    call = d.nMSI >= 0.5
    s.attrs = {}
    extra = pd.DataFrame([dict(score="nMSI >= 0.50: sensitivity", auc=float(call[y == 1].mean()), lo=np.nan, hi=np.nan),
                          dict(score="nMSI >= 0.50: specificity", auc=float((~call[y == 0]).mean()), lo=np.nan, hi=np.nan)])
    s = pd.concat([s, extra], ignore_index=True)
    s.to_csv(os.path.join(OUT, "experimental_labels_auc.csv"), index=False)
    print(d.round(3).to_string(index=False)); print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
