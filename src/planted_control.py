#!/usr/bin/env python3
"""planted_control.py - can nMSI detect a planted paired-word feature at the
lengths of quadruplex motifs, and does it detect a planted guanine-tract
feature at the same lengths?

For sequence lengths L = 25, 40 and 60 nt, 2,000 sequences per class:
 * mirror feature: a random segment u of length h = round(L/4) followed by its
   reverse rev(u), placed at a random offset in an i.i.d. background; control:
   u followed by an independent random segment of the same composition as u
   (a permutation of u), so both classes have the same composition;
 * guanine-tract feature: four runs GGG separated by random loops of 1-7 nt
   (the Quadparser consensus) in the background; control: a uniform random
   permutation of the whole feature sequence (identical composition; tracts
   survive only by chance).
Backgrounds are i.i.d. at uniform composition and, separately, G-rich
(pG = 0.40, pA = pC = pT = 0.20). AUC of nMSI for feature versus control
(higher nMSI scored as the feature for the mirror case; reported as is, so
values below 1/2 mean the feature lowers nMSI). G4Hunter is reported for the
guanine-tract case as the positive control. Seed 20261110. Writes
results/additional/planted_control.csv.

License: MIT
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "additional"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261110)
BG = {"uniform": [.25, .25, .25, .25], "G-rich": [.2, .2, .4, .2]}   # A C G T
N = 2000


def nm(x):
    return E.idx_msi(np.asarray(x, np.int8))[1]


def mirror_pair(L, p):
    h = max(3, round(L / 4))
    bg = rng.choice(4, L, p=p)
    u = rng.choice(4, h, p=p)
    o = int(rng.integers(0, L - 2 * h + 1))
    a = bg.copy(); a[o:o + h] = u; a[o + h:o + 2 * h] = u[::-1]
    b = bg.copy(); b[o:o + h] = u; b[o + h:o + 2 * h] = rng.permutation(u)
    return a, b


def tract_pair(L, p):
    while True:
        loops = rng.integers(1, 8, 3)
        motif = [2, 2, 2]
        for lp in loops:
            motif += list(rng.choice(4, lp, p=p)) + [2, 2, 2]
        if len(motif) <= L:
            break
    bg = rng.choice(4, L, p=p)
    o = int(rng.integers(0, L - len(motif) + 1))
    a = bg.copy(); a[o:o + len(motif)] = motif
    b = rng.permutation(a)                                 # same composition, arrangement destroyed
    return a, b


rows = []
for bgname, p in BG.items():
    for L in (25, 40, 60):
        for feat, fn in (("mirror repeat", mirror_pair), ("guanine tracts", tract_pair)):
            A, B = zip(*[fn(L, p) for _ in range(N)])
            sa = np.array([nm(a) for a in A]); sb = np.array([nm(b) for b in B])
            ok = np.isfinite(sa) & np.isfinite(sb)
            y = np.r_[np.ones(ok.sum()), np.zeros(ok.sum())]
            row = dict(background=bgname, L=L, feature=feat, n=int(ok.sum()),
                       auc_nMSI=roc_auc_score(y, np.r_[sa[ok], sb[ok]]))
            if feat == "guanine tracts":
                ga = np.array([E.g4hunter(np.asarray(a, np.int8)) for a in A])
                gb = np.array([E.g4hunter(np.asarray(b, np.int8)) for b in B])
                row["auc_G4Hunter"] = roc_auc_score(np.r_[np.ones(N), np.zeros(N)], np.r_[ga, gb])
            rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "planted_control.csv"), index=False)
