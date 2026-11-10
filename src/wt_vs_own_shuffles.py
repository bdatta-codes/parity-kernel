#!/usr/bin/env python3
"""wt_vs_own_shuffles.py - do the 538 wild types of the mutagenesis experiment
score above or below their own shuffles?

For every wild type, 200 mononucleotide (uniform permutation) and 200
dinucleotide (Altschul-Erickson) shuffles; per sequence, the fraction of its
shuffles that it outscores (ties one half); the mean over sequences and a
Wilcoxon signed-rank test of those fractions against 1/2. Seed 20261104.
Writes results/si/wt_vs_own_shuffles.json.

License: MIT
"""
import json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E
from shuffles import dinuc_shuffle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(20261104)
seqs, _ = E.load()
K = 200
out = {}
for kind in ("mono", "dinuc"):
    for j, name in ((1, "nMSI"), (0, "MSI")):
        out.setdefault(kind, {})[name] = []
fr = {k: {"nMSI": [], "MSI": []} for k in ("mono", "dinuc")}
for s in seqs:
    x = E.enc(s); w = E.idx_msi(x)
    for kind in ("mono", "dinuc"):
        v = np.array([E.idx_msi(E.mono(x, rng) if kind == "mono" else dinuc_shuffle(x, rng)) for _ in range(K)])
        for j, name in ((1, "nMSI"), (0, "MSI")):
            a = v[:, j][np.isfinite(v[:, j])]
            fr[kind][name].append(np.mean((w[j] > a) + 0.5 * (w[j] == a)))
res = {"n_sequences": len(seqs), "shuffles_per_sequence": K}
for kind in fr:
    for name, f in fr[kind].items():
        f = np.array(f)
        res[f"{kind}_{name}_mean_fraction_beaten"] = float(f.mean())
        res[f"{kind}_{name}_wilcoxon_p"] = float(stats.wilcoxon(f - 0.5).pvalue)
os.makedirs(os.path.join(ROOT, "results", "si"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "results", "si", "wt_vs_own_shuffles.json"), "w"), indent=1)
print(res)
