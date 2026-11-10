#!/usr/bin/env python3
"""table1.py — Table 1: the baseline of the frame-resolved index for N words
counted without overlap at uniform composition. Closed form (ratio of
expectations, parity.nmsi_baseline_exact), exact expectation
(parity.nmsi_expectation_conditional) and the mean over 400,000 simulated
sequences (seed 20261025). Writes results/table1_closed_form.csv.

License: MIT
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPS = 400_000
rng = np.random.default_rng(20261025)
rev = np.array([(i % 4) * 16 + ((i // 4) % 4) * 4 + i // 16 for i in range(64)])
nonpal = (np.arange(64) // 16) != (np.arange(64) % 4)


def simulate(N):
    vals, done, chunk = [], 0, max(1000, int(2e7 // N))
    while done < REPS:
        r = min(chunk, REPS - done)
        W = rng.integers(0, 64, size=(r, N))
        C = np.zeros((r, 64), np.int32)
        np.add.at(C, (np.repeat(np.arange(r), N), W.ravel()), 1)
        Cr = C[:, rev]
        num = np.abs(C - Cr)[:, nonpal].sum(1) / 2
        den = (C + Cr)[:, nonpal].sum(1) / 2
        ok = den > 0
        vals.append(1 - num[ok] / den[ok]); done += r
    v = np.concatenate(vals)
    return v.mean(), v.std() / np.sqrt(len(v))


rows = []
for N in (10, 20, 30, 50, 100, 200, 500):
    c = parity.nmsi_baseline_exact(N)
    e = parity.nmsi_expectation_conditional(N)
    m, se = simulate(N)
    rows.append(dict(N=N, closed_form=round(c, 4), exact=round(e, 4), simulated=round(m, 4),
                     simulated_se=round(se, 5), closed_minus_exact=round(c - e, 4),
                     exact_minus_simulated_in_se=round((e - m) / se, 1)))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(ROOT, "results", "table1_closed_form.csv"), index=False)
print(T.to_string(index=False))
