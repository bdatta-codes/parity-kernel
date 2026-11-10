#!/usr/bin/env python3
"""consensus_positive_subset.py - the mutagenesis results restricted to wild types
that satisfy the Quadparser consensus.

Inclusion in the 538-sequence pool requires four or more G>=3 tracts, not a
consensus-positive wild type, and 40 wild types already fail the consensus
(their loops exceed seven bases). For their mutants, "fails the consensus" was
true before mutation, so abolition is not verified for those 304 pairs. This
script recomputes concordance, ties, the fair-coin sign test and the exact tails
against both paired nulls on the 498 consensus-positive wild types, from the
results/pairs_*.csv (no new shuffles). Writes
results/additional/consensus_positive_subset.csv.

License: MIT
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CONV = ["T_centre", "A_centre", "C_centre", "two_tract", "tract_ablate", "T_3prime", "T_5prime", "scramble"]
seqs, _ = E.load()
pos = {i for i, s in enumerate(seqs) if E.is_canonical(s)}
rows = []
for conv in CONV:
    R = pd.read_csv(os.path.join(RES, f"pairs_{conv}.csv"))
    R = R[R.i.isin(pos)]
    row = dict(convention=conv, n=len(R))
    d = R.wt_nMSI.values - R.mut_nMSI.values
    win, tie = d > 1e-12, np.isclose(d, 0); lose = ~win & ~tie
    k = float(win.sum() + 0.5 * tie.sum())
    row.update(delta0_pct=100 * tie.mean(), concord_pct=100 * k / len(d),
               p_fair=stats.binomtest(int(win.sum()), int(win.sum() + lose.sum()), 0.5, alternative="greater").pvalue)
    for null in ("nMSI_mono", "nMSI_di"):
        g, t = R[f"g_{null}"].values, R[f"t_{null}"].values
        row[f"null_{null}"] = float(np.mean(g + t / 2)); row[f"p_{null}"] = E.three_outcome_sf(k, g, t)
    rows.append(row)
    print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()})
print("consensus-positive wild types:", len(pos), "of", len(seqs))
os.makedirs(os.path.join(RES, "additional"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "additional", "consensus_positive_subset.csv"), index=False)
