#!/usr/bin/env python3
"""kappa_inf.py -- the large-N overlap constant kappa derived in closed form
(closed_form.kappa_inf), set against the values fitted by simulation.

kappa_inf = C - C', where C is the non-overlapping constant of Eq. 3 and C' the
sliding-window constant from the per-position variance of each pair difference
under independent bases (overlapping-word moments). Fitted kappa absorbs, in
addition, the finite-N terms over the length range of the fit, so it lies a
little below kappa_inf on short sequences.

Writes results/closed_form/kappa_inf.csv (Table 2 grid, k = 3) and
kappa_inf_word_length.csv (k = 3-5, uniform composition).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import closed_form as cf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "closed_form")


def main():
    t2 = pd.read_csv(os.path.join(OUT, "table2_kappa.csv"))
    rows = []
    for gc in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
        for sk in (0.0, 0.4, 0.8):
            ki, C, Cp = cf.kappa_inf(3, cf.composition(gc, sk))
            r = dict(GC=gc, skew=sk, kappa_inf=ki, C=C, C_prime=Cp)
            for run in ("replicate_4000", "reference_100000"):
                r[f"kappa_fitted_{run}"] = float(t2[(t2.run == run) & np.isclose(t2.GC, gc)][f"skew_{sk}"].iloc[0])
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "kappa_inf.csv"), index=False)
    t3 = pd.read_csv(os.path.join(OUT, "tableS7_word_length.csv"))
    rows3 = []
    for k in (3, 4, 5):
        ki, C, Cp = cf.kappa_inf(k)
        rows3.append(dict(k=k, kappa_inf=ki, C=C, C_prime=Cp,
                          kappa_fitted=float(t3[(t3.k == k) & (t3.pairing == "mirror")].kappa.iloc[0])))
    pd.DataFrame(rows3).to_csv(os.path.join(OUT, "kappa_inf_word_length.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False)); print(pd.DataFrame(rows3).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
