#!/usr/bin/env python3
"""si_sensitivity.py — Supplementary Table S10.

For every convention, the exact one-sided three-outcome tail of the nMSI
concordance under the paired null is recomputed (a) with every per-pair
probability g (null wild type higher) displaced together by 0.05, at the
expense of the probability that the null mutant is higher and with the tie
probability t unchanged, in the direction that weakens (+) and strengthens (-)
the result (systematic misspecification), and (b) with each g perturbed
independently by N(0, 0.025^2), kept within [0, 1 - t], 1,000 times
(estimation error). 0.025 is the largest Monte Carlo standard error of a
proportion estimated from 400 draws, and conservative for 1,000.

    python3 src/si_sensitivity.py          # reads results/pairs_*.csv

Writes results/supplementary_tables/TableS10_null_sensitivity.csv (a) and
TableS10b_independent_error.csv (b; seed 20261025).

License: MIT
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SE, K_SE = 0.025, 2
CONV = [("T_centre", "G→T, tract centre"), ("A_centre", "G→A, tract centre"),
        ("C_centre", "G→C, tract centre"), ("two_tract", "G→T, two tracts only"),
        ("tract_ablate", "first tract replaced by thymine"), ("T_3prime", "G→T, 3′ edge"),
        ("T_5prime", "G→T, 5′ edge"), ("scramble", "whole-sequence scramble")]


def tri_sf(k, g, t):
    """P(S >= k), S = sum of per-pair outcomes 1 (prob g), 1/2 (prob t), 0."""
    dist = np.zeros(2 * len(g) + 1); dist[0] = 1.0
    for gi, ti in zip(g, t):
        li = max(0.0, 1.0 - gi - ti)
        new = dist * li; new[1:] += dist[:-1] * ti; new[2:] += dist[:-2] * gi
        dist = new
    return float(dist[int(round(2 * k)):].sum())


def shift(g, t, delta):
    """Move probability delta from 'mutant higher' to 'wild type higher' (delta > 0
    weakens the result), keeping the tie probability and staying inside [0, 1 - t]."""
    return np.clip(g + delta, 0.0, 1.0 - t)



rows = []
for key, label in CONV:
    d = pd.read_csv(os.path.join(RES, f"pairs_{key}.csv"))
    x = d.wt_nMSI - d.mut_nMSI
    k = float((x > 1e-12).sum() + 0.5 * np.isclose(x, 0).sum())
    row = dict(convention=label)
    for null, col in (("mono", "nMSI_mono"), ("dinuc", "nMSI_di")):
        g, t = d[f"g_{col}"].values, d[f"t_{col}"].values
        row[f"p_{null}"] = tri_sf(k, g, t)
        row[f"p_{null}_plus2SE"] = tri_sf(k, shift(g, t, K_SE * SE), t)
        row[f"p_{null}_minus2SE"] = tri_sf(k, shift(g, t, -K_SE * SE), t)
    rows.append(row)
T = pd.DataFrame(rows)
os.makedirs(os.path.join(RES, "supplementary_tables"), exist_ok=True)
T.to_csv(os.path.join(RES, "supplementary_tables", "TableS10_null_sensitivity.csv"), index=False)
pd.set_option("display.width", 200)
print(T.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# Independent Monte Carlo error. The displacement above moves every per-pair
# null together, which tests a SYSTEMATIC misspecification of the null of that
# size. Estimation error from a finite number of shuffles is instead independent
# across pairs; here each per-pair null is perturbed independently by
# N(0, 0.025^2) (kept within [0, 1 - t]) and the tail recomputed, 1,000 times.
rng = np.random.default_rng(20261025)
R = 1000
rows = []
for key, label in CONV:
    d = pd.read_csv(os.path.join(RES, f"pairs_{key}.csv"))
    x = d.wt_nMSI - d.mut_nMSI
    k = float((x > 1e-12).sum() + 0.5 * np.isclose(x, 0).sum())
    row = dict(convention=label)
    for null, col in (("mono", "nMSI_mono"), ("dinuc", "nMSI_di")):
        g, t = d[f"g_{col}"].values, d[f"t_{col}"].values
        p0 = tri_sf(k, g, t)
        ps = np.array([tri_sf(k, shift(g, t, rng.normal(0, SE, len(g))), t) for _ in range(R)])
        row[f"p_{null}"] = p0
        row[f"p_{null}_median"] = float(np.median(ps))
        row[f"p_{null}_q025"] = float(np.quantile(ps, 0.025))
        row[f"p_{null}_q975"] = float(np.quantile(ps, 0.975))
        row[f"{null}_verdict_flips_pct"] = 100.0 * float(np.mean((ps < 0.05) != (p0 < 0.05)))
    rows.append(row)
U = pd.DataFrame(rows)
U.to_csv(os.path.join(RES, "supplementary_tables", "TableS10b_independent_error.csv"), index=False)
print()
print(U.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
