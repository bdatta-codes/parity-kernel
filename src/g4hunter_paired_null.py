#!/usr/bin/env python3
"""g4hunter_paired_null.py - the G4Hunter control under the paired nulls of
Supplementary Table S13.

For every verified pair of the seven designed conventions (the scramble mutant
is random and is not regenerated here),
the aligned wild type and mutant are shuffled jointly exactly as in
experiment_538.paired_null (200 mononucleotide and 100 dinucleotide draws per
pair; seeds 20261112 and 20261113), and the per-pair probabilities that the null
wild type has the higher G4Hunter score (g) and that the two tie (t) are
estimated. The exact three-outcome tail of the observed G4Hunter concordance is
then computed as for nMSI. Writes results/additional/g4hunter_paired_null.csv.

License: MIT
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E
from shuffles import dinuc_shuffle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); OUT = os.path.join(RES, "additional"); os.makedirs(OUT, exist_ok=True)
CONV = ["T_centre", "A_centre", "C_centre", "two_tract", "tract_ablate", "T_3prime", "T_5prime", "scramble"]


def main():
    seqs, _ = E.load()
    rm, rd = np.random.default_rng(20261112), np.random.default_rng(20261113)
    rows = []
    for key in CONV:
        d = pd.read_csv(os.path.join(RES, f"pairs_{key}.csv"))
        rec = {"mono": ([], []), "dinuc": ([], [])}
        for _, r in d.iterrows():
            s = seqs[int(r.i)]
            if key == "scramble":
                continue
            m = E.mutate(s, key, np.random.default_rng(0))
            z = E.enc(s).astype(np.int16) * 4 + E.enc(m).astype(np.int16)
            for kind, K, rng in (("mono", 200, rm), ("dinuc", 100, rd)):
                v = []
                for _ in range(K):
                    zz = rng.permutation(z) if kind == "mono" else dinuc_shuffle(z, rng)
                    v.append(E.g4hunter((zz // 4).astype(np.int8)) - E.g4hunter((zz % 4).astype(np.int8)))
                v = np.array(v)
                rec[kind][0].append(float(np.mean(v > 1e-12))); rec[kind][1].append(float(np.mean(np.isclose(v, 0))))
        if key == "scramble":
            continue
        dg = d.wt_g4.values - d.mut_g4.values
        k = float((dg > 1e-12).sum() + 0.5 * np.isclose(dg, 0).sum())
        row = dict(convention=key, n=len(d), concordance=k / len(d))
        for kind in ("mono", "dinuc"):
            g, t = np.array(rec[kind][0]), np.array(rec[kind][1])
            row[f"null_{kind}"] = float(np.mean(g + t / 2)); row[f"p_{kind}"] = E.three_outcome_sf(k, g, t)
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "g4hunter_paired_null.csv"), index=False)


if __name__ == "__main__":
    main()
