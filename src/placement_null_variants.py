#!/usr/bin/env python3
"""placement_null_variants.py - robustness of the placement null.

(1) Tract-only placement: as placement_null.py, but the substitutions are placed
    only at guanines lying in G>=3 tracts, so that real and null mutants both hit
    tract guanines and differ in which tract positions are hit, not in tract
    versus loop. Seed 20261114.
(2) From the main per-pair results: pairs with at least 20 kept null mutants only,
    and the mean q weighted by the number of kept null mutants.
Writes results/additional/placement_null_variants.csv.

License: MIT
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E
import placement_null as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "additional")


def boot(v, rng):
    bs = [np.mean(rng.choice(v, v.size)) for _ in range(2000)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    rng = np.random.default_rng(20261114)
    seqs, _ = E.load()
    rows = []
    for key, label in P.CONV:
        d = pd.read_csv(os.path.join(RES, f"pairs_{key}.csv"))
        qs = {"nMSI": [], "MSI": [], "G4Hunter": []}
        for _, r in d.iterrows():
            s = seqs[int(r.i)]
            m = E.mutate(s, key, np.random.default_rng(0))
            diff = [p for p in range(len(s)) if s[p] != m[p]]
            new = [m[p] for p in diff]
            tpos = np.array([p for a, b in E.tracts(s) for p in range(a, b)])
            if len(tpos) < len(diff):
                continue
            real = P.scores(m); null = []; seen = set(); tries = 0
            while len(null) < P.K and tries < P.MAX_TRY:
                tries += 1
                pos = rng.choice(tpos, size=len(diff), replace=False)
                if set(pos.tolist()) == set(diff):
                    continue
                y = list(s)
                for p, b in zip(pos, rng.permutation(new)):
                    y[p] = b
                y = "".join(y)
                if y not in seen and E.is_canonical(y):
                    seen.add(y); null.append(P.scores(y))
            if not null:
                continue
            nv = np.array(null)
            for j, nm in enumerate(("nMSI", "MSI", "G4Hunter")):
                qs[nm].append(P.q(nv[:, j], real[j]))
        row = dict(variant="tract guanines only", convention=label, pairs=len(qs["nMSI"]))
        for nm, v in qs.items():
            v = np.array(v, float); v = v[np.isfinite(v)]
            lo, hi = boot(v, rng) if v.size else (np.nan, np.nan)
            row.update({f"{nm}_mean_q": float(v.mean()) if v.size else np.nan, f"{nm}_lo": lo, f"{nm}_hi": hi})
        rows.append(row); print(row, flush=True)
    pp = pd.read_csv(os.path.join(OUT, "placement_null_per_pair.csv"))
    for _, label in P.CONV:
        x = pp[pp.convention == label]
        y = x[x.kept >= 20]
        row = dict(variant="at least 20 kept null mutants", convention=label, pairs=len(y))
        for nm in ("nMSI", "MSI", "G4Hunter"):
            v = y[f"q_{nm}"].dropna().to_numpy()
            lo, hi = boot(v, rng) if v.size else (np.nan, np.nan)
            row.update({f"{nm}_mean_q": float(v.mean()) if v.size else np.nan, f"{nm}_lo": lo, f"{nm}_hi": hi})
        rows.append(row)
        row = dict(variant="weighted by kept null mutants", convention=label, pairs=len(x))
        for nm in ("nMSI", "MSI", "G4Hunter"):
            ok = x[f"q_{nm}"].notna()
            row[f"{nm}_mean_q"] = float(np.average(x[f"q_{nm}"][ok], weights=x.kept[ok]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "placement_null_variants.csv"), index=False)


if __name__ == "__main__":
    main()
