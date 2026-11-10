#!/usr/bin/env python3
"""matched_placement.py - does any score register consensus abolition once the
placement of the substitutions is matched?

For every wild type that satisfies the Quadparser consensus, the same
substitution (same base, same position within the tract: centre, 5' edge or
3' edge) is applied to every combination of k of its G>=3 tracts (k = 1, 2).
Each variant either fails the consensus (abolished) or still satisfies it
(retained); the two differ only in WHICH tracts were hit, not in how many, at
what tract position or with what base, and they have identical composition.
For each wild type with at least one variant of each kind, q = mean over
abolished variants of P(score of a retained variant > score of the abolished
variant), ties counted as one half; q > 1/2 means the score ranks retained
above abolished. Reported: mean q over wild types with a 2,000-resample
bootstrap interval over wild types (seed 20261115), for nMSI, MSI and
G4Hunter, and the number of wild types by source collection.
Writes results/additional/matched_placement.csv.

License: MIT
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E
import placement_null as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "additional"); os.makedirs(OUT, exist_ok=True)
DESIGNS = [(1, "T", "centre"), (2, "T", "centre"), (2, "A", "centre"), (2, "C", "centre"),
           (1, "T", "5' edge"), (2, "T", "5' edge"), (1, "T", "3' edge"), (2, "T", "3' edge")]


def q(nullv, realv):
    a = np.asarray(nullv, float); a = a[np.isfinite(a)]
    if a.size == 0 or not np.isfinite(realv):
        return np.nan
    return float(np.mean(a > realv + 1e-12) + 0.5 * np.mean(np.isclose(a, realv)))


def main():
    seqs, src = E.load()
    rng = np.random.default_rng(20261115)
    rows = []
    for k, base, where in DESIGNS:
        res = []
        for i, s in enumerate(seqs):
            if not E.is_canonical(s):
                continue
            tr = E.tracts(s)
            pos = (lambda a, b: (a + b) // 2) if where == "centre" else ((lambda a, b: a) if where == "5' edge" else (lambda a, b: b - 1))
            ab, ok = [], []
            for comb in itertools.combinations(range(len(tr)), k):
                y = list(s)
                for j in comb:
                    y[pos(*tr[j])] = base
                y = "".join(y); sc = P.scores(y)
                (ok if E.is_canonical(y) else ab).append(sc)
            if ab and ok:
                ab, ok = np.array(ab), np.array(ok)
                res.append(dict(src=src[i], **{nm: np.nanmean([q(ok[:, j], r) for r in ab[:, j]])
                                              for j, nm in enumerate(("nMSI", "MSI", "G4Hunter"))}))
        R = pd.DataFrame(res)
        row = dict(design=f"G→{base}, {where}, {k} tract{'s' if k > 1 else ''}", wild_types=len(R))
        for s_ in sorted(set(src)):
            row[f"n_{s_}"] = int((R.src == s_).sum())
        for nm in ("nMSI", "MSI", "G4Hunter"):
            v = R[nm].dropna().to_numpy()
            bs = [np.mean(rng.choice(v, v.size)) for _ in range(2000)]
            row.update({f"{nm}_mean_q": float(v.mean()), f"{nm}_lo": float(np.percentile(bs, 2.5)),
                        f"{nm}_hi": float(np.percentile(bs, 97.5))})
        rows.append(row); print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "matched_placement.csv"), index=False)


if __name__ == "__main__":
    main()
