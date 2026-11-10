#!/usr/bin/env python3
"""matched_placement_length.py -- the placement-matched test, matched also on the
length of the tracts that were hit.

matched_placement.py compares consensus-abolishing and consensus-retaining variants
made with the same substitution, tract position and number of tracts hit. They still
differ in WHICH tracts were hit, and that is confounded with tract length: hitting the
edge of a G>=4 tract leaves a G3 run and usually retains the consensus, whereas hitting
a G3 tract abolishes it. Here every abolishing variant is compared only with retaining
variants of the same wild type whose hit tracts have the same lengths (as a sorted
tuple), so the two differ only in the position of the hit tracts in the sequence.
q = P(retaining variant scores higher), ties one half; mean over wild types with at
least one length-matched comparison, with a 2,000-resample bootstrap interval over
wild types. Also reported: the mean length of the hit tracts in each class before
matching, and q matched additionally on the two bases flanking each hit tract
(columns *_flank; '^'/'$' mark a sequence end), since the flanks decide whether
the boundary words a substitution creates are scored or cancel.

Usage:  python3 src/matched_placement_length.py [--set pooled|a549|both]
Writes results/additional/matched_placement_length.csv and
results/a549_full/matched_placement_length.csv
"""
import argparse, itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import experiment_538 as E
import placement_null as P
from matched_placement import DESIGNS, q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def analyse(seqs, label):
    rng = np.random.default_rng(20261119)
    rows = []
    for k, base, where in DESIGNS:
        pos = (lambda a, b: (a + b) // 2) if where == "centre" else ((lambda a, b: a) if where == "5' edge" else (lambda a, b: b - 1))
        res = []; len_ab = []; len_ok = []
        for s in seqs:
            if not E.is_canonical(s):
                continue
            tr = E.tracts(s); ab, ok = [], []
            for comb in itertools.combinations(range(len(tr)), k):
                y = list(s)
                for j in comb:
                    y[pos(*tr[j])] = base
                y = "".join(y); key = tuple(sorted(tr[j][1] - tr[j][0] for j in comb))
                fl = lambda a, b: (s[a - 1] if a > 0 else "^", s[b] if b < len(s) else "$")
                key2 = tuple(sorted((tr[j][1] - tr[j][0],) + fl(*tr[j]) for j in comb))
                (ok if E.is_canonical(y) else ab).append((key, key2, P.scores(y)))
            len_ab += [np.mean(kk) for kk, _, _ in ab]; len_ok += [np.mean(kk) for kk, _, _ in ok]
            if not (ab and ok):
                continue
            row_ = {}
            for tag, ki in (("", 0), ("_flank", 1)):
                per = {nm: [] for nm in ("nMSI", "MSI", "G4Hunter")}
                for va in ab:
                    same = np.array([vo[2] for vo in ok if vo[ki] == va[ki]])
                    if same.size:
                        for j, nm in enumerate(("nMSI", "MSI", "G4Hunter")):
                            per[nm].append(q(same[:, j], va[2][j]))
                if per["nMSI"]:
                    row_.update({nm + tag: np.nanmean(v) for nm, v in per.items()})
            if "nMSI" in row_:
                res.append(row_)
        R = pd.DataFrame(res)
        row = dict(set=label, design=f"G→{base}, {where}, {k} tract{'s' if k > 1 else ''}", wild_types=len(R),
                   mean_hit_length_abolishing=float(np.mean(len_ab)), mean_hit_length_retaining=float(np.mean(len_ok)),
                   wild_types_flank=int(R["nMSI_flank"].notna().sum()) if "nMSI_flank" in R else 0)
        for nm in ("nMSI", "MSI", "G4Hunter", "nMSI_flank", "MSI_flank", "G4Hunter_flank"):
            v = R[nm].dropna().to_numpy() if nm in R else np.array([])
            if v.size:
                bs = [np.mean(rng.choice(v, v.size)) for _ in range(2000)]
                row.update({f"{nm}_mean_q": float(v.mean()), f"{nm}_lo": float(np.percentile(bs, 2.5)),
                            f"{nm}_hi": float(np.percentile(bs, 97.5))})
        rows.append(row); print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="both", choices=["pooled", "a549", "both"])
    a = ap.parse_args()
    if a.set in ("pooled", "both"):
        seqs, _ = E.load()
        analyse(seqs, "pooled 538").to_csv(os.path.join(ROOT, "results", "additional", "matched_placement_length.csv"), index=False)
    if a.set in ("a549", "both"):
        import a549_full_mutagenesis as A
        analyse(list(A.load().seq), "A549 eligible").to_csv(
            os.path.join(ROOT, "results", "a549_full", "matched_placement_length.csv"), index=False)


if __name__ == "__main__":
    main()
