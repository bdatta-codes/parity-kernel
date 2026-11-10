#!/usr/bin/env python3
"""placement_null.py - does an index register the abolition of a quadruplex
beyond the guanine it removes?

Composition-matched folding versus non-folding contrast. For every verified
wild-type/abolished-mutant pair of the mutagenesis experiment (pairs_*.csv),
the same substitutions (same number, same new bases) are placed instead at
guanines chosen uniformly at random from the wild type, and the resulting
"placement-null" mutant is kept only if it still satisfies the Quadparser
consensus, i.e. it is predicted to fold. The real mutant and the kept null
mutants then have identical base composition and differ only in where the
guanines were removed: the real mutant is abolished, the null mutants are not.
CAUTION: because null mutants must stay consensus-positive, they hit loops and
fewer tracts than the real mutant (which hits every tract), so this contrast
confounds abolition with placement; matched_placement.py removes that
confound. The G4Hunter "pass" here reflects the placement difference.

For each pair and index f, q = P(f(null mutant) > f(real mutant)) + P(tie)/2
over K kept, distinct null mutants (duplicate placements are discarded). An index that registers abolition beyond guanine loss
ranks the folding null mutants above the abolished real mutant (q > 1/2);
one that does not gives q near 1/2. G4Hunter is the positive control: it must
pass this test for the test to be informative.

Reported per convention: number of pairs with at least one kept null mutant,
mean q with a 95% bootstrap interval over pairs (2,000 resamples), and a
Wilcoxon signed-rank test of q - 1/2. K = 100 kept null mutants per pair, at
most 4,000 attempts; seed 20261105. Writes results/additional/placement_null.csv.

License: MIT
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "additional"); os.makedirs(OUT, exist_ok=True)
K, MAX_TRY = 100, 4000
CONV = [("T_centre", "G→T, tract centre"), ("A_centre", "G→A, tract centre"),
        ("C_centre", "G→C, tract centre"), ("two_tract", "G→T, two tracts only"),
        ("tract_ablate", "first tract replaced by thymine"), ("T_3prime", "G→T, 3′ edge"),
        ("T_5prime", "G→T, 5′ edge")]


def scores(s):
    x = E.enc(s); msi, nmsi = E.idx_msi(x)
    return nmsi, msi, E.g4hunter(x)


def q(null_vals, real):
    a = np.asarray(null_vals, float); a = a[np.isfinite(a)]
    if a.size == 0 or not np.isfinite(real):
        return np.nan
    return float(np.mean(a > real + 1e-12) + 0.5 * np.mean(np.isclose(a, real)))


def main():
    rng = np.random.default_rng(20261105)
    seqs, _ = E.load()
    rows, per_pair = [], []
    for key, label in CONV:
        d = pd.read_csv(os.path.join(RES, f"pairs_{key}.csv"))
        qs = {"nMSI": [], "MSI": [], "G4Hunter": []}
        kept_counts = []
        for _, r in d.iterrows():
            s = seqs[int(r.i)]
            m = E.mutate(s, key, np.random.default_rng(0))     # deterministic for designed conventions
            diff = [p for p in range(len(s)) if s[p] != m[p]]
            new = [m[p] for p in diff]
            gpos = np.array([p for p, c in enumerate(s) if c == "G"])
            real = scores(m)
            null = []; seen = set()
            tries = 0
            while len(null) < K and tries < MAX_TRY:
                tries += 1
                pos = rng.choice(gpos, size=len(diff), replace=False)
                if set(pos.tolist()) == set(diff):
                    continue
                y = list(s)
                for p, b in zip(pos, rng.permutation(new)):
                    y[p] = b
                y = "".join(y)
                if y not in seen and E.is_canonical(y):
                    seen.add(y); null.append(scores(y))
            kept_counts.append(len(null))
            if not null:
                continue
            nv = np.array(null)
            pp = dict(convention=label, i=int(r.i), kept=len(null))
            for j, nm in enumerate(("nMSI", "MSI", "G4Hunter")):
                v = q(nv[:, j], real[j]); qs[nm].append(v); pp[f"q_{nm}"] = v
            per_pair.append(pp)
        row = dict(convention=label, pairs=len(d), pairs_with_null=int(np.sum(np.array(kept_counts) > 0)),
                   median_kept=float(np.median(kept_counts)))
        for nm, v in qs.items():
            v = np.array(v, float); v = v[np.isfinite(v)]
            bs = [np.mean(rng.choice(v, v.size)) for _ in range(2000)]
            row[f"{nm}_mean_q"] = float(v.mean())
            row[f"{nm}_lo"] = float(np.percentile(bs, 2.5)); row[f"{nm}_hi"] = float(np.percentile(bs, 97.5))
            row[f"{nm}_wilcoxon_p"] = float(stats.wilcoxon(v - 0.5).pvalue) if np.any(v != 0.5) else 1.0
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "placement_null.csv"), index=False)
    pd.DataFrame(per_pair).to_csv(os.path.join(OUT, "placement_null_per_pair.csv"), index=False)


if __name__ == "__main__":
    main()
