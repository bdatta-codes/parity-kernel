#!/usr/bin/env python3
"""ref21_pairs.py -- the 12-of-14 wild-type/mutant result of ref. 21 tested
against its own per-pair nulls.

data/ref21_wt_mutant_pairs.tsv holds the 14 promoter quadruplexes of ref. 21 and
the G4-abolished mutants with which they were compared there (Table 6 of ref. 21;
every mutant is a multi-site substitution of the same length as its wild type).
For each pair, MSI and nMSI of both members, and the paired nulls of
experiment_538.paired_null: the aligned wild type and mutant shuffled together,
uniformly (2,000 draws) or by Altschul-Erickson over the pair alphabet (1,000
draws), giving g = P(null wild type scores higher) and t = P(tie). The observed
concordance (ties one half) is referred to the exact three-outcome tail
(experiment_538.three_outcome_sf) and, for comparison, to a fair coin.

Usage:  python3 src/ref21_pairs.py
Writes results/si/ref21_pairs.csv, results/si/ref21_pairs.json and
results/supplementary_tables/TableS1_ref21_pairs.csv
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import experiment_538 as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "si"); TAB = os.path.join(ROOT, "results", "supplementary_tables")
K_MONO, K_DI = 2000, 1000


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    d = pd.read_csv(os.path.join(ROOT, "data", "ref21_wt_mutant_pairs.tsv"), sep="\t", comment=None)
    d.columns = ["gene", "wild_type", "mutant", "design"]
    rm, rd = np.random.default_rng(20261027), np.random.default_rng(20261028)
    rows = []
    for _, r in d.iterrows():
        xw, xm = E.enc(r.wild_type), E.enc(r.mutant)
        assert xw.size == xm.size
        w, m = E.idx_msi(xw), E.idx_msi(xm)
        a = E.paired_null(xw, xm, K_MONO, "mono", rm); b = E.paired_null(xw, xm, K_DI, "dinuc", rd)
        rows.append(dict(gene=r.gene, length=xw.size, substitutions=int((xw != xm).sum()),
                         wt_MSI=w[0], mut_MSI=m[0], wt_nMSI=w[1], mut_nMSI=m[1],
                         g_MSI_mono=a["MSI"][0], t_MSI_mono=a["MSI"][1],
                         g_nMSI_mono=a["nMSI"][0], t_nMSI_mono=a["nMSI"][1],
                         g_MSI_di=b["MSI"][0], t_MSI_di=b["MSI"][1],
                         g_nMSI_di=b["nMSI"][0], t_nMSI_di=b["nMSI"][1]))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "ref21_pairs.csv"), index=False)
    summ = {}
    for idx in ("MSI", "nMSI"):
        dlt = R[f"wt_{idx}"] - R[f"mut_{idx}"]
        win, tie = int((dlt > 1e-12).sum()), int(np.isclose(dlt, 0).sum()); lose = len(R) - win - tie
        k = win + 0.5 * tie
        s = dict(concordant=k, wins=win, ties=tie,
                 p_fair=float(stats.binomtest(win, win + lose, 0.5, alternative="greater").pvalue))
        for null in ("mono", "di"):
            g, t = R[f"g_{idx}_{null}"].values, R[f"t_{idx}_{null}"].values
            s[f"null_{null}"] = float(np.mean(g + t / 2)); s[f"p_{null}"] = E.three_outcome_sf(k, g, t)
            # calibration and power: expected concordance under the null, and the
            # smallest tail any outcome could reach (all 14 wild types higher)
            s[f"expected_concordant_{null}"] = float(np.sum(g + t / 2))
            s[f"p_at_13_{null}"] = E.three_outcome_sf(13, g, t)
            s[f"p_at_14_{null}"] = E.three_outcome_sf(14, g, t)
            # sensitivity, as Supplementary Table S10: every g displaced together by
            # +/-0.05 (kept within [0, 1 - t]), and each perturbed independently by
            # N(0, 0.025^2), 1,000 times (seed 20261129)
            for dl in (-0.05, 0.05):
                gs = np.clip(g + dl, 0, 1 - t)
                s[f"p_{null}_shift{dl:+.2f}"] = E.three_outcome_sf(k, gs, t)
            rs = np.random.default_rng(20261129)
            ps = np.array([E.three_outcome_sf(k, np.clip(g + rs.normal(0, 0.025, g.size), 0, 1 - t), t)
                           for _ in range(1000)])
            s[f"p_{null}_indep_q025"] = float(np.quantile(ps, .025)); s[f"p_{null}_indep_q975"] = float(np.quantile(ps, .975))
            s[f"p_{null}_verdict_flips_pct"] = 100 * float(np.mean((ps < 0.05) != (s[f"p_{null}"] < 0.05)))
        summ[idx] = s
    json.dump(summ, open(os.path.join(OUT, "ref21_pairs.json"), "w"), indent=1)
    T = pd.DataFrame({"Gene": R.gene, "Length (nt)": R.length, "Substituted positions": R.substitutions,
                      "MSI wild type / mutant": [f"{a:.3f} / {b:.3f}" for a, b in zip(R.wt_MSI, R.mut_MSI)],
                      "MSI paired null, mono / dinuc": [f"{g + t / 2:.3f} / {g2 + t2 / 2:.3f}" for g, t, g2, t2 in
                                                        zip(R.g_MSI_mono, R.t_MSI_mono, R.g_MSI_di, R.t_MSI_di)],
                      "nMSI wild type / mutant": [f"{a:.3f} / {b:.3f}" for a, b in zip(R.wt_nMSI, R.mut_nMSI)],
                      "nMSI paired null, mono / dinuc": [f"{g + t / 2:.3f} / {g2 + t2 / 2:.3f}" for g, t, g2, t2 in
                                                         zip(R.g_nMSI_mono, R.t_nMSI_mono, R.g_nMSI_di, R.t_nMSI_di)]})
    T.to_csv(os.path.join(TAB, "TableS1_ref21_pairs.csv"), index=False)
    print(T.to_string(index=False)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
