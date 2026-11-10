#!/usr/bin/env python3
"""substitution_scan.py — every single-base substitution (G->A, G->C, G->T) at
every position of every guanine tract (G{3,}) of the 538 sequences of the
mutagenesis set (experiment_538.load). Writes results/mutagenesis_scan.csv with
one row per substitution; delta = nMSI(mutant) - nMSI(wild type), so delta > 0
means the substitution RAISES the index. Supplementary Table S19 and Figure S3.

License: MIT
"""
import os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_538 as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    seqs, src = E.load()
    rows = []
    for i, (s, so) in enumerate(zip(seqs, src)):
        wt = E.idx_msi(E.enc(s))[1]
        for t, m in enumerate(re.finditer("G{3,}", s)):
            a, b = m.start(), m.end()
            for p in range(a, b):
                pos = "5' edge" if p == a else ("3' edge" if p == b - 1 else "interior")
                for base in "ACT":
                    mut = s[:p] + base + s[p + 1:]
                    mn = E.idx_msi(E.enc(mut))[1]
                    rows.append(dict(seq=i, source=so, tract=t, tract_len=b - a,
                                     offset=p - a, position=pos, base=base,
                                     wt_nMSI=wt, mut_nMSI=mn, delta=mn - wt))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(ROOT, "results", "mutagenesis_scan.csv"), index=False)
    print(f"{len(seqs)} sequences, {S.groupby(['seq','tract']).ngroups} tracts, {len(S)} substitutions")
    for lab, g in [("all", S)] + list(S.groupby("position")):
        print(f"  {lab:9s} raise {100*(g.delta>0).mean():.1f}%  unchanged {100*(g.delta==0).mean():.1f}%  lower {100*(g.delta<0).mean():.1f}%")


if __name__ == "__main__":
    main()
