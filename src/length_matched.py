#!/usr/bin/env python3
"""length_matched.py - remove the length confound between the two halves of the
paper.

The mutagenesis sequences are short (median 25.5 nt; 98% shorter than the
51 nt at which the k = 3 baseline reaches 0.50), whereas the coding analysis
uses whole coding sequences of 120 to 16,566 nt. Two comparisons put the two
features at the same length.

Part 1 (coding at quadruplex lengths; needs the AUGUSTUS GenBank files):
for fragment lengths of 8, 12, 16, 20, 30 and 60 codons (24 to 180 nt), every
coding sequence at least that long (all 1,076 up to 30 codons) contributes one in-frame fragment at a random
codon offset, paired with a non-coding window of the same length cut at a
random offset from a random non-coding piece long enough; repeated over 200
draws. AUC (mean and 2.5-97.5% range over draws) for the frame-resolved index,
the frame-free index and Fickett's statistic, scored as in Figure 5A.

Part 2 (quadruplexes above the minimum length; needs the chr21 FASTA): the
133,601 clean 300-nt windows of results/chr21/windows300.csv.gz are labelled by
whether they contain the permissive two-tract motif (as in Table 5) or, more
strictly, the canonical consensus G3+N1-7 x4 on either strand. Windows with and
without a motif are matched on GC content (0.01 bins, equal numbers drawn at
random per bin), and the AUC of nMSI, of the symmetry deficit and of |G4Hunter|
(the positive control) is computed on the matched sets, 50 random matchings.

    python3 src/length_matched.py --genbank data/fly2L.gb data/fly2R.gb \\
        --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz

Seeds 20261106 (part 1) and 20261107 (part 2). Writes
results/additional/length_matched_coding.csv and length_matched_chr21.csv.

License: MIT
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coding_sequence_analysis as C
from coding_comparators import fickett_asymmetry
import parity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "additional"); os.makedirs(OUT, exist_ok=True)


def auc(score, y):
    ok = np.isfinite(score)
    return C.roc_auc(score[ok], y[ok])


def part1(gb, n_draws=200):
    rng = np.random.default_rng(20261106)
    coding, noncoding = C.build_sets_genbank(gb, 120)
    rows = []
    for Lc in (8, 12, 16, 20, 30, 60):
        L = 3 * Lc
        pieces = [p for p in noncoding if len(p) >= L]
        usable = [s for s in coding if len(s) // 3 >= Lc]
        acc = {"frame_resolved": [], "frame_free": [], "fickett": []}
        for _ in range(n_draws):
            frag = []
            for s in usable:
                nc = len(s) // 3
                o = int(rng.integers(0, nc - Lc + 1)) * 3
                frag.append(s[o:o + L])
            neg = []
            for _ in usable:
                p = pieces[int(rng.integers(len(pieces)))]
                o = int(rng.integers(0, len(p) - L + 1)); neg.append(p[o:o + L])
            allseq = frag + neg
            y = np.array([1] * len(frag) + [0] * len(neg))
            fr = np.array([C.nmsi_frame0(x) for x in allseq])
            ff = np.array([C.nmsi_framefree(x) for x in allseq])
            fk = np.array([fickett_asymmetry(x) for x in allseq])
            acc["frame_resolved"].append(auc(1 - fr, y))
            acc["frame_free"].append(auc(1 - ff, y))
            acc["fickett"].append(auc(fk, y))
        row = dict(codons=Lc, nt=L, n_coding=len(usable), n_draws=n_draws)
        for k, v in acc.items():
            v = np.array(v); row[f"auc_{k}"] = v.mean()
            row[f"auc_{k}_lo"] = np.percentile(v, 2.5); row[f"auc_{k}_hi"] = np.percentile(v, 97.5)
        rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "length_matched_coding.csv"), index=False)


QP = re.compile(r"(?=(G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}))")
QPC = re.compile(r"(?=(C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}))")


def part2(fasta, n_match=50):
    import chromosome21_windows as W
    rng = np.random.default_rng(20261107)
    seq = W.read_single_fasta(fasta).upper()
    w = pd.read_csv(os.path.join(ROOT, "results", "chr21", "windows300.csv.gz"))
    win = [seq[s:s + 300] for s in w.start]
    w["consensus"] = [bool(QP.search(x) or QPC.search(x)) for x in win]
    w["g4h_abs"] = [abs(parity.g4hunter(x)) for x in win]
    w["permissive"] = w.g4_coverage > 0
    w["gcbin"] = (w.gc * 100).round().astype(int)
    rows = []
    for lab in ("permissive", "consensus"):
        res = {"nMSI": [], "deficit": [], "G4Hunter": []}
        for _ in range(n_match):
            pos, neg = [], []
            for _, g in w.groupby("gcbin"):
                a = g.index[g[lab]].values; b = g.index[~g[lab]].values
                k = min(len(a), len(b))
                if k:
                    pos += list(rng.choice(a, k, replace=False)); neg += list(rng.choice(b, k, replace=False))
            idx = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            res["nMSI"].append(auc(w.nmsi.values[idx], y))
            res["deficit"].append(auc(w.deficit.values[idx], y))
            res["G4Hunter"].append(auc(w.g4h_abs.values[idx], y))
        row = dict(label=lab, n_with=int(w[lab].sum()), n_matched_each=len(pos))
        for k, v in res.items():
            v = np.array(v); row[f"auc_{k}"] = v.mean()
            row[f"auc_{k}_lo"] = np.percentile(v, 2.5); row[f"auc_{k}_hi"] = np.percentile(v, 97.5)
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "length_matched_chr21.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--genbank", nargs="+")
    ap.add_argument("--chr21")
    a = ap.parse_args()
    if a.chr21:
        part2(a.chr21)
    if a.genbank:
        part1(a.genbank)
