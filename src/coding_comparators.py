#!/usr/bin/env python3
"""
coding_comparators.py
=====================

The two comparator statistics used against the parity index in Figure 5A of

    "A Closed-Form Expectation for Intra-Strand Parity Indices and the Limits of Their Use as G-Quadruplex Descriptors"  (Arya, Pancholi & Datta)

  (1) FICKETT'S POSITION-ASYMMETRY STATISTIC, the positional component of the
      TESTCODE measure of Fickett, J.W. (1982) Nucleic Acids Res., 10,
      5303-5318. For each base b the sequence is read in steps of three from
      each of the three offsets, giving counts n0(b), n1(b), n2(b), and the
      position value is

          A(b) = max(n0, n1, n2) / (min(n0, n1, n2) + 1)

      exactly as defined in that paper. Coding sequence concentrates particular
      bases at particular codon positions, so A(b) is larger in coding than in
      non-coding sequence of the same length.

      NOTE ON SCOPE. This module implements the position-asymmetry component
      only, which is what the manuscript compares against; it does NOT
      implement full TESTCODE, whose final score combines four position values
      and four composition values through the lookup tables of probabilities
      and weights published in Fickett's Table 1. Those tables are not
      reproduced here, so no TESTCODE probability is computed and none should
      be reported from this code.

  (2) A HEXAMER LOG-LIKELIHOOD RATIO of the kind used in modern coding-potential
      tools, after Wang, L. et al. (2013) Nucleic Acids Res., 41, e74 (CPAT).
      In-frame hexamer frequencies are estimated separately on coding and on
      non-coding training sequences with add-one smoothing, and a sequence is
      scored by the mean log ratio over its in-frame hexamers. The model is
      fitted on training folds only and scored out of fold.

VALIDATION
----------
Neither statistic has published per-sequence reference values of the kind used
elsewhere in this work to validate G4Hunter (1.71 for the human telomeric
repeat, 2.59 for c-MYC Pu27). Fickett's paper reports classification accuracy
on his own test set rather than per-sequence values, and CPAT reports AUCs on
its training and test sets. Both implementations here are therefore validated
by deterministic unit tests whose expected values follow from the definitions,
and by the qualitative behaviour each statistic is published as having. Run

    python3 src/coding_comparators.py --selftest

which prints every test and its expected value.

Authors: Arjun Arya, Heli Pancholi, Bhaskar Datta
License: MIT
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np

BASES = "ACGT"


# --------------------------------------------------------------------------- #
# (1) Fickett's position asymmetry
# --------------------------------------------------------------------------- #
def fickett_position_values(seq: str) -> Dict[str, float]:
    """A(b) = max(n0,n1,n2) / (min(n0,n1,n2) + 1) for each base b, where nk(b)
    counts occurrences of b at sequence positions congruent to k modulo 3.

    Fickett (1982), the 'position' parameter. The +1 in the denominator is his,
    and keeps the value finite when a base is absent from one position."""
    s = seq.upper()
    out: Dict[str, float] = {}
    for b in BASES:
        n = [0, 0, 0]
        for k in range(3):
            n[k] = s[k::3].count(b)
        out[b] = max(n) / (min(n) + 1)
    return out


def fickett_asymmetry(seq: str, combine: str = "mean") -> float:
    """Single position-asymmetry score for a sequence.

    combine='mean' averages the four base-wise position values; combine='max'
    takes the largest. The manuscript's Figure 5A uses the mean. Either choice
    is monotone in the same underlying quantity, but they are not identical, so
    state which is used when reporting a value."""
    v = fickett_position_values(seq)
    vals = [v[b] for b in BASES]
    if combine == "mean":
        return float(np.mean(vals))
    if combine == "max":
        return float(np.max(vals))
    raise ValueError("combine must be 'mean' or 'max'")


# --------------------------------------------------------------------------- #
# (2) Hexamer log-likelihood ratio
# --------------------------------------------------------------------------- #
class HexamerModel:
    """In-frame hexamer log-likelihood ratio, coding against non-coding.

    fit()   estimates hexamer frequencies on the two training sets with
            add-one smoothing over the 4096 hexamers.
    score() returns the mean log ratio over the in-frame hexamers of a
            sequence; higher means more coding-like."""

    def __init__(self, pseudocount: float = 1.0):
        self.pseudocount = float(pseudocount)
        self.logratio: Dict[str, float] | None = None

    @staticmethod
    def _inframe_hexamers(seq: str) -> List[str]:
        s = seq.upper()
        return [s[i:i + 6] for i in range(0, len(s) - 5, 3)
                if all(c in BASES for c in s[i:i + 6])]

    def fit(self, coding: Iterable[str], noncoding: Iterable[str]) -> "HexamerModel":
        words = [a + b + c + d + e + f
                 for a in BASES for b in BASES for c in BASES
                 for d in BASES for e in BASES for f in BASES]
        cc, nn = Counter(), Counter()
        for s in coding:
            cc.update(self._inframe_hexamers(s))
        for s in noncoding:
            nn.update(self._inframe_hexamers(s))
        pc = self.pseudocount
        tc = sum(cc.values()) + pc * len(words)
        tn = sum(nn.values()) + pc * len(words)
        self.logratio = {w: math.log(((cc[w] + pc) / tc) / ((nn[w] + pc) / tn))
                         for w in words}
        return self

    def score(self, seq: str) -> float:
        if self.logratio is None:
            raise RuntimeError("fit() the model before scoring")
        h = self._inframe_hexamers(seq)
        if not h:
            return float("nan")
        return float(np.mean([self.logratio[w] for w in h]))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def hexamer_cv_auc(coding: Sequence[str], noncoding: Sequence[str],
                   folds: int = 5, seed: int = 0) -> float:
    """Out-of-fold AUC of the hexamer log-likelihood ratio, coding against
    non-coding. Fold assignment is by sequence, so no sequence contributes to
    the model that scores it."""
    rng = np.random.default_rng(seed)
    cod = list(coding); nc = list(noncoding)
    fc = rng.integers(0, folds, len(cod))
    fn = rng.integers(0, folds, len(nc))
    scores, labels = [], []
    for k in range(folds):
        tr_c = [s for s, f in zip(cod, fc) if f != k]
        tr_n = [s for s, f in zip(nc, fn) if f != k]
        te_c = [s for s, f in zip(cod, fc) if f == k]
        te_n = [s for s, f in zip(nc, fn) if f == k]
        if not tr_c or not tr_n or (not te_c and not te_n):
            continue
        m = HexamerModel().fit(tr_c, tr_n)
        for s in te_c:
            v = m.score(s)
            if not math.isnan(v): scores.append(v); labels.append(1)
        for s in te_n:
            v = m.score(s)
            if not math.isnan(v): scores.append(v); labels.append(0)
    if not scores:
        return float("nan")
    return _roc_auc(np.asarray(scores), np.asarray(labels))


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest() -> int:
    ok = True

    def check(name, got, want, tol=1e-9):
        nonlocal ok
        good = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
        ok &= bool(good)
        print(f"[{'OK ' if good else 'FAIL'}] {name}: got {got}, expected {want}")

    print("Fickett position asymmetry — definition A(b) = max(n)/(min(n)+1)")
    # A sequence in which G sits only at position 0 modulo 3.
    s = "GAT" * 10                      # G at offset 0, A at 1, T at 2
    v = fickett_position_values(s)
    check("A(G) for (GAT)x10", v["G"], 10 / 1)
    check("A(A) for (GAT)x10", v["A"], 10 / 1)
    check("A(C) for (GAT)x10", v["C"], 0.0)
    # Uniform sequence: every base 10 times at every codon position -> 10/(10+1)
    s2 = "ACGT" * 30                     # length 120, each base 10 times per offset
    v2 = fickett_position_values(s2)
    check("A(A) for (ACGT)x30", v2["A"], 10 / 11)
    check("mean of the four base values for (GAT)x10",
          fickett_asymmetry("GAT" * 10, "mean"), (10 + 10 + 0.0 + 10) / 4)

    print("\nFickett published behaviour — positional bias is higher in coding-like sequence")
    rng = np.random.default_rng(0)
    codons = ["ATG"] + ["".join(rng.choice(list("ACGT"), 2)) + "G" for _ in range(200)]
    coding_like = "".join(codons)                       # third position fixed to G
    random_like = "".join(rng.choice(list("ACGT"), len(coding_like)))
    a_cod = fickett_asymmetry(coding_like)
    a_non = fickett_asymmetry(random_like)
    print(f"[{'OK ' if a_cod > a_non else 'FAIL'}] coding-like {a_cod:.3f} > random {a_non:.3f}")
    ok &= a_cod > a_non

    print("\nHexamer model — log ratio, add-one smoothing")
    m = HexamerModel().fit(["ATGAAAATGAAA"], ["TTTTTTTTTTTT"])
    lr_a = m.logratio["ATGAAA"]; lr_t = m.logratio["TTTTTT"]
    print(f"[{'OK ' if lr_a > 0 > lr_t else 'FAIL'}] ATGAAA scores positive "
          f"({lr_a:.3f}) and TTTTTT negative ({lr_t:.3f})")
    ok &= (lr_a > 0 > lr_t)
    m2 = HexamerModel().fit(["TTTTTTTTTTTT"], ["ATGAAAATGAAA"])
    sym = abs(m2.logratio["ATGAAA"] + m.logratio["ATGAAA"])
    check("swapping the classes negates the log ratio", round(sym, 12), 0.0, 1e-9)

    print("\nHexamer model — out-of-fold discrimination on synthetic sequences")
    rng = np.random.default_rng(7)
    w = np.array([3.0 if c[2] in "GC" else 1.0
                  for c in ["".join(x) for x in np.array(
                      np.meshgrid(*[list("ACGT")] * 3)).T.reshape(-1, 3)]])
    w = w / w.sum()
    sense = ["".join(x) for x in np.array(np.meshgrid(*[list("ACGT")] * 3)).T.reshape(-1, 3)]
    cod = ["".join(rng.choice(sense, 120, p=w)) for _ in range(60)]
    non = ["".join(rng.choice(list("ACGT"), 360)) for _ in range(60)]
    auc = hexamer_cv_auc(cod, non, folds=5, seed=1)
    print(f"[{'OK ' if auc > 0.9 else 'FAIL'}] out-of-fold AUC on biased vs random: {auc:.3f} (> 0.9)")
    ok &= auc > 0.9

    print("\n[selftest] ALL PASSED" if ok else "\n[selftest] FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)
