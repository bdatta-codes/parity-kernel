#!/usr/bin/env python3
"""
chromosome21_windows.py
=======================

The human chromosome 21 window analyses of the paper

    "A Closed-Form Expectation for Intra-Strand Parity Indices and the Limits of
     Their Use as G-Quadruplex Descriptors"
     (Arya, Pancholi & Datta)

This reproduces every chromosome-21 number in the Results:

  (1) TABLE 4 / FIGURE 6A  -- observed nMSI on 700 random clean windows at each
      of six window lengths (60, 120, 300, 600, 1200, 3000 nt), each compared
      against the analytic sliding-window baseline for that window's OWN length
      and base composition, reported as a mean z-score. Real DNA is
      consistently less mirror-symmetric than an independent-base model
      predicts, and the gap grows with window length (~ -11 sd at 3 kb).

  (2) FIGURE 6B  -- across all 133,601 clean, non-overlapping 300-nt windows,
      the symmetry deficit (predicted - observed nMSI) correlated with GC
      content (rho = +0.409) and with CpG under-representation (+0.157); the
      CpG term survives holding GC constant (partial rho = +0.322).

  (3) TABLE 5 / FIGURE 6C  -- across the same 133,601 windows, 48% of which
      carry a permissive two-tract quadruplex motif, quadruplex density and
      mirror symmetry correlate at rho = -0.248; holding GC content constant
      leaves a weak residual of the opposite sign (partial rho = +0.069).

DATA AND DEFINITIONS
--------------------
The paper uses chromosome 21 of the GRCh38 primary assembly as distributed by
Ensembl (Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz, 46,709,983 bp; the
sequence is identical in every Ensembl release of GRCh38). It is not
redistributed here:

    python3 src/chromosome21_windows.py \\
        --chr21 Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz --out results/chr21/

  * Clean sequence: maximal runs of A, C, G, T; a window is clean only if it
    contains no N. Repeats are NOT excluded by default (the Ensembl 'dna' file
    carries no soft-masking); with a soft-masked FASTA, --mask-repeats treats
    lower-case bases as unclean.
  * Table 4 / Figure 6A: 700 windows per length at random start positions
    within clean sequence (windows may overlap).
  * Figure 6B and Table 5 / Figure 6C: every non-overlapping 300-nt window
    tiled within the clean runs of the whole chromosome.
  * Quadruplex density: fraction of a window's positions covered by a
    permissive two-guanine-tract motif, (G{2,}N{1,7}){3}G{2,}, on either
    strand (its C-run equivalent on the given strand).
  * CpG under-representation: (f(GpC) - f(CpG)) / (f(GpC) + f(CpG)).
  * Partial correlation: Spearman correlation between the residuals of the
    ranked variables after linear regression on the ranked control.

  * Baseline SD for z-scores: 400 independent-base sequences simulated at the
    window's length and at its base composition rounded to 0.01 (cached per
    rounded composition).

Dependencies: numpy, scipy, pandas, biopython (for gzipped-FASTA reading only;
a plain .fa is read without Biopython).

Authors: Arjun Arya, Heli Pancholi, Bhaskar Datta
         Indian Institute of Technology Gandhinagar
License: MIT
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
from typing import Iterable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity  # noqa: E402

# The sliding-window correction constant from the paper: C' = C - kappa = 3.899
# (Eq. 4; kappa fitted once near uniform composition, value 0.615). The
# composition-dependent refit of kappa is discussed in the paper (Table 2); for
# the chromosome-21 comparison the paper uses the near-uniform sliding-window
# constant, which is exactly what makes the genomic overshoot visible.
KAPPA_UNIFORM = 0.615


# --------------------------------------------------------------------------- #
# Sliding-window baseline for a window of length L at its own composition
# --------------------------------------------------------------------------- #
def nmsi_baseline_sliding(L: int, pA: float, pC: float, pG: float, pT: float,
                          kappa: float = KAPPA_UNIFORM) -> float:
    """Predicted nMSI for a length-L sliding-window scan at the given base
    composition.

    A sliding window over a sequence of length L counts N = L - 2 overlapping
    trinucleotides. The composition-dependent EXACT closed form
    (parity.nmsi_baseline_exact) gives the expectation for N *non-overlapping*
    draws; the sharing of two letters between neighbouring windows is absorbed
    by the single fitted constant kappa, added as kappa/sqrt(N) (Eq. 4). This
    is exactly the baseline used in the repository quick-start
    (`nmsi_baseline_exact(N) + 0.615/sqrt(N)`); it is the predicted baseline
    of Table 4.

    Near uniform composition kappa = 0.615. On short, GC-rich, G-skewed windows
    the correction should strictly be refitted at the composition of interest
    (paper Table 2); the paper reports Table 4 against this near-uniform
    constant, which is what makes the genomic overshoot visible.
    """
    N = L - 2
    if N <= 0:
        return float("nan")
    return parity.nmsi_baseline_exact(N, pA, pC, pG, pT) + kappa / np.sqrt(N)


# Monte Carlo baseline SDs are reused across windows of the same length and
# near-identical composition. Without this cache a full Table 4 pass (700
# windows at each of six lengths) regenerates n_ref random sequences per
# window, about eight minutes of redundant simulation; with it, under a
# second.
_SD_CACHE: dict = {}


def nmsi_baseline_sd(L: int, n_ref: int = 400, seed: int = 0,
                     pACGT: Tuple[float, float, float, float] | None = None) -> float:
    """Baseline standard deviation of nMSI at length L, estimated from Monte
    Carlo draws at the given composition (uniform if pACGT is None). Used to
    turn each window's (observed - predicted) into a z-score.

    Results are memoised on (L, composition rounded to two decimals, n_ref,
    seed); the baseline SD varies smoothly with composition, so rounding costs
    nothing measurable and makes the per-window loop tractable."""
    p = np.array(pACGT if pACGT is not None else (.25, .25, .25, .25), dtype=float)
    p = p / p.sum()
    key = (L, tuple(np.round(p, 2)), n_ref, seed)
    if key in _SD_CACHE:
        return _SD_CACHE[key]
    # an independent stream for each length and (rounded) composition, so that
    # the SD estimates of different windows do not share their Monte Carlo error
    rng = np.random.default_rng([seed, L] + [int(round(100 * q)) for q in key[1]] + [7])
    bases = np.array(list("ACGT"))
    vals = []
    for _ in range(n_ref):
        s = "".join(rng.choice(bases, size=L, p=p))
        _, nm = parity.msi_scores(s)
        if nm is not None:
            vals.append(nm)
    out = float(np.std(vals)) if vals else float("nan")
    _SD_CACHE[key] = out
    return out


# --------------------------------------------------------------------------- #
# FASTA reading and clean-window sampling
# --------------------------------------------------------------------------- #
def read_single_fasta(path: str) -> str:
    """Read a one-record FASTA (chr21). Preserves case so soft-masking can be
    used to exclude repeats.  Handles .gz transparently."""
    opener = gzip.open if path.endswith(".gz") else open
    seq_parts: List[str] = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if seq_parts:
                    # stop after the first record
                    break
                continue
            seq_parts.append(line.strip())
    return "".join(seq_parts)


def _acgt_fraction(win: str) -> float:
    """Fraction of the window that is unambiguous ACGT. The sequence is
    upper-cased on reading unless --mask-repeats is given, in which case
    lower-case (soft-masked) bases count as unclean."""
    if not win:
        return 0.0
    good = sum(1 for c in win if c in "ACGT")
    return good / len(win)


def clean_tiles(seq: str, L: int = 300):
    """Start positions of every non-overlapping L-nt window tiled from the
    start of each maximal clean (ACGT) run, in chromosome order."""
    import re
    out = []
    for m in re.finditer("[ACGT]+", seq):
        for j in range(0, m.end() - m.start() - L + 1, L):
            out.append(m.start() + j)
    return out


def sample_clean_windows(seq: str, L: int, n: int, min_acgt: float,
                         rng: np.random.Generator,
                         max_tries_factor: int = 200) -> List[str]:
    """Sample n random windows of length L whose ACGT (unmasked) fraction is at
    least min_acgt.  Random start positions, sampled without replacement of the
    exact start; windows may overlap, as in the paper's random-window design."""
    S = len(seq)
    out: List[str] = []
    tries = 0
    max_tries = n * max_tries_factor
    seen_starts = set()
    while len(out) < n and tries < max_tries:
        tries += 1
        start = int(rng.integers(0, S - L + 1))
        if start in seen_starts:
            continue
        seen_starts.add(start)
        win = seq[start:start + L]
        if _acgt_fraction(win) >= min_acgt:
            out.append(win.upper())
    if len(out) < n:
        sys.stderr.write(
            f"[warn] L={L}: only {len(out)} clean windows found in {tries} tries "
            f"(wanted {n}); result may differ from the paper.\n")
    return out


# --------------------------------------------------------------------------- #
# Composition helpers
# --------------------------------------------------------------------------- #
def base_probs(s: str) -> Tuple[float, float, float, float]:
    s = s.upper()
    n = sum(c in "ACGT" for c in s) or 1
    return (s.count("A") / n, s.count("C") / n, s.count("G") / n, s.count("T") / n)



def cpg_underrepresentation(s: str) -> float:
    """(f(GpC) - f(CpG)) / (f(GpC) + f(CpG)); positive when CpG is depleted
    relative to its mirror dinucleotide GpC, which is what feeds the parity
    numerator."""
    s = s.upper()
    ncg = sum(1 for i in range(len(s) - 1) if s[i] == "C" and s[i + 1] == "G")
    ngc = sum(1 for i in range(len(s) - 1) if s[i] == "G" and s[i + 1] == "C")
    return (ngc - ncg) / (ngc + ncg) if (ngc + ncg) else 0.0


_G2 = None


def quadruplex_coverage(s: str) -> float:
    """Fraction of positions covered by a permissive two-guanine-tract motif
    (G{2,}N{1,7}){3}G{2,} or its C-run equivalent (the motif on the other
    strand), the quadruplex density of Table 5 and Figure 6C."""
    import re
    global _G2
    if _G2 is None:
        _G2 = [re.compile(r"(?=((?:%s{2,}[ACGT]{1,7}){3}%s{2,}))" % (b, b)) for b in "GC"]
    s = s.upper()
    cov = np.zeros(len(s), dtype=bool)
    for rx in _G2:
        for m in rx.finditer(s):
            cov[m.start():m.start() + len(m.group(1))] = True
    return float(cov.mean()) if len(s) else 0.0





# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z."""
    from scipy.stats import rankdata
    xr, yr, zr = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        B = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ coef

    from scipy.stats import spearmanr
    r, p = spearmanr(resid(xr, zr), resid(yr, zr))
    return float(r), float(p)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def analysis_table4(seq: str, lengths: Iterable[int], n_windows: int,
                    min_acgt: float, seed: int) -> "list[dict]":
    """Table 4 / Figure 6A: observed vs predicted nMSI and mean z per length."""
    rows = []
    for L in lengths:
        rng = np.random.default_rng(seed + L)
        wins = sample_clean_windows(seq, L, n_windows, min_acgt, rng)
        obs, pred, zs = [], [], []
        for w in wins:
            _, nm = parity.msi_scores(w)
            if nm is None:
                continue
            pA, pC, pG, pT = base_probs(w)
            base = nmsi_baseline_sliding(L, pA, pC, pG, pT)
            sd = nmsi_baseline_sd(L, seed=seed, pACGT=(pA, pC, pG, pT))
            obs.append(nm)
            pred.append(base)
            if sd and sd > 0:
                zs.append((nm - base) / sd)
        rows.append(dict(window_nt=L, n=len(obs),
                         observed_nmsi=round(float(np.mean(obs)), 3),
                         predicted_baseline=round(float(np.mean(pred)), 3),
                         mean_z=round(float(np.mean(zs)), 2) if zs else float("nan")))
    return rows


def _tile_set(seq: str, L: int = 300):
    return clean_tiles(seq, L)


_PRED_CACHE: dict = {}


def _predicted_300(w: str) -> float:
    key = tuple(w.count(b) for b in "ACGT")
    if key not in _PRED_CACHE:
        n = sum(key)
        _PRED_CACHE[key] = nmsi_baseline_sliding(len(w), *(k / n for k in key))
    return _PRED_CACHE[key]


def window_table(seq: str, starts, L: int = 300):
    """Per-window statistics for the tiled 300-nt windows."""
    import pandas as pd
    rows = []
    for st in starts:
        w = seq[st:st + L]
        _, nm = parity.msi_scores(w)
        if nm is None:
            continue
        rows.append(dict(start=st, nmsi=nm, gc=parity.gc_content(w),
                         predicted=_predicted_300(w),
                         cpg_under=cpg_underrepresentation(w),
                         g4_coverage=quadruplex_coverage(w)))
    d = pd.DataFrame(rows)
    d["deficit"] = d["predicted"] - d["nmsi"]
    return d


def analysis_fig6b(d) -> dict:
    """Figure 6B: deficit vs GC and vs CpG under-representation, and the
    partial with CpG holding GC constant."""
    from scipy import stats
    rho_gc, p_gc = stats.spearmanr(d.deficit, d.gc)
    rho_cpg, p_cpg = stats.spearmanr(d.deficit, d.cpg_under)
    pr, pp = partial_spearman(d.deficit.values, d.cpg_under.values, d.gc.values)
    return dict(n=len(d),
                deficit_vs_gc_rho=round(float(rho_gc), 3), deficit_vs_gc_p=float(p_gc),
                deficit_vs_cpgunder_rho=round(float(rho_cpg), 3),
                deficit_vs_cpgunder_p=float(p_cpg),
                deficit_vs_cpgunder_partial_gc_rho=round(float(pr), 3),
                deficit_vs_cpgunder_partial_gc_p=float(pp))


def analysis_table5(d) -> dict:
    """Table 5 / Figure 6C: quadruplex density vs symmetry, GC vs symmetry,
    quadruplex density vs GC, and the partial removing GC."""
    from scipy import stats
    nm, g, q = d.nmsi.values, d.gc.values, d.g4_coverage.values
    has = q > 0
    rho_g4_sym, p1 = stats.spearmanr(q, nm)
    rho_gc_sym, p2 = stats.spearmanr(g, nm)
    rho_g4_gc, p3 = stats.spearmanr(q, g)
    pr, pp = partial_spearman(q, nm, g)
    return dict(n=len(nm), frac_windows_with_g4=round(float(has.mean()), 3),
                mean_nmsi_with_g4=round(float(nm[has].mean()), 4),
                mean_nmsi_without_g4=round(float(nm[~has].mean()), 4),
                g4_vs_symmetry_rho=round(float(rho_g4_sym), 3), g4_vs_symmetry_p=float(p1),
                gc_vs_symmetry_rho=round(float(rho_gc_sym), 3), gc_vs_symmetry_p=float(p2),
                g4_vs_gc_rho=round(float(rho_g4_gc), 3), g4_vs_gc_p=float(p3),
                g4_vs_symmetry_partial_gc_rho=round(float(pr), 3),
                g4_vs_symmetry_partial_gc_p=float(pp))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chr21", required=True,
                    help="human chromosome 21 FASTA (.fa or .fa.gz); the paper "
                         "uses Ensembl GRCh38 Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz")
    ap.add_argument("--out", default="results/chr21",
                    help="output directory for the CSV tables")
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[60, 120, 300, 600, 1200, 3000],
                    help="window lengths for Table 4 (nt)")
    ap.add_argument("--n-windows", type=int, default=700,
                    help="random clean windows per length for Table 4")
    ap.add_argument("--min-acgt", type=float, default=1.0,
                    help="minimum ACGT fraction for a Table 4 window to be clean")
    ap.add_argument("--mask-repeats", action="store_true",
                    help="treat soft-masked (lower-case) bases as unclean; the "
                         "paper does not mask repeats")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    sys.stderr.write(f"[read] {a.chr21}\n")
    seq = read_single_fasta(a.chr21)
    if not a.mask_repeats:
        seq = seq.upper()
    sys.stderr.write(f"[read] chromosome 21 length = {len(seq):,} bp\n")
    if len(seq) < max(a.lengths) * 2:
        sys.exit("[error] sequence too short — is this really chromosome 21?")

    os.makedirs(a.out, exist_ok=True)
    import pandas as pd

    # ---- Table 4 / Fig 6A
    sys.stderr.write("[run] Table 4 / Figure 6A: observed vs predicted nMSI by window length\n")
    t4 = analysis_table4(seq, a.lengths, a.n_windows, a.min_acgt, a.seed)
    df4 = pd.DataFrame(t4)
    df4.to_csv(os.path.join(a.out, "table4_windows_by_length.csv"), index=False)
    print("\n=== Table 4: observed / predicted nMSI, mean z per window length ===")
    print(df4.to_string(index=False))

    # ---- Fig 6B
    sys.stderr.write("[run] Figure 6B: deficit vs GC and CpG under-representation\n")
    starts = _tile_set(seq)
    sys.stderr.write(f"[run] scoring {len(starts):,} tiled 300-nt windows\n")
    allw = window_table(seq, starts)
    allw.to_csv(os.path.join(a.out, "windows300.csv.gz"), index=False)
    b = analysis_fig6b(allw)
    pd.DataFrame([b]).to_csv(os.path.join(a.out, "fig6b_deficit_cpg.csv"), index=False)
    print("\n=== Figure 6B: symmetry deficit vs composition (n={}) ===".format(b["n"]))
    print(f"  deficit vs GC content        rho = {b['deficit_vs_gc_rho']:+.3f}")
    print(f"  deficit vs CpG under-rep.    rho = {b['deficit_vs_cpgunder_rho']:+.3f}")
    print(f"  deficit vs CpG | GC constant rho = {b['deficit_vs_cpgunder_partial_gc_rho']:+.3f} "
          f"(p = {b['deficit_vs_cpgunder_partial_gc_p']:.2e})")

    # ---- Table 5 / Fig 6C
    sys.stderr.write("[run] Table 5 / Figure 6C: quadruplex content vs symmetry\n")
    t5 = analysis_table5(allw)
    pd.DataFrame([t5]).to_csv(os.path.join(a.out, "table5_g4_symmetry.csv"), index=False)
    print("\n=== Table 5: quadruplex content vs mirror symmetry (n={}) ===".format(t5["n"]))
    print(f"  windows with a G4 motif                       {t5['frac_windows_with_g4']*100:.0f}%")
    print(f"  mean nMSI with / without G4                   "
          f"{t5['mean_nmsi_with_g4']:.4f} / {t5['mean_nmsi_without_g4']:.4f}")
    print(f"  G4 density vs mirror symmetry          rho = {t5['g4_vs_symmetry_rho']:+.3f}")
    print(f"  GC content vs mirror symmetry          rho = {t5['gc_vs_symmetry_rho']:+.3f}")
    print(f"  G4 density vs GC content               rho = {t5['g4_vs_gc_rho']:+.3f}")
    print(f"  G4 vs symmetry, holding GC constant    rho = {t5['g4_vs_symmetry_partial_gc_rho']:+.3f} "
          f"(p = {t5['g4_vs_symmetry_partial_gc_p']:.2g})")

    print(f"\n[saved] tables written to {a.out}/")
    sys.stderr.write(
        "\n[note] Table 4 depends on --seed through the random windows; Figure "
        "6B and Table 5 use every clean tile and are deterministic.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
