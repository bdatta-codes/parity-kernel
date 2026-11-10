#!/usr/bin/env python3
"""
experiment_538.py — wild type vs G4-abolished mutant, with the null measured.

Runs the wild-type-versus-mutant comparison on every confirmed or nominated
G-quadruplex sequence available (538 sequences), under declared mutagenesis
conventions, tested against a fair coin and against measured paired nulls.

Design
------
1. Sequences: all confirmed / nominated G4 sequences with >= 4 G-tracts of
   length >= 3, so that a quadruplex can actually be abolished.
2. Mutants: seven substitution conventions spanning the designs in general use
   (G->T / G->A / G->C at a defined position in each tract; minimal two-tract
   disruption; whole-tract ablation), plus a whole-sequence scramble reference.
3. Abolition is predicted computationally, not assumed: a mutant is kept only
   if it fails the canonical Quadparser regex and its G4Hunter score drops.
   40 wild types already fail the regex (a loop exceeds seven bases); results
   without them are in consensus_positive_subset.py (Supplementary Table S9).
4. Null (paired_null): the aligned wild type and mutant are shuffled together,
   as one sequence over the 16 (wild-type base, mutant base) symbols, uniformly
   (1,000 draws) or by Altschul-Erickson over that alphabet (400 draws); this
   gives per pair the probabilities g (null wild type higher) and t (tie).
   An unpaired null (independent shuffles of each member), which does not
   preserve the pairing, is given in the p0_* columns for comparison
   (Supplementary Methods S8).
5. Test (three_outcome_sf): each pair scores 1, 1/2 or 0; the exact upper tail
   of the concordance under the per-pair (g, t) is computed by dynamic
   programming. The fair-coin test is the sign test on untied pairs.
6. G4Hunter is run through the same pipeline; any guanine substitution lowers
   it, so its 100% concordance is a check of the pipeline, not independent
   evidence of abolition.

Seeds 20260909 (mutation and mononucleotide null) and 20260910 (dinucleotide
null). Writes results/experiment_538_results.csv, results/pairs_<convention>.csv
(one row per verified pair, with g and t) and results/verification_filter_check.csv
(the effect of the G4Hunter filter, Supplementary Table S8).
"""
from __future__ import annotations
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shuffles import dinuc_shuffle
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------- primitives --
_MAP = np.full(256, -1, np.int8)
for i, c in enumerate("ACGT"):
    _MAP[ord(c)] = i
_DEC = np.array(list("ACGT"))

_REV = np.zeros(64, np.int64)
for a in range(4):
    for b in range(4):
        for c in range(4):
            _REV[a*16+b*4+c] = c*16+b*4+a
_SELF = np.array([_REV[i] == i for i in range(64)])
_PAIRS, _seen = [], set()
for i in range(64):
    j = int(_REV[i])
    if i != j and (min(i, j), max(i, j)) not in _seen:
        _seen.add((min(i, j), max(i, j))); _PAIRS.append((min(i, j), max(i, j)))
_PI = np.array([p[0] for p in _PAIRS]); _PJ = np.array([p[1] for p in _PAIRS])


def enc(s):
    x = _MAP[np.frombuffer(s.upper().encode(), np.uint8)]
    return x[x >= 0]



def idx_msi(x):
    """(MSI, nMSI) from encoded bases."""
    if x.size < 3:
        return np.nan, np.nan
    f = np.bincount(x[:-2].astype(np.int64)*16 + x[1:-1].astype(np.int64)*4
                    + x[2:].astype(np.int64), minlength=64)
    num = np.abs(f[_PI]-f[_PJ]).sum()
    dnp = (f[_PI]+f[_PJ]).sum()
    dal = dnp + 2*f[_SELF].sum()
    return (1-num/dal if dal else np.nan, 1-num/dnp if dnp else np.nan)


def g4hunter(x):
    if x.size == 0:
        return 0.0
    brk = np.flatnonzero(np.diff(x)) + 1
    st, en = np.r_[0, brk], np.r_[brk, x.size]
    v = np.zeros(x.size)
    for s, e in zip(st, en):
        L = min(e-s, 4); b = x[s]
        v[s:e] = L if b == 2 else (-L if b == 1 else 0)
    return float(v.mean())


QP = re.compile(r'G{3,}\w{1,7}G{3,}\w{1,7}G{3,}\w{1,7}G{3,}')
def is_canonical(s):
    return QP.search(s.upper()) is not None


# ----------------------------------------------------------------- shuffles --
def mono(x, rng):
    y = x.copy(); rng.shuffle(y); return y


def _advance_main_stream(x, rng, tries=60):
    """Consume draws from the main generator `rng` before each dinucleotide
    shuffle (a random walk over shuffled successor lists, with a mononucleotide
    shuffle after 60 failed walks). Its output is discarded: the dinucleotide
    shuffles themselves are Altschul-Erickson draws from DI_RNG
    (shuffles.dinuc_shuffle). The call fixes the position of the main stream,
    from which the mononucleotide nulls and the scrambled mutants are drawn,
    so that every result reproduces exactly from the stated seeds."""
    n = x.size
    if n < 4:
        return x.copy()
    base = {}
    for a, b in zip(x[:-1], x[1:]):
        base.setdefault(int(a), []).append(int(b))
    for _ in range(tries):
        e = {k: list(v) for k, v in base.items()}
        for k in e:
            rng.shuffle(e[k])
        out = np.empty(n, np.int8); out[0] = x[0]; cur = int(x[0]); ok = True
        for t in range(1, n):
            lst = e.get(cur)
            if not lst:
                ok = False; break
            nxt = lst.pop(); out[t] = nxt; cur = nxt
        if ok:
            return out
    return mono(x, rng)


def shuffle_dist(x, K, rng, kind):
    """K shuffled nMSI and MSI values. Mononucleotide shuffles use `rng`;
    dinucleotide shuffles are Altschul-Erickson draws from DI_RNG
    (_advance_main_stream only advances `rng`)."""
    m = np.empty(K); n = np.empty(K)
    for i in range(K):
        if kind == "mono":
            y = mono(x, rng)
        else:
            _advance_main_stream(x, rng)
            y = dinuc_shuffle(x, DI_RNG)
        m[i], n[i] = idx_msi(y)
    return m, n


def p_greater(a, b):
    """P(A > B) + 0.5 P(A == B) from two samples, ties counted as one half."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    u = stats.mannwhitneyu(a, b, alternative="greater", method="asymptotic").statistic
    return u/(a.size*b.size)


# ------------------------------------------------------------ paired null --
def paired_null(xw, xm, K, kind, rng):
    """Paired null for one wild-type/mutant pair of equal length.

    The two sequences are aligned position by position and encoded as one
    sequence over the 16 symbols (wild-type base, mutant base). That sequence is
    shuffled -- uniformly (kind='mono') or by the Altschul-Erickson dinucleotide
    shuffle over the pair alphabet (kind='dinuc') -- and read back as a null wild
    type and a null mutant. Each null wild type has exactly the composition
    (mono) or dinucleotide composition (dinuc) of the real wild type, each null
    mutant that of the real mutant, and the two differ at the same number of
    positions by the same substitutions, so the pairing is kept. Returns, for
    nMSI and MSI, the probabilities that the null wild type scores higher (g)
    and that the two tie (t), over the finite draws."""
    z = (xw.astype(np.int16) * 4 + xm.astype(np.int16))
    res = np.empty((K, 4))
    for k in range(K):
        zz = rng.permutation(z) if kind == "mono" else dinuc_shuffle(z, rng)
        a = idx_msi((zz // 4).astype(np.int8)); b = idx_msi((zz % 4).astype(np.int8))
        res[k] = (a[0], b[0], a[1], b[1])
    out = {}
    for lab, (i, j) in (("MSI", (0, 1)), ("nMSI", (2, 3))):
        A, B = res[:, i], res[:, j]
        ok = np.isfinite(A) & np.isfinite(B)
        A, B = A[ok], B[ok]
        out[lab] = (float(np.mean(A > B + 1e-12)), float(np.mean(np.isclose(A, B))), int(ok.sum()))
    return out


def three_outcome_sf(k, g, t):
    """P(S >= k), S = sum over pairs of 1 (prob g_i), 1/2 (prob t_i), 0 (else).
    Exact, by dynamic programming in half-units; k may be a half-integer."""
    g = np.asarray(g, float); t = np.asarray(t, float)
    dist = np.zeros(2 * len(g) + 1); dist[0] = 1.0
    for gi, ti in zip(g, t):
        li = max(0.0, 1.0 - gi - ti)
        new = dist * li
        new[1:] += dist[:-1] * ti
        new[2:] += dist[:-2] * gi
        dist = new
    return float(dist[int(round(2 * k)):].sum())


# ------------------------------------------------------------- conventions --
def tracts(s):
    return [(m.start(), m.end()) for m in re.finditer(r'G{3,}', s.upper())]


def mutate(s, conv, rng):
    l = list(s.upper()); tr = tracts(s)
    if conv == "T_centre":
        for a, b in tr: l[(a+b)//2] = 'T'
    elif conv == "T_3prime":
        for a, b in tr: l[b-1] = 'T'
    elif conv == "T_5prime":
        for a, b in tr: l[a] = 'T'
    elif conv == "A_centre":
        for a, b in tr: l[(a+b)//2] = 'A'
    elif conv == "C_centre":
        for a, b in tr: l[(a+b)//2] = 'C'
    elif conv == "two_tract":
        for a, b in tr[:2]: l[(a+b)//2] = 'T'
    elif conv == "tract_ablate":
        a, b = tr[0]
        for i in range(a, b): l[i] = 'T'
    elif conv == "scramble":
        rng.shuffle(l)
    return "".join(l)


CONV = {
 "T_centre":     "G→T at the central guanine of every tract (index (a+b)//2, i.e. position floor(t/2)+1; 3′-side of the central pair for even t)",
 "T_3prime":     "G→T at the 3′ guanine of every tract",
 "T_5prime":     "G→T at the 5′ guanine of every tract",
 "A_centre":     "G→A at the central guanine of every tract",
 "C_centre":     "G→C at the central guanine of every tract",
 "two_tract":    "G→T at the central guanine of two tracts (minimal disruption)",
 "tract_ablate": "the first tract replaced entirely by thymine",
 "scramble":     "whole-sequence scramble (reference control)",
}


# ------------------------------------- two-outcome exact tail (no ties) --
def poisson_binomial_sf(k, p):
    """P(K >= k) where K = sum of independent Bernoulli(p_i). Exact, by DP."""
    p = np.asarray(p, float); p = p[np.isfinite(p)]
    dist = np.zeros(len(p)+1); dist[0] = 1.0
    for pi in p:
        dist[1:] = dist[1:]*(1-pi) + dist[:-1]*pi
        dist[0] *= (1-pi)
    return float(dist[int(np.ceil(k)):].sum()), dist


# ------------------------------------------------------------------- main --
def load():
    seqs, src = [], []
    d = pd.read_excel(os.path.join(DATA, "298_sequences_with_MSI_nMSI.xlsx"))
    for s in d["WT_Sequence (5'-3')"]:
        seqs.append(str(s).upper()); src.append("G4Hunter reference")
    for l in open(os.path.join(DATA, "a549_grich_300.txt")):
        seqs.append(l.split('\t')[1].strip().upper()); src.append("A549 G-rich")
    onc = ["TGGGGAGGGTGGGGAGGGTGGGGAAGG","GGGGCGGGCCGGGGGCGGGG",
      "GAAGGAGGGCAGGAGGGCTCTGGGTGGGTCTGT","GCGCGGGGAGGGGAGAGGGGGCGGGAGCGCG",
      "AGGGCGGTGTGGGAAGAGGGAAGAGGGGGAGG","GGGCAGGGTGTGGGGAAACAGGG",
      "AGGGAGGGCGCTGGGAGGAGGG","CAGGGCAGCGCGCGTGTGGGAAGGGGCGGAGGGAG",
      "GGTGGGGTGGGAGTGGGGACAGGGCCGGGGGTGGGGGTGCTTGGGGACA",
      "GCTGGGCGGGCGGGCGGGCGGGTGG","CGGGGCGGGGCGGGGGCGGGGGCG",
      "GGGGCTGGGGTCCCTTGGGGAACTGTATTGGG",
      "GTTCGTGGCTGGGGGCTCGGGGCGCGGGGCGCGGGGCATGGGGC","GGGAGGGGGCGGGGAGGG"]
    for s in onc:
        seqs.append(s); src.append("oncogene promoter")
    keep = [i for i, s in enumerate(seqs)
            if set(s) <= set("ACGT") and len(tracts(s)) >= 4 and len(s) >= 16]
    return [seqs[i] for i in keep], [src[i] for i in keep]


# inputs from data/, outputs to results/, wherever the script is run from
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("PK_DATA", os.path.join(_ROOT, "data"))
OUT = os.environ.get("PK_OUT", os.path.join(_ROOT, "results"))

AUDIT = []
DI_RNG = None
PAIR_RNG = None
PAIR_DI_RNG = None


def main():
    # shuffles per member: 1,000 mononucleotide and 400 dinucleotide, as in the paper
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    KD = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    global DI_RNG
    rng = np.random.default_rng(20260909)
    DI_RNG = np.random.default_rng(20260910)   # Altschul-Erickson dinucleotide shuffles
    global PAIR_RNG, PAIR_DI_RNG
    PAIR_RNG = np.random.default_rng(20261101)     # paired null, mononucleotide
    PAIR_DI_RNG = np.random.default_rng(20261102)  # paired null, dinucleotide
    seqs, src = load()
    n = len(seqs)
    print(f"sequences with >= 4 G-tracts: {n}", file=sys.stderr)
    print(f"  by source: " + ", ".join(f"{k} {v}" for k, v in
          pd.Series(src).value_counts().items()), file=sys.stderr)

    X = [enc(s) for s in seqs]
    wt = np.array([idx_msi(x) for x in X])                     # (n,2) MSI,nMSI
    wtg4 = np.array([g4hunter(x) for x in X])

    print("  shuffling wild types...", file=sys.stderr)
    wt_mono = [shuffle_dist(x, K, rng, "mono") for x in X]
    wt_di   = [shuffle_dist(x, KD, rng, "dinuc") for x in X]

    out = []
    for conv in CONV:
        rows = []
        for i, s in enumerate(seqs):
            m = mutate(s, conv, rng)
            if len(m) != len(s):
                continue
            xm = enc(m)
            if is_canonical(m):            # abolition not achieved -> drop
                continue
            gm = g4hunter(xm)
            # audit of the G4Hunter condition (uses no random numbers;
            # results/verification_filter_check.csv, Supplementary Table S8)
            _mm, _mn = idx_msi(xm)
            if np.isfinite(_mn) and np.isfinite(wt[i, 1]):   # as for the pairs
                AUDIT.append(dict(convention=conv, g4h_pass=bool(gm < wtg4[i]),
                                  d_nmsi=wt[i, 1] - _mn, d_g4=wtg4[i] - gm))
            if gm >= wtg4[i]:              # propensity must fall
                continue
            mm, mn = idx_msi(xm)
            if not np.isfinite(mn) or not np.isfinite(wt[i, 1]):
                continue
            sm_m, sn_m = shuffle_dist(xm, K, rng, "mono")
            sm_d, sn_d = shuffle_dist(xm, KD, rng, "dinuc")
            pm = paired_null(X[i], xm, K, "mono", PAIR_RNG)
            pd_ = paired_null(X[i], xm, KD, "dinuc", PAIR_DI_RNG)
            rows.append(dict(
                i=i, source=src[i], wt_MSI=wt[i, 0], wt_nMSI=wt[i, 1],
                mut_MSI=mm, mut_nMSI=mn, wt_g4=wtg4[i], mut_g4=gm,
                # paired null (used for every reported test)
                g_nMSI_mono=pm["nMSI"][0], t_nMSI_mono=pm["nMSI"][1],
                g_MSI_mono=pm["MSI"][0], t_MSI_mono=pm["MSI"][1],
                g_nMSI_di=pd_["nMSI"][0], t_nMSI_di=pd_["nMSI"][1],
                # unpaired null (independent shuffles of each member), for comparison
                p0_nMSI_mono=p_greater(wt_mono[i][1], sn_m),
                p0_MSI_mono=p_greater(wt_mono[i][0], sm_m),
                p0_nMSI_di=p_greater(wt_di[i][1], sn_d),
            ))
        R = pd.DataFrame(rows)
        if R.empty:
            continue
        for lab, wcol, mcol, null in (("nMSI", "wt_nMSI", "mut_nMSI", "nMSI_mono"),
                                      ("MSI", "wt_MSI", "mut_MSI", "MSI_mono"),
                                      ("nMSI/dinuc", "wt_nMSI", "mut_nMSI", "nMSI_di")):
            d = R[wcol].values - R[mcol].values
            win, tie = d > 1e-12, np.isclose(d, 0)
            lose = ~win & ~tie
            k = float(win.sum() + 0.5 * tie.sum())
            g, t = R[f"g_{null}"].values, R[f"t_{null}"].values
            sf = three_outcome_sf(k, g, t)
            # fair coin: one-sided sign test on the untied pairs
            fair = stats.binomtest(int(win.sum()), int(win.sum() + lose.sum()), 0.5,
                                   alternative="greater").pvalue
            score = win + 0.5 * tie
            bs = [np.mean(rng.choice(score, len(score))) for _ in range(2000)]
            p0 = R["p0_" + null.replace("nMSI_di", "nMSI_di")].values
            out.append(dict(convention=conv, index=lab, n=len(d),
                            concordant=k, obs=k/len(d), ties=int(tie.sum()),
                            obs_lo=np.percentile(bs, 2.5), obs_hi=np.percentile(bs, 97.5),
                            null=float(np.mean(g + t / 2)), null_tie=float(np.mean(t)),
                            p_fair=fair, p_corrected=sf,
                            null_unpaired=np.nanmean(p0),
                            p_unpaired=poisson_binomial_sf(k, p0)[0],
                            mean_delta=d.mean()))
        # positive control: G4Hunter on the same pairs, same fair-coin test
        dg = R.wt_g4.values - R.mut_g4.values
        out.append(dict(convention=conv, index="G4Hunter (control)", n=len(dg),
                        concordant=float((dg > 0).sum()), obs=(dg > 0).mean(),
                        obs_lo=np.nan, obs_hi=np.nan, null=np.nan,
                        p_fair=stats.binomtest(int((dg > 0).sum()), len(dg), 0.5,
                                               alternative="greater").pvalue,
                        p_corrected=np.nan, mean_delta=dg.mean()))
        R.to_csv(os.path.join(OUT, f"pairs_{conv}.csv"), index=False)
        print(f"  {conv}: {len(R)} verified-abolished pairs", file=sys.stderr)

    # how far the G4Hunter verification condition binds, per convention
    A = pd.DataFrame(AUDIT)
    conc = lambda d: round(100 * ((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d), 1)
    rows = []
    for conv, g in A.groupby("convention", sort=False):
        k = g[g.g4h_pass]
        rows.append(dict(convention=conv, n_QP_negative=len(g), n_after_G4H_filter=len(k),
                         pct_removed=round(100 * (len(g) - len(k)) / len(g), 1),
                         nMSI_conc_QPonly=conc(g.d_nmsi), nMSI_conc_filtered=conc(k.d_nmsi),
                         G4H_conc_QPonly=round(100 * (g.d_g4 > 0).mean(), 1),
                         G4H_conc_filtered=round(100 * (k.d_g4 > 0).mean(), 1)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "verification_filter_check.csv"), index=False)
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(OUT, "experiment_538_results.csv"), index=False)
    pd.set_option("display.width", 200)
    print(O.to_string(index=False))


if __name__ == "__main__":
    main()
