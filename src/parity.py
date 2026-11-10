"""Core implementations: MSI/nMSI (reference), G4Hunter, closed-form parity baseline.

MSI/nMSI is the reference implementation used throughout; it reproduces the
MSI and nMSI values stored with the G4Hunter reference set (298 sequences and
one stored shuffle of each) to 5.55e-17 (src/si_compute.py, Table S6).
"""
from __future__ import annotations
import numpy as np
from math import lgamma, sqrt, pi, exp

# --------------------------------------------------------------------- MSI --
def msi_scores(seq: str):
    """(MSI, nMSI) of a sequence, over overlapping trinucleotides (Eq. 1)."""
    seq = seq.upper()
    trip = [seq[i:i+3] for i in range(len(seq)-2)]
    if not trip:
        return (None, None)
    freq = {}
    for w in trip:
        freq[w] = freq.get(w, 0) + 1
    num_all = den_all = num_np = den_np = 0.0
    counted = set()
    for w in freq:
        rw = w[::-1]
        key = tuple(sorted([w, rw]))
        if key in counted:
            continue
        counted.add(key)
        fw, frw = freq.get(w, 0), freq.get(rw, 0)
        num_all += abs(fw - frw); den_all += (fw + frw)
        if w != rw:
            num_np += abs(fw - frw); den_np += (fw + frw)
    MSI = 1 - num_all/den_all if den_all > 0 else None
    nMSI = 1 - num_np/den_np if den_np > 0 else None
    return (MSI, nMSI)


def gc_content(s: str) -> float:
    s = s.upper()
    n = sum(c in "ACGT" for c in s)
    return (s.count("G") + s.count("C")) / n if n else 0.0



# --------------------------------------------------------------- G4Hunter --
def g4hunter(seq: str) -> float:
    """Bedrat-Lacroix-Mergny G4Hunter: run-length score per base, capped at 4."""
    s = seq.upper(); out = []
    i = 0; n = len(s)
    while i < n:
        c = s[i]; j = i
        while j < n and s[j] == c:
            j += 1
        L = min(j - i, 4)
        v = L if c == "G" else (-L if c == "C" else 0)
        out.extend([v] * (j - i))
        i = j
    return float(np.mean(out)) if out else 0.0



# ------------------------------------------------- closed-form parity null --
_MAD_CACHE = {}

def binom_mad(m: int) -> float:
    """E|2X-m| for X~Bin(m,1/2)  =  m * C(m-1, floor((m-1)/2)) / 2^(m-1)."""
    if m <= 0:
        return 0.0
    if m in _MAD_CACHE:
        return _MAD_CACHE[m]
    k = (m - 1) // 2
    logc = lgamma(m) - lgamma(k + 1) - lgamma(m - k)      # log C(m-1,k)
    val = m * exp(logc - (m - 1) * np.log(2.0))
    _MAD_CACHE[m] = val
    return val


def _trinuc_probs(pA, pC, pG, pT):
    p = {"A": pA, "C": pC, "G": pG, "T": pT}
    out = {}
    for a in "ACGT":
        for b in "ACGT":
            for c in "ACGT":
                out[a+b+c] = p[a]*p[b]*p[c]
    return out


def nmsi_baseline_exact(N: int, pA=.25, pC=.25, pG=.25, pT=.25) -> float:
    """E[nMSI] for N non-overlapping trinucleotides, exact binomial form.

    Ratio of expectations; the denominator's CV falls as 1/sqrt(N) so the
    correction is O(1/N) (Supplementary Methods S5).
    """
    P = _trinuc_probs(pA, pC, pG, pT)
    pairs = []
    seen = set()
    for w, pw in P.items():
        rw = w[::-1]
        if w == rw:
            continue
        key = tuple(sorted([w, rw]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(pw + P[rw])
    Q = sum(pairs)                       # prob mass in non-self-reverse words
    if Q <= 0 or N <= 0:
        return float("nan")
    num = 0.0
    for ppair in pairs:
        # m ~ Binomial(N, ppair); average MAD(m) over m
        mm = np.arange(0, N + 1)
        logpmf = (lgamma(N+1) - np.array([lgamma(k+1)+lgamma(N-k+1) for k in mm])
                  + mm*np.log(ppair) + (N-mm)*np.log1p(-ppair)) if 0 < ppair < 1 else None
        if logpmf is None:
            continue
        pmf = np.exp(logpmf)
        keep = pmf > 1e-14
        num += float(np.sum(pmf[keep] * np.array([binom_mad(int(m)) for m in mm[keep]])))
    return 1.0 - num / (N * Q)


def nmsi_baseline_asymptotic(N: int, pA=.25, pC=.25, pG=.25, pT=.25) -> float:
    """Large-N form 1 - C/sqrt(N), C = (2/(Q*sqrt(pi))) * sum sqrt(P(w))."""
    P = _trinuc_probs(pA, pC, pG, pT)
    seen = set(); tot = 0.0; Q = 0.0
    for w, pw in P.items():
        rw = w[::-1]
        if w == rw:
            continue
        key = tuple(sorted([w, rw]))
        if key in seen:
            continue
        seen.add(key)
        # One sqrt(P(w)) per pair (P(w) = P(rev w) under independent bases):
        # MAD(m) ~ sqrt(2m/pi) with E m = 2NP(w) gives 2*sqrt(N P(w)/pi).
        # Written symmetrically so it stays well defined for any P.
        tot += sqrt((pw + P[rw]) / 2.0); Q += pw + P[rw]
    C = (2.0 / (Q * sqrt(pi))) * tot
    return 1.0 - C / sqrt(N), C


def nmsi_expectation_conditional(N: int, pA=.25, pC=.25, pG=.25, pT=.25) -> float:
    """Exact E[nMSI] for N words counted WITHOUT overlap (multinomial counts).

    nmsi_baseline_exact divides the expected numerator by the expected
    denominator (a ratio of expectations, error O(1/N)). Here the number b of
    non-self-reverse words is conditioned on instead: given b the denominator is
    fixed and each pair total is Binomial(b, P_pair/Q), so
        E[nMSI] = sum_b P(b | b > 0) * [1 - sum_pairs E MAD(m) / b],
    with b ~ Binomial(N, Q), which is exact. Supplementary Methods S5."""
    from scipy.stats import binom
    import numpy as _np
    P = _trinuc_probs(pA, pC, pG, pT)
    seen, pp = set(), []
    for w, pw in P.items():
        rw = w[::-1]
        key = tuple(sorted([w, rw]))
        if w == rw or key in seen:
            continue
        seen.add(key)
        pp.append(pw + P[rw])
    pp = _np.array(pp); Q = pp.sum()
    mad = _np.array([binom_mad(m) if m > 0 else 0.0 for m in range(N + 1)])
    tot = 0.0
    for b in range(1, N + 1):
        wb = binom.pmf(b, N, Q)
        if wb < 1e-16:
            continue
        m = _np.arange(b + 1)
        e_num = sum(binom.pmf(m, b, p / Q) @ mad[:b + 1] for p in pp)
        tot += wb * (1.0 - e_num / b)
    return tot / (1.0 - binom.pmf(0, N, Q))
