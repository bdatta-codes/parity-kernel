#!/usr/bin/env python3
"""closed_form.py — the closed-form baseline, its Monte Carlo validation and
everything built on it in the Results subsections "A Closed-Form Expectation
Under an Independent-Base Background" to "What a Fixed Numerical Criterion
Encodes" (Eqs. 2-4, Tables 2 and 3, Supplementary Table S7 and Supplementary Methods S10, the statements on the
reverse-complement index S₁ and the data for Figure 2).

Library part: a general closed form for a parity index on words of length k
under the mirror (reverse) or reverse-complement pairing at any independent-
base composition, and a vectorised simulator (integer word codes + bincount)
for sliding-window or non-overlapping counting.

Script part (python3 src/closed_form.py): writes results/closed_form/*.csv;
Figure 2 is then drawn by src/make_fig2.py from those CSVs.

Definitions (see also the module constants below)
  * N = L - k + 1 words for a sliding window over L nt; N = floor(L/k) for
    words counted without overlap.
  * index = 1 - sum_pairs |f(w) - f(partner)| / sum_pairs (f(w) + f(partner)),
    words that are their own partner excluded (nMSI for k = 3, mirror; S1 for
    k = 3, reverse complement). Sequences with an empty denominator are
    dropped (the index is undefined there).
  * closed form = ratio of expectations: 1 - sum_pairs E|f(w)-f(p)| / (N Q),
    with m = f(w)+f(p) ~ Bin(N, P(w)+P(p)) and f(w) | m ~ Bin(m, theta),
    theta = P(w) / (P(w) + P(p)) (theta = 1/2 for the mirror pairing).
  * kappa: E_sliding(L) ~= closed(N = L-k+1) + kappa / sqrt(N), fitted by
    least squares with no intercept; RMS residual = sqrt(mean residual^2).
  * composition: pG = GC(1+skew)/2, pC = GC(1-skew)/2, pA = pT = (1-GC)/2.

License: MIT
"""
from __future__ import annotations
import os, sys, time
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "closed_form")
SEED = 20261025
KAPPA_EQ4 = 0.615            # Eq. 4

# ----------------------------------------------------------------- words --
BASES = "ACGT"                  # codes 0..3; complement(x) = 3 - x


def composition(gc=0.5, skew=0.0):
    """(pA, pC, pG, pT) from GC content and G-over-C skew within GC."""
    return np.array([(1 - gc) / 2, gc * (1 - skew) / 2, gc * (1 + skew) / 2, (1 - gc) / 2])


@lru_cache(maxsize=None)
def digits(k):
    """(4^k, k) array of the base codes of every word, most significant first."""
    idx = np.arange(4 ** k)
    return np.stack([(idx // 4 ** (k - 1 - j)) % 4 for j in range(k)], axis=1)


def _code(d):
    k = d.shape[1]
    return (d * (4 ** np.arange(k - 1, -1, -1))).sum(1)


@lru_cache(maxsize=None)
def partner(k, pairing):
    """Index of each word's partner: 'mirror' = reverse, 'rc' = reverse complement."""
    d = digits(k)[:, ::-1]
    if pairing == "rc":
        d = 3 - d
    elif pairing != "mirror":
        raise ValueError(pairing)
    return _code(d)


def word_probs(k, p4):
    return np.prod(np.asarray(p4)[digits(k)], axis=1)


def pair_table(k, pairing, p4):
    """Per pair (each unordered {w, partner}, w != partner): P(w), P(partner)."""
    P = word_probs(k, p4)
    part = partner(k, pairing)
    idx = np.arange(4 ** k)
    canon = idx < part
    return P[canon], P[part[canon]]


def structure(k, pairing):
    part = partner(k, pairing)
    n_self = int((part == np.arange(4 ** k)).sum())
    return (4 ** k - n_self) // 2, n_self


# ----------------------------------------------------------- closed form --
def abs_dev(m, theta):
    """E|2X - m|, X ~ Bin(m, theta), vectorised over integer m >= 0.

    Uses E[X 1{X>=t}] = m theta P(Bin(m-1, theta) >= t-1) with t = floor(m/2)+1,
    so E[(2X-m)^+] = 2 m theta S(m-1, t-1) - m S(m, t) and E|D| = 2E[D^+] - E[D].
    At theta = 1/2 this is De Moivre's m C(m-1, floor((m-1)/2)) / 2^(m-1) (Eq. 2).
    """
    m = np.asarray(m, dtype=np.int64)
    out = np.zeros(m.shape, float)
    pos = m > 0
    mm = m[pos]
    t = mm // 2 + 1
    s1 = binom.sf(t - 2, mm - 1, theta)        # P(Bin(m-1) >= t-1)
    s0 = binom.sf(t - 1, mm, theta)            # P(Bin(m)   >= t)
    dplus = 2 * mm * theta * s1 - mm * s0
    out[pos] = 2 * dplus - mm * (2 * theta - 1)
    return out


def _unique_pairs(k, pairing, p4):
    pw, pp = pair_table(k, pairing, p4)
    tot = pw + pp
    th = np.where(tot > 0, pw / np.where(tot > 0, tot, 1), 0.5)
    th = np.minimum(th, 1 - th)                # |D| is symmetric in theta <-> 1-theta
    key = np.round(np.c_[tot, th] * 1e12).astype(np.int64)
    _, first, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return tot[first], th[first], cnt, tot.sum()


def closed_form(N, k=3, pairing="mirror", p4=(.25, .25, .25, .25)):
    """Ratio-of-expectations baseline for N words (scalar or iterable N)."""
    tot, th, cnt, Q = _unique_pairs(k, pairing, p4)
    Ns = np.atleast_1d(np.asarray(N, dtype=np.int64))
    res = np.empty(len(Ns))
    for i, n in enumerate(Ns):
        num = 0.0
        for p, t, c in zip(tot, th, cnt):
            if p <= 0:
                continue
            lo = int(binom.ppf(1e-15, n, p)); hi = int(binom.isf(1e-15, n, p))
            m = np.arange(max(lo, 0), min(hi, n) + 1)
            num += c * float(binom.pmf(m, n, p) @ abs_dev(m, t))
        res[i] = 1 - num / (n * Q)
    return res if np.ndim(N) else float(res[0])


def asymptotic_C(k=3, pairing="mirror", p4=(.25, .25, .25, .25)):
    """C of Eq. 3 (mirror pairing): (2/(Q sqrt(pi))) sum_pairs sqrt(P(w))."""
    pw, pp = pair_table(k, pairing, p4)
    Q = (pw + pp).sum()
    return 2 / (Q * np.sqrt(np.pi)) * np.sqrt((pw + pp) / 2).sum()


def kappa_inf(k=3, p4=(.25, .25, .25, .25)):
    """Large-N limit of the sliding-window overlap constant for the mirror pairing,
    derived rather than fitted. With independent bases, the signed difference of a
    pair, D = sum_i Y_i with Y_i = 1[word at i is w] - 1[word at i is rev w], has
    mean zero and per-position variance v = Var(Y_0) + 2 sum_{h=1}^{k-1} Cov(Y_0, Y_h)
    (classical overlapping-word moments); D is asymptotically normal, so
    E|D| ~ sqrt(2 N v / pi) and E[nMSI_sliding] -> 1 - C'/sqrt(N) with
    C' = sum_pairs sqrt(2 v / pi) / Q. Returns (kappa_inf = C - C', C, C'), C as in
    asymptotic_C. Exact enumeration over strings of length 2k - 1."""
    p4 = np.asarray(p4, float)
    L = 2 * k - 1
    strings = np.array(list(np.ndindex(*(4,) * L)))
    prob = np.prod(p4[strings], axis=1)
    pw_all = word_probs(k, p4)
    Q = 0.0; Cp = 0.0; seen = set()
    for w in range(4 ** k):
        dw = np.array([(w // 4 ** (k - 1 - j)) % 4 for j in range(k)])
        rw = dw[::-1]; r = int(sum(int(rw[j]) * 4 ** (k - 1 - j) for j in range(k)))
        if r == w or (min(w, r), max(w, r)) in seen:
            continue
        seen.add((min(w, r), max(w, r)))
        def Y(h):
            win = strings[:, h:h + k]
            return (np.all(win == dw, 1).astype(float) - np.all(win == rw, 1).astype(float))
        Y0 = Y(0)
        v = (prob * Y0 * Y0).sum() + 2 * sum((prob * Y0 * Y(h)).sum() for h in range(1, k))
        Q += pw_all[w] + pw_all[r]
        Cp += np.sqrt(2 * max(v, 0.0) / np.pi)
    Cp /= Q
    C = asymptotic_C(k, "mirror", p4)
    return C - Cp, C, Cp


def rc_ceiling(k=3, p4=(.25, .25, .25, .25)):
    """Large-N limit of the reverse-complement index: 1 - sum|P(w)-P(rc w)| / Q."""
    pw, pp = pair_table(k, "rc", p4)
    return 1 - np.abs(pw - pp).sum() / (pw + pp).sum()


def theta_departure(k, p4, pairing="rc"):
    """max over pairs of |theta - 1/2|, theta = P(w)/(P(w)+P(partner))."""
    pw, pp = pair_table(k, pairing, p4)
    return float(np.max(np.abs(pw / (pw + pp) - 0.5)))


# ------------------------------------------------------------ simulation --
def simulate(L, k=3, pairing="mirror", p4=(.25, .25, .25, .25), reps=4000,
             rng=None, overlap=True, chunk_cells=2.4e7):
    """Index values for `reps` independent-base sequences of length L.

    overlap=True: sliding window, N = L-k+1 words; False: words at k, 2k, ...
    (N = L // k). Returns the array of defined index values."""
    rng = np.random.default_rng(SEED) if rng is None else rng
    p4 = np.asarray(p4, float)
    uniform = np.allclose(p4, .25)
    step = 1 if overlap else k
    nW = (L - k) // step + 1
    K = 4 ** k
    part = partner(k, pairing)
    canon = np.flatnonzero(np.arange(K) < part)
    pc = part[canon]
    r_chunk = int(max(50, min(chunk_cells / L, chunk_cells / K / 2)))
    vals, done = [], 0
    while done < reps:
        r = min(r_chunk, reps - done)
        if uniform:
            s = rng.integers(0, 4, size=(r, L), dtype=np.int8)
        else:
            s = np.searchsorted(np.cumsum(p4)[:3], rng.random((r, L)), side="right").astype(np.int8)
        codes = np.zeros((r, nW), np.int64)
        for j in range(k):
            codes = codes * 4 + s[:, j: j + (nW - 1) * step + 1: step]
        codes += (np.arange(r, dtype=np.int64) * K)[:, None]
        C = np.bincount(codes.ravel(), minlength=r * K).reshape(r, K)
        a, b = C[:, canon], C[:, pc]
        num = np.abs(a - b).sum(1); den = (a + b).sum(1)
        ok = den > 0
        vals.append(1 - num[ok] / den[ok])
        done += r
    return np.concatenate(vals)


def fit_kappa(Ns, d):
    """Least-squares kappa for d(N) = kappa/sqrt(N); returns kappa, RMS residual."""
    Ns = np.asarray(Ns, float); d = np.asarray(d, float)
    x = 1 / np.sqrt(Ns)
    kap = float((x * d).sum() / (x * x).sum())
    return kap, float(np.sqrt(np.mean((d - kap * x) ** 2)))


def kappa_series(Ls, k, pairing, p4, reps, rng):
    rows = []
    for L in Ls:
        v = simulate(L, k, pairing, p4, reps, rng)
        N = L - k + 1
        cf = closed_form(N, k, pairing, p4)
        rows.append(dict(L=L, N=N, sim_mean=v.mean(), sim_se=v.std(ddof=1) / np.sqrt(len(v)),
                         sim_sd=v.std(ddof=1), closed=cf, d=v.mean() - cf,
                         kappa_L=(v.mean() - cf) * np.sqrt(N)))
    df = pd.DataFrame(rows)
    kap, rms = fit_kappa(df.N, df.d)
    df["fit"] = df.closed + kap / np.sqrt(df.N)
    df["resid"] = df.sim_mean - df.fit
    return df, kap, rms


def min_length(f, lo, hi, target=0.5):
    """Smallest integer L in [lo, hi] with f(L) >= target (f increasing)."""
    if f(hi) < target:
        raise ValueError("not reached")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


# =================================================================== script ==
GC_T2 = [0.20, 0.35, 0.50, 0.65, 0.80, 0.90]
SKEW_T2 = [0.0, 0.4, 0.8]
L_UNIFORM = [30, 50, 80, 120, 200, 350, 600, 1000, 1800, 3000]   # ten lengths, 30-3,000 nt
L_TS7 = [int(x) for x in np.unique(np.round(np.geomspace(60, 6000, 10)))]  # ten, 60-6,000 nt
L_T3 = [15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500]
N_T1 = [10, 20, 30, 50, 100, 200, 500]
U = (0.25, 0.25, 0.25, 0.25)


def _rng(tag):
    return np.random.default_rng([SEED, tag])


def _save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)


def part_uniform(S):
    """Eq. 4 at uniform composition, ten lengths 30-3,000 nt; Figure 2A data."""
    df, kap, rms = kappa_series(L_UNIFORM, 3, "mirror", U, 60_000, _rng(1))
    x = 1 / np.sqrt(df.N)
    kap_se = float(np.sqrt((x ** 2 * df.sim_se ** 2).sum()) / (x ** 2).sum())
    C = asymptotic_C()
    assert abs(parity.nmsi_baseline_asymptotic(100)[1] - C) < 1e-12
    _save(df, "kappa_uniform_30_3000.csv")
    # summary of Eq. 4: median of per-length kappa (used, 0.615 to three places),
    # least-squares kappa, and the RMS residual of each
    rms615 = float(np.sqrt(np.mean((df.sim_mean - (df.closed + KAPPA_EQ4 / np.sqrt(df.N))) ** 2)))
    pd.DataFrame([dict(kappa_median_per_length=float(df.kappa_L.median()), kappa_used=KAPPA_EQ4,
                       rms_residual_kappa_used=rms615, kappa_least_squares=kap,
                       rms_residual_least_squares=rms)]).to_csv(os.path.join(OUT, "kappa_summary.csv"), index=False)
    S.update(kappa_30_3000=kap, kappa_30_3000_se=kap_se, rms_30_3000=rms,
             kappa_30_3000_median_L=float(df.kappa_L.median()),
             C=C, C_prime=C - kap, C_prime_eq4=C - KAPPA_EQ4)
    # words counted without overlap (Table 1 sizes) for the open circles of 2A
    rng = _rng(2)
    rows = []
    for N in N_T1:
        v = simulate(3 * N, 3, "mirror", U, 100_000, rng, overlap=False)
        rows.append(dict(N=N, sim_mean=v.mean(), sim_se=v.std(ddof=1) / np.sqrt(len(v)),
                         closed=closed_form(N)))
    _save(pd.DataFrame(rows), "nonoverlap_uniform.csv")


def part_table2(S):
    """Table 2: kappa from 100,000 sequences per length (reference run, quoted
    in the paper) and an independent replicate of 4,000 sequences per length;
    then the full GC 0.20-0.90 x skew 0.0-0.8 range."""
    rows = []
    for tag, reps in (("replicate_4000", 4_000), ("reference_100000", 100_000)):
        rng = _rng(3 if reps == 4_000 else 4)
        for g in GC_T2:
            for s in SKEW_T2:
                p = composition(g, s)
                for L in (20, 40, 60):
                    v = simulate(L, 3, "mirror", p, reps, rng)
                    N = L - 2; cf = closed_form(N, 3, "mirror", p)
                    se = v.std(ddof=1) / np.sqrt(len(v))
                    rows.append(dict(run=tag, GC=g, skew=s, L=L, N=N, sim_mean=v.mean(),
                                     sim_sd=v.std(ddof=1), closed=cf,
                                     kappa_L=(v.mean() - cf) * np.sqrt(N),
                                     kappa_L_se=se * np.sqrt(N)))
    df = pd.DataFrame(rows)
    _save(df, "table2_kappa_per_length.csv")
    med = df.groupby(["run", "GC", "skew"]).kappa_L.median().reset_index()
    wide = med.pivot_table(index=["run", "GC"], columns="skew", values="kappa_L").reset_index()
    wide.columns = ["run", "GC"] + [f"skew_{s:.1f}" for s in SKEW_T2]
    _save(wide, "table2_kappa.csv")
    for tag in ("replicate_4000", "reference_100000"):
        d = df[df.run == tag]; m = med[med.run == tag]
        S[f"t2_{tag}"] = {(r.GC, r.skew): r.kappa_L for r in m.itertuples()}
        S[f"t2_{tag}_perlength_range"] = (d.kappa_L.min(), d.kappa_L.max())
        S[f"t2_{tag}_median_range"] = (m.kappa_L.min(), m.kappa_L.max())
    S["t2_kappaL_se_typical"] = float(df[df.run == "replicate_4000"].kappa_L_se.median())
    # finer grid over the whole stated range, 4,000 sequences per length
    rng = _rng(5); rows = []
    for g in np.round(np.arange(0.20, 0.901, 0.05), 2):
        for s in np.round(np.arange(0.0, 0.801, 0.1), 2):
            p = composition(g, s)
            for L in (20, 40, 60):
                v = simulate(L, 3, "mirror", p, 4_000, rng); N = L - 2
                rows.append(dict(GC=g, skew=s, L=L,
                                 kappa_L=(v.mean() - closed_form(N, 3, "mirror", p)) * np.sqrt(N)))
    fg = pd.DataFrame(rows)
    _save(fg, "kappa_fine_grid_per_length.csv")
    fm = fg.groupby(["GC", "skew"]).kappa_L.median()
    S["fine_median_range"] = (fm.min(), fm.max())
    S["fine_perlength_range"] = (fg.kappa_L.min(), fg.kappa_L.max())


def part_tableS7(S):
    """Supplementary Table S7: pairs, self-paired words, theta departure, kappa and RMS
    residual over 60-6,000 nt; minimum lengths; occupancy."""
    rows, per = [], []
    for k in (3, 4, 5):
        for pr in ("mirror", "rc"):
            npairs, nself = structure(k, pr)
            df, kap, rms = kappa_series(L_TS7, k, pr, U, 20_000, _rng(10 + 2 * k + (pr == "rc")))
            df.insert(0, "pairing", pr); df.insert(0, "k", k); per.append(df)
            # theta departure: max over pairs |theta - 1/2| (pA = pT; independent of GC)
            dep = [theta_departure(k, composition(g, 0.4), pr) for g in (0.2, 0.5, 0.8)]
            rows.append(dict(k=k, pairing=pr, pairs=npairs, self_paired=nself,
                             theta_departure_skew04=dep[1], theta_dep_GC_spread=max(dep) - min(dep),
                             kappa=kap, rms_residual=rms))
    T3 = pd.DataFrame(rows)
    _save(pd.concat(per), "tableS7_kappa_per_length.csv")
    # minimum lengths, mirror pairing, uniform composition, baseline = 0.50
    ml = []
    for k in (3, 4, 5):
        kap = float(T3[(T3.k == k) & (T3.pairing == "mirror")].kappa.iloc[0])
        f_h = lambda L: closed_form(L - k + 1, k)                                   # (a)
        f_s = lambda L: closed_form(L - k + 1, k) + kap / np.sqrt(L - k + 1)       # (b)
        a = min_length(f_h, k, 20_000); b = min_length(f_s, k, 20_000)
        n = min_length(lambda N: closed_form(N, k), 1, 20_000)                     # (c)
        # direct check of (b): simulated sliding-window means bracketing b
        Ls = np.unique(np.round(b * np.array([0.9, 0.95, 1.0, 1.05, 1.1])).astype(int))
        rng = _rng(30 + k)
        mu = np.array([simulate(L, k, "mirror", U, 20_000, rng).mean() for L in Ls])
        sl, ic = np.polyfit(Ls, mu, 1)
        ml.append(dict(k=k, hybrid_closed_N_eq_L_minus_k_plus_1=a,
                       sliding_closed_plus_kappa=b, sliding_simulated=(0.5 - ic) / sl,
                       nonoverlap_N=n, nonoverlap_L=k * n, kappa_used=kap,
                       baseline_at_60nt_hybrid=f_h(60), baseline_at_60nt_sliding=f_s(60),
                       occupancy_60_over_hybrid=60 / a, occupancy_60_over_sliding=60 / b))
    ML = pd.DataFrame(ml)
    T3 = T3.merge(ML[["k", "hybrid_closed_N_eq_L_minus_k_plus_1", "sliding_closed_plus_kappa",
                      "nonoverlap_L"]].assign(pairing="mirror"), on=["k", "pairing"], how="left")
    _save(T3, "tableS7_word_length.csv"); _save(ML, "min_length.csv")
    S["TS7"] = T3; S["ML"] = ML
    # Figure 2B: simulated sliding-window means below the 60-nt fit range, where
    # closed form + kappa/sqrt(N) is not valid for k = 4, 5 (it turns upward)
    rng = _rng(35); rows = []
    for k in (3, 4, 5):
        for L in (20, 30, 40, 50):
            rows.append(dict(k=k, L=L, sim_mean=simulate(L, k, "mirror", U, 20_000, rng).mean()))
    _save(pd.DataFrame(rows), "fig2B_short_sim.csv")


def part_s1(S):
    """S1 against composition: closed form at 300 nt (N = 298), GC x skew grid,
    and the large-N ceiling; Figure 2D data (GC 0.5)."""
    rows = []
    for g in np.round(np.arange(0.1, 0.901, 0.05), 2):
        for s in np.round(np.arange(0.0, 0.901, 0.05), 2):
            p = composition(g, s)
            rows.append(dict(GC=g, skew=s, S1_closed_298=closed_form(298, 3, "rc", p),
                             nMSI_closed_298=closed_form(298, 3, "mirror", p),
                             S1_ceiling=rc_ceiling(3, p)))
    G = pd.DataFrame(rows); _save(G, "s1_composition_grid_300nt.csv")
    z = G[G["skew"] == 0]
    sel = z[(z["GC"] >= 0.3 - 1e-9) & (z["GC"] <= 0.7 + 1e-9)]
    S["s1_gc_skew0_30_70"] = np.ptp(sel.S1_closed_298)
    S["mirror_gc_skew0_30_70"] = np.ptp(sel.nMSI_closed_298)
    S["s1_gc_skew0_20_80"] = np.ptp(z[(z["GC"] >= .2 - 1e-9) & (z["GC"] <= .8 + 1e-9)].S1_closed_298)
    S["s1_gc_skew0_10_90"] = np.ptp(z.S1_closed_298)
    g5 = G[G["GC"] == 0.5]
    S["s1_gc05_skew_0_085"] = np.ptp(g5[g5["skew"] <= 0.85 + 1e-9].S1_closed_298)
    S["s1_gc05_skew_0_08"] = np.ptp(g5[g5["skew"] <= 0.8 + 1e-9].S1_closed_298)
    S["s1_gc05_skew_0_07"] = np.ptp(g5[g5["skew"] <= 0.7 + 1e-9].S1_closed_298)
    S["s1_gc3070_skew08"] = np.ptp(G[(G["GC"] >= .3 - 1e-9) & (G["GC"] <= .7 + 1e-9) & (G["skew"] <= .8 + 1e-9)].S1_closed_298)
    # Figure 2D: closed form + kappa (uniform-fitted, per pairing), Monte Carlo check
    T3 = S["TS7"]
    kap_m = float(T3[(T3.k == 3) & (T3.pairing == "mirror")].kappa.iloc[0])
    kap_r = float(T3[(T3.k == 3) & (T3.pairing == "rc")].kappa.iloc[0])
    rng = _rng(40); rows = []
    for s in np.round(np.arange(0, 0.851, 0.05), 2):
        p = composition(0.5, s)
        r = dict(skew=s, mirror_closed=closed_form(298, 3, "mirror", p) + kap_m / np.sqrt(298),
                 rc_closed=closed_form(298, 3, "rc", p) + kap_r / np.sqrt(298),
                 rc_ceiling=rc_ceiling(3, p), kappa_mirror=kap_m, kappa_rc=kap_r)
        if round(s * 20) % 4 == 0:      # MC at skew 0, 0.2, ..., 0.8
            r["mirror_sim"] = simulate(300, 3, "mirror", p, 20_000, rng).mean()
            r["rc_sim"] = simulate(300, 3, "rc", p, 20_000, rng).mean()
        rows.append(r)
    _save(pd.DataFrame(rows), "fig2D_skew_300nt.csv")


def part_table3(S):
    """Table 3 and Figure 2C: fixed nMSI >= 0.50 against the length-matched
    sliding-window baseline at uniform composition."""
    rng = _rng(50); rows = []
    for L in L_T3:
        v = simulate(L, 3, "mirror", U, 200_000, rng); N = L - 2
        mean_eq4 = closed_form(N) + KAPPA_EQ4 / np.sqrt(N)
        rows.append(dict(L=L, N=N, mean_closed_kappa0615=mean_eq4, mean_sim=v.mean(),
                         sd_sim=v.std(ddof=1)))
    T = pd.DataFrame(rows)
    T["z_sim_sd"] = (0.5 - T.mean_closed_kappa0615) / T.sd_sim
    coef = float((T.sd_sim * np.sqrt(T.N)).mean())     # SD ~ coef/sqrt(N), 15-500 nt
    T["sd_fitted"] = coef / np.sqrt(T.N)
    T["z_fitted_sd"] = (0.5 - T.mean_closed_kappa0615) / T.sd_fitted
    T["z_fully_simulated"] = (0.5 - T.mean_sim) / T.sd_sim
    _save(T, "table3_fixed_criterion.csv"); S["T3"] = T


def main():
    os.makedirs(OUT, exist_ok=True)
    S = {}; t0 = time.time()
    for f in (part_uniform, part_table2, part_tableS7, part_s1, part_table3):
        t = time.time(); f(S); print(f"{f.__name__}: {time.time() - t:.0f} s", flush=True)
    print(f"total {time.time() - t0:.0f} s; outputs in {OUT}")


if __name__ == "__main__":
    main()
