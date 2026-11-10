#!/usr/bin/env python3
"""reversible_memory.py - a reversible background preserves P(w) = P(rev w) but
can still move the expected nMSI, in either direction.

Compares the mean sliding-window nMSI at 300 nt under (a) a reversible
first-order chain with J(i,j) proportional to 1 + 9*delta_ij (self-transition
probability 10/13 = 0.77, uniform stationary composition), (b) independent
uniform bases and (c) a reversible chain that never repeats a base (J proportional to 1 - delta_ij); 20,000 sequences each, seed 20261026. Writes
results/si/reversible_memory.json (cited in the Results and Supplementary Methods S11).

License: MIT
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(20261026)
REV = np.array([(i % 4) * 16 + ((i // 4) % 4) * 4 + i // 16 for i in range(64)])
NONPAL = (np.arange(64) // 16) != (np.arange(64) % 4)


def nmsi_batch(X):
    t = X[:, :-2] * 16 + X[:, 1:-1] * 4 + X[:, 2:]
    C = np.zeros((X.shape[0], 64))
    np.add.at(C, (np.repeat(np.arange(X.shape[0]), t.shape[1]), t.ravel()), 1)
    Cr = C[:, REV]
    return 1 - (np.abs(C - Cr)[:, NONPAL].sum(1) / 2) / ((C + Cr)[:, NONPAL].sum(1) / 2)


def chain(T, pi, n, L):
    X = np.empty((n, L), int); X[:, 0] = rng.choice(4, n, p=pi); cum = np.cumsum(T, 1)
    for t in range(1, L):
        u = rng.random(n); X[:, t] = np.minimum((u[:, None] > cum[X[:, t - 1]]).sum(1), 3)
    return X


J = np.ones((4, 4)) + np.eye(4) * 9; J /= J.sum(); pi = J.sum(1); T = J / pi[:, None]
assert np.allclose(pi[:, None] * T, (pi[:, None] * T).T)       # detailed balance
a = nmsi_batch(chain(T, pi, 20000, 300)); b = nmsi_batch(rng.integers(0, 4, size=(20000, 300)))
# (c) a reversible chain that never repeats a base (J proportional to 1 - delta_ij)
J2 = np.ones((4, 4)) - np.eye(4); J2 /= J2.sum(); pi2 = J2.sum(1); T2 = J2 / pi2[:, None]
assert np.allclose(pi2[:, None] * T2, (pi2[:, None] * T2).T)
c = nmsi_batch(chain(T2, pi2, 20000, 300))
out = dict(self_transition=float(T[0, 0]), L=300, n=20000,
           mean_nmsi_reversible_memory=float(a.mean()), se_reversible=float(a.std() / np.sqrt(len(a))),
           mean_nmsi_independent=float(b.mean()), se_independent=float(b.std() / np.sqrt(len(b))),
           mean_nmsi_reversible_no_repeat=float(c.mean()), se_reversible_no_repeat=float(c.std() / np.sqrt(len(c))))
os.makedirs(os.path.join(ROOT, "results", "si"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "si", "reversible_memory.json"), "w"), indent=1)
print(out)
