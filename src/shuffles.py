"""shuffles.py — sequence shuffles that preserve composition exactly.

dinuc_shuffle implements the Altschul-Erickson construction (Altschul & Erickson
1985, Mol. Biol. Evol. 2:526), in the form analysed by Kandel et al. (1996,
Discrete Appl. Math. 71:171): a dinucleotide-preserving shuffle is a random
Eulerian path through the multigraph whose vertices are bases and whose edges
are the dinucleotides of the sequence. For every vertex other than the final
base, one outgoing edge is chosen at random to be its last exit; the choice is
accepted only if these last exits form a tree directed into the final base
(otherwise it is redrawn). The remaining exits of each vertex are put in random
order before its last exit, and the path is read off from the first base. Every
draw holds all sixteen dinucleotide counts, the first base and the last base
exactly, and each Eulerian path is equally likely. There is no fallback.

License: MIT
"""
import numpy as np


def mono_shuffle(x, rng):
    """Uniform permutation of the bases: holds composition and length."""
    y = np.array(x, copy=True)
    rng.shuffle(y)
    return y


def dinuc_shuffle(x, rng):
    """Altschul-Erickson dinucleotide-preserving shuffle of an integer array."""
    x = np.asarray(x)
    n = x.size
    if n < 3:
        return x.copy()
    first, last = int(x[0]), int(x[-1])
    out_edges = {}
    for a, b in zip(x[:-1].tolist(), x[1:].tolist()):
        out_edges.setdefault(a, []).append(b)
    verts = [v for v in out_edges if v != last]
    # 1. random last-exit arborescence directed into `last` (rejection)
    while True:
        last_exit = {v: out_edges[v][int(rng.integers(len(out_edges[v])))] for v in verts}
        ok = True
        for v in verts:
            seen, u = set(), v
            while u != last:
                if u in seen or u not in last_exit:
                    ok = False
                    break
                seen.add(u)
                u = last_exit[u]
            if not ok:
                break
        if ok:
            break
    # 2. order every vertex's exits: the others at random, the last exit at the end
    order = {}
    for v, succ in out_edges.items():
        rest = list(succ)
        if v in last_exit:
            rest.remove(last_exit[v])
        rng.shuffle(rest)
        if v in last_exit:
            rest.append(last_exit[v])
        order[v] = rest
    # 3. read off the Eulerian path
    out = np.empty(n, dtype=x.dtype)
    out[0] = first
    ptr = {v: 0 for v in order}
    cur = first
    for t in range(1, n):
        nxt = order[cur][ptr[cur]]
        ptr[cur] += 1
        out[t] = nxt
        cur = nxt
    return out
