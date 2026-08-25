# -*- coding: utf-8 -*-
"""
Plan A: FPRM fusion-free proxy prescreen (probe P1 passed, top-8 retention 71.7%~84.6%).

Flow:
  enumerate all 2^n shifts, each only doing the Möbius transform (no fusion), computing proxy scores
    S0 = Σ t(n−|S|)   (exact pre-fusion ΣT)
    P  = count of one-variable-apart pairs (density of C3 necessarily-fusible pairs)
    S1 = S0 − P       (smaller is better)
  take the top-k candidates (sorted by (S1, c) ascending, deterministic) and always include c=0;
  full fusion runs only on these k+1 candidates.

Design position (choice relation, NOT min): with fprm_proxy=True the proxy version **replaces** the
default F path
  - n ≥ 9: proxy full enumeration + top-k fusion, intended to upgrade the current shift climb2
    (a more complete enumeration);
  - n ≤ 8: default F is full-enumeration complete fusion, the proxy version is 12%~28% worse in quality
    (P1 measurements) -- the choice relation allows switching on demand; no min, so tests are not slowed
    and differences are not masked (owner's instruction, exploration-phase discipline).
Always including c=0 implies the proxy result ≤ the C path (dominance chain preserved).
"""
from config import config
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn


def proxy_scores(fn, n):
    """Proxy scores for all 2^n shifts; returns [(S1, c, S0, P)]."""
    out = []
    for c in range(1 << n):
        entries = anf_pool(shift_fn(fn, c, n), n)
        masks = [cb.mask for cb, _, _ in entries]
        mset = set(masks)
        s0 = sum(config['t_cost'](n - m.bit_count()) for m in masks)
        p = 0
        for m in masks:
            for j in range(n):
                if not (m >> j) & 1 and (m | (1 << j)) in mset:
                    p += 1
        out.append((s0 - p, c, s0, p))
    return out


def fprm_proxy_best(fn, n, k=8):
    """
    Full fusion of the top-k proxy candidates (always including c=0); returns (kept, standalone, best_c).

    Deterministic: candidates are the first k sorted by (S1, c) ascending; full-fusion results are
    compared in evaluation order, ties keep the earlier one.
    """
    rows = sorted(proxy_scores(fn, n), key=lambda r: (r[0], r[1]))
    cands = [c for _, c, _, _ in rows[:k]]
    if 0 not in cands:
        cands.append(0)
    best = None
    for c in cands:
        kept, standalone = fuse_pool_global(anf_pool(shift_fn(fn, c, n), n), n)
        v = pool_t_cost(kept, standalone)
        if best is None or v < best[0]:
            best = (v, kept, standalone, c)
    return best[1], best[2], best[3]
