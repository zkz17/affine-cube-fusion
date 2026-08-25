# -*- coding: utf-8 -*-
"""
Macro M2 production implementation (P9-finalized form, 2025-08-22).

Theory: docs/macro_m2_theory.md §2-3. For a δ=2 pair (A,B) and a candidate h (the classical
exorlink divergent-literal family, P6a mechanism), first apply the two-sided rewrite
{A,B} → {A⊕h, B⊕h} (pair move imm, at a fixed point necessarily ≤ 0, otherwise the δ=2
immediate move would already have been taken), then pick a third segment C to fuse with A′
(g_fuse > 0 and total gain imm + g_fuse > 0), applying:

    {A, B, C} → { B⊕h, (A⊕h)⊕C }

The B-side symmetric twin is enumerated as well. The macro strictly decreases
(XOR invariance is guaranteed by h⊕h=0), so per-case T never regresses; measured time cost
0.1~2.9 ms/case at n≤7 (P9); the production integration is behind a switch (default on).

Production notes:
  - C prescreen: a necessary condition for gain(A′,C)>0 is dim(K) differing by ≤1
    (theory doc generalization #6); dim bucketing reduces the C scan from O(m) to per-bucket;
    `m2_best_move(pool, n, prescreen=…)` can be checked against a full scan
    (tests/test_macro_m2.py does the differential verification);
  - round cap max_rounds (default 30, same as the P9 probe);
  - the XOR-invariant assertion runs only under LIMBDD_FUSION_DEBUG=1 (O(m·2^n), off in
    production for n≥8).
"""
from collections import defaultdict

from backend.CompleteBranch import CompleteBranch, plan_from_affine
from backend.fusion.BranchFusion import fuse_pool_global, shift_fn, FUSION_DEBUG
from backend.fusion.ClosedForm import canonical_affine, fuse_closed
from config import config

TC = config['t_cost']


# ------------------------------------------------------------- P6a mechanics port ----

def rows_of(cb, r, n):
    """Canonical constraint rows of a general affine segment: ((mask, const), ...), with the
    root-mask shift baked in (base ^= r).

    affine_rows() strips the pivot bit from row masks (the pivot is implicit); add back (1 << pivot).
    """
    base, basis = canonical_affine(cb, r)
    plan = plan_from_affine(base, basis, n)
    tmp = CompleteBranch(n, synthetic=plan)
    return tuple(((1 << col) | mask, const) for col, mask, const in tmp.affine_rows())


def pair_info(ra, rb, n):
    """Classical distance δ between two constraint-row sets and the raw material for candidate h
    (all single-row divergent candidates)."""
    da = dict((m, c) for m, c in ra)
    db = dict((m, c) for m, c in rb)
    agree, opp, oneside_a, oneside_b = [], [], [], []
    for m in da:
        if m in db:
            if da[m] == db[m]:
                agree.append((m, da[m]))
            else:
                opp.append(m)
        else:
            oneside_a.append((m, da[m]))
    for m in db:
        if m not in da:
            oneside_b.append((m, db[m]))
    return len(opp) + len(oneside_a) + len(oneside_b), agree, opp, oneside_a, oneside_b


def h_candidates(agree, opp, oneside_a, oneside_b):
    """Constraint-row lists of candidate h: one row per divergent variable (opp takes one from each side)."""
    out = []
    for m in opp:
        out.append(agree + [(m, 0)])
        out.append(agree + [(m, 1)])
    for m, c in oneside_a:
        out.append(agree + [(m, c)])
    for m, c in oneside_b:
        out.append(agree + [(m, c)])
    return out


def cb_from_rows(rows, n):
    """Constraint-row set → CompleteBranch (synthetic representation). Rows must be linearly independent."""
    pivots = {}
    for m, c in rows:
        mm, cc = m, c
        while mm:
            low = mm & (-mm)
            if low in pivots:
                mm ^= pivots[low][0]
                cc ^= pivots[low][1]
            else:
                pivots[low] = (mm, cc)
                break
        if mm == 0 and cc:
            return None          # unsatisfiable, not a cube
    basis, base = [], 0
    pivot_cols = {v.bit_length() - 1 for v in pivots}
    for f in range(n):
        if f in pivot_cols:
            continue
        v = 1 << f
        for m, c in pivots.values():
            if (m >> f) & 1:
                v |= 1 << (m & (-m)).bit_length() - 1
        basis.append(v)
    for m, c in pivots.values():
        if c:
            base |= 1 << ((m & (-m)).bit_length() - 1)
    return CompleteBranch(n, synthetic=plan_from_affine(base, basis, n))


# ------------------------------------------------------------- macro M2 ---------

def seg_from(res, root, n):
    return CompleteBranch(n, synthetic=plan_from_affine(res.base ^ root, res.basis, n))


def _pool_xor(pool, n):
    x = 0
    for cb, r, _ in pool:
        x ^= shift_fn(cb.fn(), r, n)
    return x


def m2_best_move(pool, n, prescreen=True):
    """
    Best macro M2 on a fixed-point pool. Returns (total, idxA, idxB, k, rA, rB, rc) or None.

    total = imm + g_fuse > 0 is the true total gain:
      imm     = t(A)+t(B) − t(A⊕h) − t(B⊕h)   (pair move; ≤ 0 at a fixed point);
      g_fuse  = t(A⊕h)+t(C) − t((A⊕h)⊕C)     (A′+C fusion gain).
    With prescreen=True, C is only scanned in the buckets whose dim(K) differs from rA.dim by ≤1
    (theory generalization #6); differential correctness is verified against a full scan by
    tests/test_macro_m2.py.
    """
    aff = [canonical_affine(cb, r) for cb, r, _ in pool]
    dims = [len(basis) for _, basis in aff]
    rows = [rows_of(pool[i][0], pool[i][1], n) for i in range(len(pool))]
    if prescreen:
        buckets = defaultdict(list)
        for k, d in enumerate(dims):
            buckets[d].append(k)

        def c_candidates(d_a):
            out = []
            for d in (d_a - 1, d_a, d_a + 1):
                out.extend(buckets[d])
            return out
    else:
        def c_candidates(d_a):
            return range(len(pool))

    m = len(pool)
    best = None
    for i in range(m):
        ti = pool[i][0].t_cost()
        for j in range(i + 1, m):
            d, agree, opp, oa, ob = pair_info(rows[i], rows[j], n)
            if d != 2:
                continue
            for rows_h in h_candidates(agree, opp, oa, ob):
                ch = cb_from_rows(rows_h, n)
                if ch is None:
                    continue
                ah = canonical_affine(ch)
                r1 = fuse_closed(aff[i], ah, n)
                r2 = fuse_closed(aff[j], ah, n)
                if not (r1.ok and r2.ok and not r1.cancel and not r2.cancel):
                    continue
                imm = ti + pool[j][0].t_cost() - TC(n - r1.dim) - TC(n - r2.dim)
                for rA, rB, idxA, idxB in ((r1, r2, i, j), (r2, r1, j, i)):
                    tA = TC(n - rA.dim)
                    affA = (rA.base, rA.basis)
                    for k in c_candidates(rA.dim):
                        if k == idxA or k == idxB:
                            continue
                        rc = fuse_closed(affA, aff[k], n)
                        if not rc.ok or rc.cancel:
                            continue
                        g_fuse = tA + pool[k][0].t_cost() - TC(n - rc.dim)
                        total = imm + g_fuse
                        if total > 0 and (best is None or total > best[0]):
                            best = (total, idxA, idxB, k, rA, rB, rc)
    return best


def m2_interleaved(kept, standalone, n, max_rounds=30):
    """
    Macro M2 interleaved schedule (production form, P9 convention): the pool must already be fused
    to a fixed point.

    Loop: best macro M2 → apply → fuse_pool_global re-fusion, at most max_rounds iterations;
    returns (kept, standalone) with the same structure as fuse_pool_global (emission grouping
    semantics unchanged). Every step strictly decreases, so per-case T never regresses; under
    LIMBDD_FUSION_DEBUG=1 the XOR invariant is asserted every round.
    """
    pool = [(cb, r, key) for cb, r, key in kept] + [(cb, 0, None) for cb in standalone]
    if FUSION_DEBUG:
        expected_xor = _pool_xor(pool, n)
    for _ in range(max_rounds):
        mv = m2_best_move(pool, n)
        if mv is None:
            break
        total, idxA, idxB, k, rA, rB, rc = mv
        b1 = seg_from(rB, pool[idxB][1], n)
        fused = seg_from(rc, 0, n)
        pool = [e for q, e in enumerate(pool) if q not in (idxA, idxB, k)] + \
               [(b1, pool[idxB][1], pool[idxB][2]), (fused, 0, None)]
        if FUSION_DEBUG:
            assert _pool_xor(pool, n) == expected_xor, 'M2 move broke the pool XOR invariant'
        kept2, st2 = fuse_pool_global(pool, n)
        pool = [(cb, r, key) for cb, r, key in kept2] + [(cb, 0, None) for cb in st2]
        if FUSION_DEBUG:
            assert _pool_xor(pool, n) == expected_xor, 'post-M2 re-fusion broke the pool XOR invariant'
    kept = [(cb, r, key) for cb, r, key in pool if key is not None]
    standalone = [cb for cb, r, key in pool if key is None]
    return kept, standalone
