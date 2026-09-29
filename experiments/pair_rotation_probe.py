# -*- coding: utf-8 -*-
"""
P2 probe (Plan B entry gate): redemption rate of fixed-point pair rotation vs GL search.

Pair-rotation identity: for a fusible pair (A,B) (d=A⊕B ≠ 0),
for any w ∈ K_d, {A,B} → {A∘τ_w, B∘τ_w} preserves the pool XOR and each segment's T exactly,
while changing the coset position relative to third-party segments (the fusion relation truly changes).

Setup:
  (a) fused fixed point of the FPRM optimal shift (current FS-path final value);
  (b) local search on the fixed-point pool of (a) via pair rotation + full-pool re-fusion (T monotonically decreases, strict improvement);
  (c) GL search (production gl_search, deterministic).

Output: ΣT(a/b/c), redemption rate = (a−b)/(a−c), move/wall-time statistics.
Decision rule: redemption rate ≥ 30% and wall time ≤ GL/10 ⇒ adopt; 15%~30% send to adjudication; <15% abandon.
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.CompleteBranch import CompleteBranch, plan_from_affine
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.ClosedForm import canonical_affine, fuse_closed
from backend.fusion.GLSearch import fprm_best, gl_search

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'pair_rotation_probe.txt')
PROFILE = {5: 20, 6: 20, 7: 20}


def w_candidates(basis, cap=12):
    """Deterministic w candidates from K_d: canonical basis vectors + all pairwise sums (truncated)."""
    out = list(basis)
    k = len(basis)
    for i in range(k):
        for j in range(i + 1, k):
            out.append(basis[i] ^ basis[j])
    return out[:cap]


def rotate_pair(pool, i, j, w, n):
    """Simultaneously shift segments i/j by w in effective-function space (synthetic representation; control count exactly unchanged)."""
    out = list(pool)
    for idx in (i, j):
        cb, r, key = pool[idx]
        base, basis = canonical_affine(cb, r)
        new_cb = CompleteBranch(n, synthetic=plan_from_affine(base ^ w, basis, n))
        out[idx] = (new_cb, r, key)
    return out


def pair_rotation_search(pool0, n, max_moves=200, verbose=False):
    """
    Deterministic local search via pair rotation + full-pool re-fusion (probe setup, not incremental).

    pool0: fixed-point entry list of the fusion [(cb, r, key)].
    Each step enumerates fusible pairs with d≠0 × w candidates and takes the **first** strict improvement (ΣT drops after re-fusion);
    after acceptance the new pool = the fixed point output by re-fusion; stops when there is no improvement.
    T strictly monotonically decreases ⇒ termination is natural.
    Returns (ΣT, moves, wall, statistics).
    """
    cost = pool_t_cost(*fuse_pool_global(pool0, n))
    pool = [(cb, r, key) for cb, r, key in pool0]
    moved = 0
    stat = Counter()
    t0 = time.perf_counter()
    while moved < max_moves:
        aff = [canonical_affine(cb, r) for cb, r, _ in pool]
        m = len(pool)
        found = None
        for i in range(m):
            for j in range(i + 1, m):
                res = fuse_closed(aff[i], aff[j], n)
                if not res.ok or res.cancel:
                    continue
                stat['fusable_pairs'] += 1
                for w in w_candidates(res.basis):
                    new_pool = rotate_pair(pool, i, j, w, n)
                    kept, standalone = fuse_pool_global(new_pool, n)
                    c = pool_t_cost(kept, standalone)
                    if c < cost:
                        found = (c, kept, standalone, w)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            break
        cost = found[0]
        pool = [(cb, r, key) for cb, r, key in found[1]] + [(cb, 0, None) for cb in found[2]]
        moved += 1
        stat['moves'] += 1
    return cost, moved, time.perf_counter() - t0, stat


def main():
    lines = ['P2 probe: fixed-point pair rotation vs GL search realization rate', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        s_a = s_b = s_c = 0
        t_b = t_c = 0.0
        moves_sum = pairs_sum = 0
        better_b = better_c = 0
        for fn in data:
            vF, cF = fprm_best(fn, n)
            pool0 = fuse_pool_global(anf_pool(shift_fn(fn, cF, n), n), n)
            a = pool_t_cost(*pool0)
            entries0 = [(cb, r, key) for cb, r, key in pool0[0]] + \
                       [(cb, 0, None) for cb in pool0[1]]

            t0 = time.perf_counter()
            b, moves, tb, stat = pair_rotation_search(entries0, n)
            t_b += time.perf_counter() - t0

            t0 = time.perf_counter()
            kept_g, st_g, _, _ = gl_search(fn, n)
            t_c += time.perf_counter() - t0
            c = pool_t_cost(kept_g, st_g)

            s_a += a
            s_b += b
            s_c += c
            moves_sum += moves
            pairs_sum += stat['fusable_pairs']
            better_b += b < a
            better_c += c < a

        msg = [
            f'====== n = {n}({cases} cases)======',
            f'ΣT: (a) FPRM fixed point={s_a}  (b) pair rotation={s_b}  (c) GL={s_c}',
            f'relative to (a): b {(s_b/s_a-1):+.2%}  c {(s_c/s_a-1):+.2%}',
            f'realization rate = (a−b)/(a−c) = {(s_a-s_b)/max(s_a-s_c,1):.2%}  (b better {better_b}/{cases}, c better {better_c}/{cases})',
            f'rotation: avg moves {moves_sum/cases:.1f}, avg fusable pairs per round {pairs_sum/max(moves_sum,1):.1f}, '
            f'wall {t_b/cases:.2f}s/case vs GL {t_c/cases:.2f}s/case',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nresults written to {OUT}')


if __name__ == '__main__':
    main()
