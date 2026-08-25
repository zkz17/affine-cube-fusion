# -*- coding: utf-8 -*-
"""
G-2.2 probe: gain measurement of GL(n,2) linear-subgroup search (phase 3 preliminary study, n ≤ 7 only).

Principle (Optimization_roadmap.md §3.3): the fusion engine is equivariant under AGL(n,2), but the
ANF/Möbius decomposition is bound to the standard coordinate basis ⇒ the pool must be **regenerated**
after a change of coordinates to get a new starting point. The circuit of a linear map A is a CNOT
network (**T-free**, with the cost in CX: wrapper = 2 × #CNOT).

Variants:
  c0 : identity (current ANF path)
  cF : FPRM shift-optimal (G-2.1)
  L  : GL sampling + CNOT-neighborhood climbing (c=0)
  LF : first shift by cF, then GL sampling + climbing (affine combination)
Metrics: fused ΣT, wrapper CX of the best map (2×#CNOT), number of evaluations.

Output: experiments/phase0_results/gl_probe.txt
Usage: ./venv/bin/python experiments/gl_probe.py
"""
import argparse
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'gl_probe.txt')

# n -> (cases, GL sample count, climb step limit)
PROFILE = {
    5: (30, 32, 10),
    6: (12, 32, 8),
    7: (4, 24, 6),
}


def apply_linear(fn, A, n):
    """g(x) = f(Ax). A is represented by row masks: (Ax)_i = parity(A[i] & x)."""
    out = 0
    for x in range(1 << n):
        y = 0
        for i, r in enumerate(A):
            if (r & x).bit_count() & 1:
                y |= 1 << i
        if (fn >> y) & 1:
            out |= 1 << x
    return out


def compose_transvection_right(A, c, t, n):
    """A ∘ E, E = CNOT(c → t) acting on the input: row i XORs bit c when bit t is set."""
    out = A[:]
    for i in range(n):
        if (out[i] >> t) & 1:
            out[i] ^= 1 << c
    return out


def rand_gl(n, k, rng):
    """Product of k random transvections (necessarily invertible); returns (A, cnot list)."""
    A = [(1 << i) for i in range(n)]
    cnots = []
    for _ in range(k):
        c = rng.randrange(n)
        t = rng.randrange(n)
        if c == t:
            continue
        A = compose_transvection_right(A, c, t, n)
        cnots.append((c, t))
    return A, cnots


def make_cost(fn, n, cache):
    def cost(A):
        key = tuple(A)
        if key not in cache:
            kept, standalone = fuse_pool_global(anf_pool(apply_linear(fn, A, n), n), n)
            cache[key] = pool_t_cost(kept, standalone)
        return cache[key]
    return cost


def climb(A0, cnots0, cost, n, max_steps):
    """Steepest ascent over the CNOT neighborhood (strict improvement); returns (A, cnots, value)."""
    A, cnots = A0[:], list(cnots0)
    cur = cost(tuple(A))
    while True:
        best, best_A, best_move = cur, None, None
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                A2 = compose_transvection_right(A, c, t, n)
                v = cost(tuple(A2))
                if v < best:
                    best, best_A, best_move = v, A2, (c, t)
        if best_A is None:
            break
        A = best_A
        cnots.append(best_move)
        cur = best
        if max_steps and len(cnots) - len(cnots0) >= max_steps:
            break
    return A, cnots, cur


def fprm_best(fn, n):
    """Optimal c by full enumeration over the translation subgroup (matching the production _fprm_entries convention for n≤8)."""
    best, best_c = None, 0
    for c in range(1 << n):
        g = shift_fn(fn, c, n)
        kept, standalone = fuse_pool_global(anf_pool(g, n), n)
        v = pool_t_cost(kept, standalone)
        if best is None or v < best:
            best, best_c = v, c
    return best, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20250822)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    lines = ['G-2.2 probe: gain measurement of GL(n,2) linear-subgroup search (n ≤ 7 only)', '']
    for n, (cases, n_sample, max_steps) in PROFILE.items():
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            data = [int(line.rstrip('\n')) for line in f.readlines()[:cases]]

        stat = Counter()
        s_c0 = s_cF = s_L = s_LF = 0
        wrap_cx = 0
        t0 = time.perf_counter()
        for fn in data:
            cache = {}
            cost = make_cost(fn, n, cache)

            c0 = cost(tuple([1 << i for i in range(n)]))
            vF, cF = fprm_best(fn, n)   # vF=cost, cF=translation index (used by the LF variant)

            # ---- Variant L: c=0, GL sampling + climbing ----
            best_A, best_cnots, best_v = None, None, None
            for _ in range(n_sample):
                k = rng.choice([n // 2, n, 3 * n // 2])
                A, cnots = rand_gl(n, max(k, 1), rng)
                A, cnots, v = climb(A, cnots, cost, n, max_steps)
                if best_v is None or v < best_v:
                    best_A, best_cnots, best_v = A, cnots, v
            tL = best_v

            # ---- Variant LF: shift by cF first, then GL ----
            fn_pre = shift_fn(fn, cF, n)
            cache2 = {}
            cost2 = make_cost(fn_pre, n, cache2)
            best_v2, best_cnots2 = None, None
            for _ in range(n_sample):
                k = rng.choice([n // 2, n, 3 * n // 2])
                A, cnots = rand_gl(n, max(k, 1), rng)
                A, cnots, v = climb(A, cnots, cost2, n, max_steps)
                if best_v2 is None or v < best_v2:
                    best_v2, best_cnots2 = v, cnots
            tLF = best_v2

            s_c0 += c0
            s_cF += vF
            s_L += tL
            s_LF += tLF
            wrap_cx += 2 * len(best_cnots) + 2 * len(best_cnots2)
            stat['L_better_than_cF'] += tL < vF
            stat['LF_better_than_cF'] += tLF < vF
            stat['LF_better_than_L'] += tLF < tL
            stat['evals'] += len(cache) + len(cache2)

        wall = time.perf_counter() - t0
        msg = [
            f'====== n = {n}({cases} cases,{wall:.1f}s)======',
            f'ΣT: c0={s_c0} cF={s_cF} L={s_L} LF={s_LF}',
            f'relative to c0: cF {(s_cF/s_c0-1):+.2%}  L {(s_L/s_c0-1):+.2%}  LF {(s_LF/s_c0-1):+.2%}',
            f'relative to cF: L {(s_L/s_cF-1):+.2%}  LF {(s_LF/s_cF-1):+.2%}',
            f'L better than cF: {stat["L_better_than_cF"]}/{cases};LF better than cF: {stat["LF_better_than_cF"]}/{cases};'
            f'LF better than L: {stat["LF_better_than_L"]}/{cases}',
            f'best-map wrapper CX total: {wrap_cx}(avg {wrap_cx/max(cases,1):.1f}/case;'
            f'counted only for winners; must be included in the CX convention if adopted in production)',
            f'avg evaluation count: {stat["evals"]/max(cases,1):.0f}/case',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nResults written to {OUT}')


if __name__ == '__main__':
    main()
