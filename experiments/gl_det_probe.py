# -*- coding: utf-8 -*-
"""
G-2.2 de-randomization comparison probe (n ≤ 7 only).

Compare three starting-point strategies (everything else identical: steepest ascent, strict improvement, step cap):
  rand  : 32 random transvection-product starting points (fixed seed, probe convention)
  det   : deterministic starting points = {identity on f, identity on f∘τ_cF} ∪ {all single CNOT E_ij on f∘τ_cF}
  single: only two starting points (identity on f / on f∘τ_cF) —— measure how necessary multiple starts are
Also test the first-improvement variant detFI (takes the first improving neighbor each step, cheaper evaluation).

Metrics: ΣT (relative gain vs cF), average evaluation count, wrapper CX.
Output: experiments/phase0_results/gl_det_probe.txt
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
import experiments.gl_probe as G

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'gl_det_probe.txt')

PROFILE = {5: (40, 3), 6: (15, 3), 7: (4, 3)}   # n -> (cases, step cap)


def climb_fi(A0, cnots0, cost, n, max_steps):
    """first-improvement hill climbing: scan neighbors in order, take the first strict improvement."""
    A, cnots = A0[:], list(cnots0)
    cur = cost(tuple(A))
    while True:
        moved = False
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                A2 = G.compose_transvection_right(A, c, t, n)
                v = cost(tuple(A2))
                if v < cur:
                    A, cur = A2, v
                    cnots.append((c, t))
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
        if len(cnots) - len(cnots0) >= max_steps:
            break
    return A, cnots, cur


def run_variant(fn, n, strategy, max_steps):
    """Return (ΣT value, wrapper CX, evaluation count). strategy ∈ rand/det/single/detFI."""
    vF, cF = G.fprm_best(fn, n)
    fn_pre = shift_fn(fn, cF, n)
    cache = {}
    cost = G.make_cost(fn, n, cache)
    cache2 = {}
    cost2 = G.make_cost(fn_pre, n, cache2)

    best_v, best_cnots = vF, []          # cF itself is always a candidate (the guarantee of the dominance chain G ≤ F)

    def consider(A, cnots, v):
        nonlocal best_v, best_cnots
        if v < best_v:
            best_v, best_cnots = v, list(cnots)

    ident = [1 << i for i in range(n)]
    if strategy in ('rand',):
        import random
        rng = random.Random(20250822)
        for _ in range(32):
            k = rng.choice([n // 2, n, 3 * n // 2])
            A0, cnots0 = G.rand_gl(n, max(k, 1), rng)
            A, cnots, v = G.climb(A0, cnots0, cost, n, max_steps)
            consider(A, cnots, v)
        for _ in range(32):
            k = rng.choice([n // 2, n, 3 * n // 2])
            A0, cnots0 = G.rand_gl(n, max(k, 1), rng)
            A, cnots, v = G.climb(A0, cnots0, cost2, n, max_steps)
            consider(A, cnots, v)
    elif strategy in ('single',):
        A, cnots, v = G.climb(ident, [], cost, n, max_steps)
        consider(A, cnots, v)
        A, cnots, v = G.climb(ident, [], cost2, n, max_steps)
        consider(A, cnots, v)
    else:  # det / detFI
        climber = climb_fi if strategy == 'detFI' else G.climb
        A, cnots, v = climber(ident, [], cost, n, max_steps)
        consider(A, cnots, v)
        A, cnots, v = climber(ident, [], cost2, n, max_steps)
        consider(A, cnots, v)
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                A0 = G.compose_transvection_right(ident, c, t, n)
                A, cnots, v = climber(A0, [(c, t)], cost2, n, max_steps)
                consider(A, cnots, v)

    return best_v, 2 * len(best_cnots), len(cache) + len(cache2)


def main():
    lines = ['G-2.2 de-randomization comparison probe (rand vs det vs single vs detFI)', '']
    for n, (cases, max_steps) in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = {k: [0, 0, 0] for k in ('cF', 'rand', 'det', 'single', 'detFI')}   # ΣT, wrapCX, evals
        t0 = time.perf_counter()
        for fn in data:
            vF, _ = G.fprm_best(fn, n)
            acc['cF'][0] += vF
            for k in ('rand', 'det', 'single', 'detFI'):
                v, wx, ev = run_variant(fn, n, k, max_steps)
                acc[k][0] += v
                acc[k][1] += wx
                acc[k][2] += ev
        wall = time.perf_counter() - t0
        msg = [f'====== n = {n}({cases} cases,{wall:.1f}s)======']
        for k in ('cF', 'rand', 'det', 'single', 'detFI'):
            sT, wx, ev = acc[k]
            rel = '' if k == 'cF' else f'  (vs cF {(sT/acc["cF"][0]-1):+.2%})'
            msg.append(f'{k:>7}: ΣT={sT}{rel}  total wrapper CX={wx}  avg evals={ev/max(cases,1):.0f}/case')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
