# -*- coding: utf-8 -*-
"""
Plan C probe: redemption rate of hill climbing on the G-2.1 shift hypercube relative to full enumeration.

For each test case:
  cost(c) = pool_t_cost(fuse_pool_global(anf_pool(shift_fn(tt, c, n)), n))   (with cache)
  1. full      : optimum of full enumeration over 2^n values of c (ground-truth baseline);
  2. climb0    : local optimum of steepest ascent from c=0 (Hamming neighborhood, strict improvement, cache reuse);
  3. climb2    : two starts, c=0 and c=all-ones, take the better;
  4. rand64    : fixed-seed sample of 64 (the current budget policy for production n≥9), including c=0.

Redemption rate = (cost0 − cost_variant) / (cost0 − cost_full) (skip the case when the denominator is 0).

Output: experiments/phase0_results/fprm_climb_probe.txt
Usage: ./venv/bin/python experiments/fprm_climb_probe.py
"""
import argparse
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'fprm_climb_probe.txt')

# n -> cases
PROFILE = {6: 40, 7: 30, 8: 15, 9: 6, 10: 2}


def make_cost_fn(tt_bitstr, n, cache):
    def cost(c):
        if c in cache:
            return cache[c]
        kept, standalone = fuse_pool_global(anf_pool(shift_fn(tt_bitstr, c, n), n), n)
        v = pool_t_cost(kept, standalone)
        cache[c] = v
        return v
    return cost


def climb(cost, start, n):
    """Steepest ascent: evaluate all n Hamming neighbors at each step (cache reuse); move only on strict improvement."""
    cur = start
    while True:
        best, best_c = cost(cur), None
        for j in range(n):
            nb = cur ^ (1 << j)
            v = cost(nb)
            if v < best:
                best, best_c = v, nb
        if best_c is None:
            break
        cur = best_c
    return cost(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20250821)
    args = ap.parse_args()

    lines = ['Plan-C probe: shift-hypercube climbing vs full enumeration realization rate', '']
    for n, cases in PROFILE.items():
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            data = [int(line.rstrip('\n')) for line in f.readlines()[:cases]]

        rng = random.Random(args.seed)
        stat = Counter()
        s_base = s_full = s_c0 = s_c2 = s_r64 = 0
        t0 = time.perf_counter()
        for tt_int in data:
            tt = TruthTable(tt_int, n)
            cache = {}
            cost = make_cost_fn(tt.bitstr, n, cache)

            base = cost(0)

            # Full enumeration (ground-truth baseline)
            full_best = min(cost(c) for c in range(1 << n))

            # Hill climbing: c=0 single start / c=0 + all-ones dual start.
            # Use a separate cache to count the true evaluation volume (if production adopts climbing, it will not do full enumeration first).
            cache_climb = {}
            cost_climb = make_cost_fn(tt.bitstr, n, cache_climb)
            before = len(cache_climb)
            c0_best = climb(cost_climb, 0, n)
            stat['climb0_evals'] += len(cache_climb) - before
            c2_best = c0_best
            ones = (1 << n) - 1
            if ones != 0:
                before = len(cache_climb)
                b2 = climb(cost_climb, ones, n)
                stat['climb2_evals'] += len(cache_climb) - before
                c2_best = min(c0_best, b2)

            # fixed-seed sampling of 64 (the current n≥9 budget policy)
            sample = [0] + rng.sample(range(1, 1 << n), 63)
            r64_best = min(cost(c) for c in sample)

            s_base += base
            s_full += full_best
            s_c0 += c0_best
            s_c2 += c2_best
            s_r64 += r64_best

            denom = base - full_best
            stat['cases'] += 1
            stat['eval_per_case'] += len(cache)
            if denom > 0:
                stat['room'] += 1
                stat['real_c0'] += (base - c0_best) / denom
                stat['real_c2'] += (base - c2_best) / denom
                stat['real_r64'] += (base - r64_best) / denom
                if c2_best <= full_best:
                    stat['c2_optimal'] += 1
                if r64_best <= full_best:
                    stat['r64_optimal'] += 1

        wall = time.perf_counter() - t0
        room = stat['room']
        msg = [
            f'====== n = {n}({cases} cases,{wall:.1f}s)======',
            f'ΣT: base={s_base} full={s_full} climb0={s_c0} climb2={s_c2} rand64={s_r64}',
            f'relative to base: full {(s_full/s_base-1):+.2%} climb0 {(s_c0/s_base-1):+.2%} '
            f'climb2 {(s_c2/s_base-1):+.2%} rand64 {(s_r64/s_base-1):+.2%}',
            f'realization rate ({room}/{stat["cases"]} cases with improvement room): '
            f'climb0 mean {stat["real_c0"]/max(room,1):.1%}, climb2 {stat["real_c2"]/max(room,1):.1%}, '
            f'rand64 {stat["real_r64"]/max(room,1):.1%}',
            f'hits of the full-enumeration optimum: climb2 {stat["c2_optimal"]}/{room}, rand64 {stat["r64_optimal"]}/{room}',
            f'avg net evaluations: climb0 {stat["climb0_evals"]/max(stat["cases"],1):.0f}, '
            f'climb2 delta {stat["climb2_evals"]/max(stat["cases"],1):.0f}(full enumeration is 2^{n}={1 << n})',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nresults written to {OUT}')


if __name__ == '__main__':
    main()
