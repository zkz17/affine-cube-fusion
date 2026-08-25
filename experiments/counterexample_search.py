# -*- coding: utf-8 -*-
"""
Random counterexample search for the pairwise completeness conjecture (open problem in CompleteBranch_algebra.md §4).

Conjecture (k=3 base case): if the XOR of three affine cubes (branch segments) is an affine cube D,
then there must exist a pair (i,j) whose XOR is an affine cube (C1–C4, i.e. 2-to-1 fusible).

Search: randomly generate triples of CompleteBranch (random SyntheticPlan),
and select those where d = f1^f2^f3 is an affine cube and no pair is fusible (is_fusable=False) = counterexample.

Usage: python counterexample_search.py n examples [seed]
Output: stdout + experiments/phase0_results/cex_search_{n}_{seed}.txt
"""
import argparse
import os
import random
import sys
import time

# Project root (script lives under experiments/, so go up one level); add to sys.path so limbdd/utils is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENTS = os.path.join(ROOT, 'experiments')

from backend.CompleteBranch import CompleteBranch, SyntheticPlan, is_affine_cube
from backend.fusion.BranchFusion import is_fusable


def random_cube(n, rng):
    """Random affine cube: random X translation + random CNOT chain + random control subset/polarity."""
    x_trans = rng.getrandbits(n) & ((1 << n) - 1)
    cnot_ops = []
    for _ in range(rng.randrange(0, n + 1)):
        c = rng.randrange(n)
        t = rng.randrange(n)
        if c != t:
            cnot_ops.append((c, t))
    controls = tuple((v, rng.randrange(2)) for v in range(n) if rng.randrange(2))
    return CompleteBranch(n, synthetic=SyntheticPlan(x_trans, tuple(cnot_ops), controls))


def search(n, examples, seed):
    rng = random.Random(seed)
    total = 0          # total number of triples
    cand = 0           # candidates where d is an affine cube
    cex = 0            # counterexamples (candidate and no pair fusible)
    dim_hist = {}
    t0 = time.time()
    lines = [f'# pairwise-completeness counterexample search  n={n}  examples={examples}  seed={seed}']

    for i in range(examples):
        a = random_cube(n, rng)
        b = random_cube(n, rng)
        c = random_cube(n, rng)
        d = a.fn() ^ b.fn() ^ c.fn()
        total += 1
        ok, dim = is_affine_cube(d, n)
        if not ok:
            continue
        cand += 1
        dim_hist[dim] = dim_hist.get(dim, 0) + 1
        if is_fusable(a, b) or is_fusable(a, c) or is_fusable(b, c):
            continue
        cex += 1
        pab = is_affine_cube(a.fn() ^ b.fn(), n)
        pac = is_affine_cube(a.fn() ^ c.fn(), n)
        pbc = is_affine_cube(b.fn() ^ c.fn(), n)
        msg = (f'COUNTEREXAMPLE #{cex} at example {i}:\n'
               f'  A = {a.synthetic!r}\n  B = {b.synthetic!r}\n  C = {c.synthetic!r}\n'
               f'  D dim = {dim}\n'
               f'  pair (affine?, dim): AB=({pab[0]}, {pab[1]}), AC=({pac[0]}, {pac[1]}), BC=({pbc[0]}, {pbc[1]})')
        print(msg)
        lines.append(msg)

    lines += [
        '',
        f'# stats: total {total}, d-affine candidates {cand} ({cand/max(1,total):.2%}), counterexamples {cex}',
        f'# candidate dim histogram: {dict(sorted(dim_hist.items()))}',
        f'# time {time.time()-t0:.1f}s',
    ]
    print('\n'.join(lines))
    out_dir = os.path.join(EXPERIMENTS, 'phase0_results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'cex_search_{n}_{seed}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[written to experiments/phase0_results/cex_search_{n}_{seed}.txt]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('n', type=int)
    ap.add_argument('examples', type=int)
    ap.add_argument('seed', type=int, nargs='?', default=20250815)
    args = ap.parse_args()
    search(args.n, args.examples, args.seed)


if __name__ == '__main__':
    main()
