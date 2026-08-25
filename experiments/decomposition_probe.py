# -*- coding: utf-8 -*-
"""
Decomposition-path probe (Q1 measurement): compare three "truth table → branch pool" decomposition heuristics.

  A. BDD + reorder(sifting) + fusion   —— current default backend (the main pipeline outside the algebra document);
  B. BDD (initial order only, no reorder) + fusion —— remove reorder;
  C. ANF decomposition + fusion (pure algebra, no graph stage) —— ANF monomials are affine cubes, pool built directly for fusion.

Metrics: post-fusion pool ΣT, per-case runtime, pool size. Samples come from data/test_cases/test_cases_{n}.
Output: experiments/phase0_results/decomposition_probe_{n}.txt
"""
import argparse
import os
import random
import sys
import time

# Project root (the script lives under experiments/, so go up one level); add to sys.path so limbdd/utils can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENTS = os.path.join(ROOT, 'experiments')

from utils.TruthTable import TruthTable
from backend.OracleSynthesizer import OracleSynthesizer
from backend.CompleteBranch import CompleteBranch
from limbdd.Extraction import abs_lim, extract_complete_branches
from backend.fusion.BranchFusion import fuse_pool_global, pool_t_cost, anf_pool


def anf_of(f, n):
    """Fast Möbius transform: truth table → ANF coefficient table (int, bit m = monomial coefficient)."""
    a = f
    size = 1 << n
    for j in range(n):
        step = 1 << j
        block = step << 1
        for base in range(0, size, block):
            for m in range(base, base + step):
                if (a >> m) & 1:
                    a ^= 1 << (m + step)
    return a


def bdd_pool(tt, reorder):
    """BDD graph stage (reorder can be disabled) → global entries."""
    syn = OracleSynthesizer(reorder=reorder, ancilla_num=0, fused_backend=True)
    syn.construct(tt)
    pieces = syn._collect_pieces_fused()
    n = tt.num_var
    entries = []
    for idx, piece in enumerate(pieces):
        r = abs_lim(piece.root.lim, piece.var_order)
        for cb in extract_complete_branches(piece, n):
            entries.append((cb, r, idx))
    return entries, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n5', type=int, default=200)
    ap.add_argument('--n6', type=int, default=200)
    ap.add_argument('--n7', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20250815)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for n, count in ((5, args.n5), (6, args.n6), (7, args.n7)):
        if count <= 0:
            continue
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            lines = f.readlines()
        cases = [int(l.rstrip('\n')) for l in rng.sample(lines, min(count, len(lines)))]
        stats = {'A': [0, 0.0, 0], 'B': [0, 0.0, 0], 'C': [0, 0.0, 0]}  # [ΣT, Σtime, Σpool]
        out_lines = [f'# decomposition path probe n={n} cases={len(cases)}']
        for ci, tt in enumerate(cases):
            ttobj = TruthTable(tt, n)
            for key in ('A', 'B', 'C'):
                t0 = time.perf_counter()
                if key in ('A', 'B'):
                    entries, nn = bdd_pool(ttobj, reorder=(key == 'A'))
                else:
                    entries = anf_pool(ttobj.bitstr, n)
                    nn = n
                    x = 0
                    for cb, _, _ in entries:
                        x ^= cb.fn()
                    assert x == ttobj.bitstr, 'ANF pool XOR invariant violated'
                kept, st = fuse_pool_global(entries, nn)
                t = pool_t_cost(kept, st)
                dt = time.perf_counter() - t0
                stats[key][0] += t
                stats[key][1] += dt
                stats[key][2] += len(entries)
            out_lines.append(f'{ci} {tt} A={stats["A"][0]}(p{stats["A"][2]}) '
                             f'B={stats["B"][0]}(p{stats["B"][2]}) C={stats["C"][0]}(p{stats["C"][2]})')
        summary = [f'# n={n} summary:']
        for key, name in (('A', 'BDD+reorder+fusion (current)'), ('B', 'BDD no-reorder+fusion'), ('C', 'ANF+fusion (pure algebra)')):
            summary.append(f'  {name}: ΣT {stats[key][0]}, time {stats[key][1]:.2f}s, '
                           f'avg initial pool size {stats[key][2]/len(cases):.0f}')
        summary.append(f'  B vs A: {1 - stats["B"][0]/max(1,stats["A"][0]):+.1%}；'
                       f'C vs A: {1 - stats["C"][0]/max(1,stats["A"][0]):+.1%}')
        print('\n'.join(summary))
        out = os.path.join(EXPERIMENTS, 'phase0_results', f'decomposition_probe_{n}.txt')
        with open(out, 'w') as f:
            f.write('\n'.join(out_lines + summary) + '\n')
        print(f'[written to {out}]')


if __name__ == '__main__':
    main()
