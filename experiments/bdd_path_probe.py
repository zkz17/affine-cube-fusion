# -*- coding: utf-8 -*-
"""
P3: BDD path-structure dissection probe (n ≤ 7 only, for theoretical analysis).

Compare the "path-level" pool ΣT of four variants (no min; each evaluated independently):
  a : complete BDD paths (split mode + try_absorb_branch on, current state)
  b : BDD with split mode and absorb removed (Shannon/LIM construction + split_neg + reorder only)
  c : ANF pool (c=0)
  f : FPRM full-enumeration optimum (shift)
Answers:
  Q1: a vs b — is graph rewriting fully covered by the fusion engine (equal/better ⇒ removable; worse ⇒ room for the engine)
  Q2: does a really dominate c at small n, and does BDD still contribute over f under the current backend;
      features of the cases where a wins (deg, |ANF|, |BDD pool|).

Output: experiments/phase0_results/bdd_path_probe.txt
"""
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from limbdd.LimBDDManager import LimBDDManager
from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn, anf_degree

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'bdd_path_probe.txt')

PROFILE = {4: (150,), 5: (100,), 6: (60,), 7: (30,)}


def bdd_pool_cost(tt, patterns, absorb):
    """BDD path-pool ΣT. When patterns/absorb=False, monkey-patch to disable the corresponding graph rewrites."""
    orig_split = LimBDDManager.split_with_pattern
    orig_absorb = LimBDDManager.try_absorb_branch
    if not patterns:
        LimBDDManager.split_with_pattern = lambda self, limbdd, pattern: None
    if not absorb:
        LimBDDManager.try_absorb_branch = staticmethod(lambda *a, **k: None)
    try:
        synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
        entries = synth._bdd_entries(tt, None)
        kept, standalone = fuse_pool_global(entries, tt.num_var)
        return pool_t_cost(kept, standalone), len(entries)
    finally:
        LimBDDManager.split_with_pattern = orig_split
        LimBDDManager.try_absorb_branch = orig_absorb


def main():
    lines = ['P3:BDD path-structure dissection (a=complete, b=no pattern/absorb, c=ANF, f=FPRM)', '']
    for n, (cases,) in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        s = {k: 0 for k in 'abcf'}
        stat = Counter()
        t0 = time.perf_counter()
        for fn in data:
            tt = TruthTable(fn, n)
            a, m_a = bdd_pool_cost(tt, True, True)
            b, m_b = bdd_pool_cost(tt, False, False)
            kept, st = fuse_pool_global(anf_pool(fn, n), n)
            c = pool_t_cost(kept, st)
            best_f = min(pool_t_cost(*fuse_pool_global(anf_pool(shift_fn(fn, cc, n), n), n))
                         for cc in range(1 << n))
            s['a'] += a; s['b'] += b; s['c'] += c; s['f'] += best_f
            stat['a<b'] += a < b; stat['a>b'] += a > b
            stat['a<c'] += a < c; stat['a<c_eq'] += a <= c
            stat['a<f'] += a < best_f; stat['b<f'] += b < best_f
            stat['deg'] += anf_degree(fn, n)
            stat['m_a'] += m_a
            stat['m_c'] += len(anf_pool(fn, n))
            if a < c:
                stat['win_a_c_cases'] += 1
                stat['win_deg'] += anf_degree(fn, n)
                stat['win_m_a'] += m_a
                stat['win_m_c'] += len(anf_pool(fn, n))
        wall = time.perf_counter() - t0
        w = stat['win_a_c_cases']
        msg = [
            f'====== n = {n}({cases} cases,{wall:.1f}s)======',
            f'ΣT: a={s["a"]} b={s["b"]} c={s["c"]} f={s["f"]}',
            f'Q1(a vs b): better {stat["a<b"]} / equal {cases - stat["a<b"] - stat["a>b"]} / worse {stat["a>b"]}',
            f'Q2: a better than c(ANF) {stat["a<c"]}/{cases};a better than f(FPRM) {stat["a<f"]}/{cases};'
            f'b better than f {stat["b<f"]}/{cases}',
            f'pool size: BDD(a) mean {stat["m_a"]/cases:.1f} vs ANF mean {stat["m_c"]/cases:.1f};'
            f'mean deg {stat["deg"]/cases:.1f}',
        ]
        if w:
            msg.append(f'features of the {w} cases with a<c: deg {stat["win_deg"]/w:.1f} | '
                       f'BDD pool {stat["win_m_a"]/w:.1f} vs ANF pool {stat["win_m_c"]/w:.1f}')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
