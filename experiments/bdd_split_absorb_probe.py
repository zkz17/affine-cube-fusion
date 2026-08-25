# -*- coding: utf-8 -*-
"""
P4: BDD path two-stage separation probe (n ≤ 7 only; theoretical branch; no production changes).

Variants (path-level, no min):
  full        : split mode + absorb both on (current state a)
  patt_only   : absorb only off
  absorb_only : split mode only off
  none        : both off (≈ P3's b)
Metrics:
  fused ΣT = final value of the global fusion engine;
  raw ΣT   = total cost of the extracted branch pool **before fusion** (Σ t_cost) — used to check
             whether "split directly reduces the total path length per the design criterion";
  m        = number of pool segments.
Output: experiments/phase0_results/bdd_split_absorb_probe.txt
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
from backend.fusion.BranchFusion import fuse_pool_global, pool_t_cost

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'bdd_split_absorb_probe.txt')

PROFILE = {5: (100,), 6: (60,), 7: (30,)}

VARIANTS = ('full', 'patt_only', 'absorb_only', 'none')


def bdd_pool(tt, patterns, absorb):
    orig_split = LimBDDManager.split_with_pattern
    orig_absorb = LimBDDManager.try_absorb_branch
    if not patterns:
        LimBDDManager.split_with_pattern = lambda self, limbdd, pattern: None
    if not absorb:
        LimBDDManager.try_absorb_branch = staticmethod(lambda *a, **k: None)
    try:
        synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
        entries = synth._bdd_entries(tt, None)
        return entries
    finally:
        LimBDDManager.split_with_pattern = orig_split
        LimBDDManager.try_absorb_branch = orig_absorb


def main():
    lines = ['P4:BDD path two-stage separation (raw ΣT before fusion / fused ΣT / pool segment count)', '']
    for n, (cases,) in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        sums = {v: [0, 0, 0] for v in VARIANTS}   # fused, raw, m
        cnt = {v: Counter() for v in VARIANTS}
        t0 = time.perf_counter()
        for fn in data:
            tt = TruthTable(fn, n)
            for v in VARIANTS:
                patt = v in ('full', 'patt_only')
                abso = v in ('full', 'absorb_only')
                entries = bdd_pool(tt, patt, abso)
                raw = sum(cb.t_cost() for cb, _, _ in entries)
                kept, st = fuse_pool_global(entries, n)
                fused = pool_t_cost(kept, st)
                sums[v][0] += fused
                sums[v][1] += raw
                sums[v][2] += len(entries)
        wall = time.perf_counter() - t0
        msg = [f'====== n = {n}({cases} cases,{wall:.1f}s)======']
        for v in VARIANTS:
            f, r, m = sums[v]
            msg.append(f'{v:>11}: fused={f}  raw={r}  m_mean={m/cases:.1f}')
        # Separation effects
        msg.append(f'absorb-only effect (full vs patt_only): fused {sums["full"][0]-sums["patt_only"][0]:+d}, '
                   f'raw {sums["full"][1]-sums["patt_only"][1]:+d}')
        msg.append(f'split-only effect (full vs absorb_only): fused {sums["full"][0]-sums["absorb_only"][0]:+d}, '
                   f'raw {sums["full"][1]-sums["absorb_only"][1]:+d}')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
