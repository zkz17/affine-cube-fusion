# -*- coding: utf-8 -*-
"""
P1 probe (entry bar for plan A): correlation and top-k retention of FPRM fusion-free proxy scores vs fused ΣT.

Proxies (all intermediate products of anf_pool, zero extra Möbius):
  L  = Σ |S| (total literal count)
  S0 = Σ t(n−|S|) (exact pre-fusion ΣT, near-linear table)
  P  = count of pairs differing by one variable (density of C3 necessarily-fusible pairs)
  S1 = S0 − P (λ=1 combination)

Output: Spearman rank correlation (per-case mean), top-k retention curve
      (k ∈ {1,2,4,8,16,32}; retention = the gain share of the top-k best versus the full-enumeration best).
Decision rule (docs/fprm_proxy_and_pair_rotation.md §4): top-8 retention ≥ 70% ⇒ adopt.
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import config
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'proxy_probe.txt')
PROFILE = {5: 60, 6: 40, 7: 15}
KS = (1, 2, 4, 8, 16, 32)


def proxies(fn, n):
    """(proxy score, fused truth, c0 value) for every shift c. Single Möbius/proxy pass plus one full fusion per case."""
    rows = []
    c0_cost = None
    for c in range(1 << n):
        entries = anf_pool(shift_fn(fn, c, n), n)
        masks = [cb.mask for cb, _, _ in entries]
        mset = set(masks)
        L = sum(m.bit_count() for m in masks)
        S0 = sum(config['t_cost'](n - m.bit_count()) for m in masks)
        P = 0
        for m in masks:
            for j in range(n):
                if not (m >> j) & 1 and (m | (1 << j)) in mset:
                    P += 1
        kept, standalone = fuse_pool_global(entries, n)
        fused = pool_t_cost(kept, standalone)
        rows.append((c, L, S0, P, S0 - P, fused))
        if c == 0:
            c0_cost = fused
    return rows, c0_cost


def spearman(xs, ys):
    from scipy.stats import spearmanr
    rho, _ = spearmanr(xs, ys)
    return rho


def main():
    lines = ['P1 probe: proxy score vs fused ΣT correlation and top-k retention', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        corr = Counter()
        ret = {k: Counter() for k in KS}
        c0_sum = full_sum = 0
        t0 = time.perf_counter()
        for fn in data:
            rows, c0 = proxies(fn, n)
            # Correlation (per case)
            for name, idx in (('L', 1), ('S0', 2), ('P', 3), ('S1', 4)):
                corr[name] += spearman([r[idx] for r in rows], [r[5] for r in rows])
            # top-k retention (smaller score is better)
            full_best = min(r[5] for r in rows)
            for k in KS:
                top = sorted(rows, key=lambda r: r[4])[:k]      # S1 combined score
                best = min(r[5] for r in top)
                ret[k]['num'] += 1
                if full_best < c0:
                    ret[k]['sum'] += (c0 - best) / (c0 - full_best)
                else:
                    ret[k]['skip'] += 1
            c0_sum += c0
            full_sum += full_best
        wall = time.perf_counter() - t0
        msg = [f'====== n = {n}({cases} cases,{wall:.1f}s)======',
               f'Spearman mean: L={corr["L"]/cases:+.3f}  S0={corr["S0"]/cases:+.3f}  '
               f'P={corr["P"]/cases:+.3f}  S1={corr["S1"]/cases:+.3f}',
               f'ΣT: c0={c0_sum} full-enumeration best={full_sum}({(full_sum/c0_sum-1):+.2%})']
        for k in KS:
            num, s, skip = ret[k]['num'], ret[k]['sum'], ret[k]['skip']
            msg.append(f'top-{k:>2}: mean retention {s/max(num-skip,1):.2%}  (measurable {num-skip}/{num})')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nresults written to {OUT}')


if __name__ == '__main__':
    main()
