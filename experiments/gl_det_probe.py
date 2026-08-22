# -*- coding: utf-8 -*-
"""
G-2.2 去随机化对比探针(仅 n ≤ 7)。

对照三种起点策略(其余完全一致:最陡上升、严格改进、步数上限):
  rand  : 32 个随机转vections 乘积起点(固定种子,探针口径)
  det   : 确定性起点 = {恒等 on f, 恒等 on f∘τ_cF} ∪ {全部单 CNOT E_ij on f∘τ_cF}
  single: 仅两个起点(恒等 on f / on f∘τ_cF)——测多起点的必要量
另测 first-improvement 变体 detFI(每步取第一个改进邻居,评估更省)。

指标:ΣT(对 cF 的相对收益)、平均评估次数、wrapper CX。
输出:experiments/phase0_results/gl_det_probe.txt
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
import experiments.gl_probe as G

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'gl_det_probe.txt')

PROFILE = {5: (40, 3), 6: (15, 3), 7: (4, 3)}   # n -> (cases, 步数上限)


def climb_fi(A0, cnots0, cost, n, max_steps):
    """first-improvement 爬山:邻居按序扫描,取第一个严格改进。"""
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
    """返回 (ΣT 值, wrapper CX, 评估数)。strategy ∈ rand/det/single/detFI。"""
    vF, cF = G.fprm_best(fn, n)
    fn_pre = shift_fn(fn, cF, n)
    cache = {}
    cost = G.make_cost(fn, n, cache)
    cache2 = {}
    cost2 = G.make_cost(fn_pre, n, cache2)

    best_v, best_cnots = vF, []          # cF 本身恒为候选(支配链 G ≤ F 的保证)

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
    lines = ['G-2.2 去随机化对比探针(rand vs det vs single vs detFI)', '']
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
            msg.append(f'{k:>7}: ΣT={sT}{rel}  wrapperCX 合计={wx}  平均评估={ev/max(cases,1):.0f}/例')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
