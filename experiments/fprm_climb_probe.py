# -*- coding: utf-8 -*-
"""
C 方案探测:G-2.1 平移超立方上的爬山搜索,相对全枚举的兑现率。

对每个用例:
  cost(c) = pool_t_cost(fuse_pool_global(anf_pool(shift_fn(tt, c, n)), n))   (带缓存)
  1. full      : 全枚举 2^n 个 c 的最优(基准真值);
  2. climb0    : 从 c=0 最陡上升(Hamming 邻域、严格改进、缓存复用)的局部最优;
  3. climb2    : 从 c=0 与 c=全 1 两个起点,取较优;
  4. rand64    : 固定种子采样 64 个(生产 n≥9 的现行预算策略),含 c=0。

兑现率 = (cost0 − cost_variant) / (cost0 − cost_full)(分母为 0 时跳过该例)。

输出:experiments/phase0_results/fprm_climb_probe.txt
用法:./venv/bin/python experiments/fprm_climb_probe.py
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
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

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
    """最陡上升:每步评估全部 n 个 Hamming 邻居(缓存复用),严格改进才移动。"""
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

    lines = ['C 方案探测:平移超立方爬山 vs 全枚举的兑现率', '']
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

            # 全枚举(基准真值)
            full_best = min(cost(c) for c in range(1 << n))

            # 爬山:c=0 单起点 / c=0 + 全1 双起点。
            # 用独立缓存计数真实评估量(生产若选爬山,不会先做全枚举)。
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

            # 固定种子采样 64(现行 n≥9 预算策略)
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
            f'相对 base: full {(s_full/s_base-1):+.2%} climb0 {(s_c0/s_base-1):+.2%} '
            f'climb2 {(s_c2/s_base-1):+.2%} rand64 {(s_r64/s_base-1):+.2%}',
            f'兑现率(有提升空间的 {room}/{stat["cases"]} 例): '
            f'climb0 均值 {stat["real_c0"]/max(room,1):.1%}, climb2 {stat["real_c2"]/max(room,1):.1%}, '
            f'rand64 {stat["real_r64"]/max(room,1):.1%}',
            f'命中全枚举最优: climb2 {stat["c2_optimal"]}/{room}, rand64 {stat["r64_optimal"]}/{room}',
            f'平均净评估次数: climb0 {stat["climb0_evals"]/max(stat["cases"],1):.0f}, '
            f'climb2 增量 {stat["climb2_evals"]/max(stat["cases"],1):.0f}(全枚举为 2^{n}={1 << n})',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
