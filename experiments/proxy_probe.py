# -*- coding: utf-8 -*-
"""
P1 探针(方案 A 入场门槛):FPRM 免融合代理分 vs 融合 ΣT 的相关性与 top-k 保留率。

代理(均为 anf_pool 中间产物,零额外 Möbius):
  L  = Σ |S|(字面量总数)
  S0 = Σ t(n−|S|)(融合前精确 ΣT,近线性表)
  P  = 差一变量对计数(C3 必然可融对密度)
  S1 = S0 − P(λ=1 组合)

输出:Spearman 秩相关(逐例取均值)、top-k 保留率曲线
      (k ∈ {1,2,4,8,16,32},保留率 = top-k 最优相对全枚举最优的收益比例)。
决策规则(docs/fprm_proxy_and_pair_rotation.md §4):top-8 保留率 ≥ 70% ⇒ 落地。
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import config
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'proxy_probe.txt')
PROFILE = {5: 60, 6: 40, 7: 15}
KS = (1, 2, 4, 8, 16, 32)


def proxies(fn, n):
    """全部平移 c 的 (代理分, 融合真值, c0 值)。只算一次 Möbius/代理 + 一次完整融合。"""
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
    lines = ['P1 探针:代理分 vs 融合 ΣT 的相关性与 top-k 保留率', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        corr = Counter()
        ret = {k: Counter() for k in KS}
        c0_sum = full_sum = 0
        t0 = time.perf_counter()
        for fn in data:
            rows, c0 = proxies(fn, n)
            # 相关性(逐例)
            for name, idx in (('L', 1), ('S0', 2), ('P', 3), ('S1', 4)):
                corr[name] += spearman([r[idx] for r in rows], [r[5] for r in rows])
            # top-k 保留率(score 越小越好)
            full_best = min(r[5] for r in rows)
            for k in KS:
                top = sorted(rows, key=lambda r: r[4])[:k]      # S1 组合分
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
               f'Spearman 均值: L={corr["L"]/cases:+.3f}  S0={corr["S0"]/cases:+.3f}  '
               f'P={corr["P"]/cases:+.3f}  S1={corr["S1"]/cases:+.3f}',
               f'ΣT: c0={c0_sum} 全枚举最优={full_sum}({(full_sum/c0_sum-1):+.2%})']
        for k in KS:
            num, s, skip = ret[k]['num'], ret[k]['sum'], ret[k]['skip']
            msg.append(f'top-{k:>2}: 保留率均值 {s/max(num-skip,1):.2%}  (可测 {num-skip}/{num})')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
