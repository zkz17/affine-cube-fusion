# -*- coding: utf-8 -*-
"""
Phase 6 残差测量：成对完备性在真实流水线池上的工程检验（决策依据）。

层一（本征残差）：对 fuse_pool_global 不动点池（已无任何正收益 2-to-1 对），
  - k=3：全枚举三元组，d = f_i⊕f_j⊕f_k 为仿射立方且正收益者 = 2-to-1 贪心不可达的机会；
  - k≥4：随机抽样子集（size 4..5）同样判定（抽样口径，见输出）；
层二（贪心残差）：随机打乱条目顺序重跑 fuse_pool_global 若干次，观察最终 ΣT 是否变化
  （顺序敏感性 ⟹ 贪心匹配存在被锁死的机会）。

用法：python residual_measurement.py [--n4 200 --n5 200 --n6 100 --n7 50 ...]
输出：stdout + experiments/phase0_results/residual_measurement_{n}.txt
"""
import argparse
import collections
import os
import random
import sys
import time

# 项目根目录（脚本位于 experiments/ 下，故上溯一级）；加入 sys.path 保证 limbdd/utils 可导入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENTS = os.path.join(ROOT, 'experiments')

from utils.TruthTable import TruthTable
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.CompleteBranch import abs_lim, extract_complete_branches, is_affine_cube, t_cost
from limbdd.fusion.BranchFusion import fuse_pool_global, shift_fn


def case_entries(tt, reorder=True):
    """完整图阶段（与 process_fused_global 同构）后收集 (cb, root_mask, piece_idx) 条目。"""
    syn = LimBDDOracleSynthesizer(reorder=reorder, ancilla_num=0, fused_backend=True)
    syn.construct(tt)
    pieces = syn._collect_pieces_fused()
    n = tt.num_var
    entries = []
    for idx, piece in enumerate(pieces):
        r = abs_lim(piece.root.lim, piece.var_order)
        for cb in extract_complete_branches(piece, n):
            entries.append((cb, r, idx))
    return entries, n


def final_pool_eff(entries, n, rng=None):
    """fuse_pool_global 不动点池 → [(cb, eff_fn, t)]；rng 提供时打乱条目顺序（层二）。"""
    if rng is not None:
        entries = rng.sample(entries, len(entries))
    kept, standalone = fuse_pool_global(entries, n)
    pool = [(cb, shift_fn(cb.fn(), r, n), cb.t_cost()) for cb, r, _ in kept]
    pool += [(cb, cb.fn(), cb.t_cost()) for cb in standalone]
    return pool


def measure_case(entries, n, k4_sizes, k4_samples, shuffles, rng):
    pool = final_pool_eff(entries, n)
    n_pool = len(pool)

    # ---- 层一 k=3 全枚举 ----
    hits3, nonpos3 = [], 0
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            for k in range(j + 1, n_pool):
                d = pool[i][1] ^ pool[j][1] ^ pool[k][1]
                ok, dim = is_affine_cube(d, n)
                if not ok:
                    continue
                gain = pool[i][2] + pool[j][2] + pool[k][2] - t_cost(n - dim)
                if gain > 0:
                    hits3.append((dim, gain))
                else:
                    nonpos3 += 1

    # ---- 层一 k>=4 抽样 ----
    hits4 = []
    for s in k4_sizes:
        if n_pool < s:
            continue
        for _ in range(k4_samples):
            idxs = rng.sample(range(n_pool), s)
            d = 0
            tsum = 0
            for i in idxs:
                d ^= pool[i][1]
                tsum += pool[i][2]
            ok, dim = is_affine_cube(d, n)
            if not ok:
                continue
            gain = tsum - t_cost(n - dim)
            if gain > 0:
                hits4.append((s, dim, gain))

    # ---- 层二 贪心顺序敏感性 ----
    t_vals = []
    for _ in range(shuffles):
        p2 = final_pool_eff(entries, n, rng)
        t_vals.append(sum(x[2] for x in p2))

    return {
        'n_pool': n_pool,
        't_final': sum(x[2] for x in pool),
        'hits3': hits3,
        'nonpos3': nonpos3,
        'hits4': hits4,
        't_shuffle_min': min(t_vals),
        't_shuffle_max': max(t_vals),
    }


def load_cases(n, count, seed):
    with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
        lines = f.readlines()
    rng = random.Random(seed * 100 + n)
    return [int(l.rstrip('\n')) for l in rng.sample(lines, min(count, len(lines)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n4', type=int, default=200)
    ap.add_argument('--n5', type=int, default=200)
    ap.add_argument('--n6', type=int, default=100)
    ap.add_argument('--n7', type=int, default=50)
    ap.add_argument('--k4-samples', type=int, default=500, help='每个 k>=4 尺寸的抽样子集数')
    ap.add_argument('--shuffles', type=int, default=10, help='层二打乱重跑次数')
    ap.add_argument('--seed', type=int, default=20250815)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for n, count in ((4, args.n4), (5, args.n5), (6, args.n6), (7, args.n7)):
        if count <= 0:
            continue
        cases = load_cases(n, count, args.seed)
        agg = collections.Counter()
        lines = [f'# 残差测量 n={n} cases={len(cases)} k4_samples={args.k4_samples} shuffles={args.shuffles}']
        t0 = time.time()
        for ci, tt in enumerate(cases):
            try:
                entries, nn = case_entries(TruthTable(tt, n), reorder=True)
                if not entries:
                    continue
                m = measure_case(entries, nn, (4, 5), args.k4_samples, args.shuffles, rng)
                agg['cases'] += 1
                agg['n_pool'] += m['n_pool']
                agg['t_final'] += m['t_final']
                agg['hits3'] += len(m['hits3'])
                agg['hits3_gain'] += sum(g for _, g in m['hits3'])
                agg['nonpos3'] += m['nonpos3']
                agg['hits4'] += len(m['hits4'])
                agg['hits4_gain'] += sum(g for _, _, g in m['hits4'])
                if m['t_shuffle_min'] < m['t_shuffle_max']:
                    agg['greedy_vary'] += 1
                    agg['greedy_gap'] += m['t_shuffle_max'] - m['t_shuffle_min']
                if m['hits3'] or m['hits4'] or m['t_shuffle_min'] < m['t_shuffle_max']:
                    line = (f'  [{ci}] tt={tt} N={m["n_pool"]} T={m["t_final"]} '
                            f'hits3={len(m["hits3"])}(gain {sum(g for _, g in m["hits3"])}) '
                            f'hits4={len(m["hits4"])} nonpos3={m["nonpos3"]} '
                            f'shuffle T [{m["t_shuffle_min"]}, {m["t_shuffle_max"]}]')
                    print(line)
                    lines.append(line)
            except Exception as e:
                print(f'  [{ci}] tt={tt} FAILED: {type(e).__name__}: {e}')
                lines.append(f'  [{ci}] tt={tt} FAILED: {type(e).__name__}: {e}')

        nc = agg['cases']
        summary = [
            '',
            f'# n={n} 汇总（{nc} cases，{time.time()-t0:.1f}s）:',
            f'  不动点池规模均值: {agg["n_pool"]/max(1,nc):.1f}; 最终 ΣT: {agg["t_final"]}',
            f'  层一 k=3 正收益残差: {agg["hits3"]} 个（收益上界 {agg["hits3_gain"]}）；'
            f'非正收益仿射三元组: {agg["nonpos3"]}',
            f'  层一 k>=4 抽样残差: {agg["hits4"]} 个（收益上界 {agg["hits4_gain"]}）',
            f'  层二 贪心顺序敏感: {agg["greedy_vary"]}/{nc} cases 有 T 变化（总缺口 {agg["greedy_gap"]}）',
        ]
        print('\n'.join(summary))
        lines += summary
        out = os.path.join(EXPERIMENTS, 'phase0_results', f'residual_measurement_{n}.txt')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'[已写入 {out}]')


if __name__ == '__main__':
    main()
