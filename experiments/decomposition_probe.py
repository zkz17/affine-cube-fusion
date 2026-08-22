# -*- coding: utf-8 -*-
"""
分解路径探针（Q1 量测）：对比三种"真值表 → 分支池"的分解启发式。

  A. BDD + reorder(sifting) + 融合   —— 当前默认后端（代数文档外的主管线）；
  B. BDD（仅初始序，无 reorder）+ 融合 —— 移除 reorder；
  C. ANF 分解 + 融合（纯代数，无图阶段）—— ANF 单项式即仿射立方，直接建池融合。

指标：融合后池 ΣT、每例耗时、池规模。样本来自 data/test_cases/test_cases_{n}。
输出：experiments/phase0_results/decomposition_probe_{n}.txt
"""
import argparse
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
from limbdd.CompleteBranch import CompleteBranch, abs_lim, extract_complete_branches
from limbdd.fusion.BranchFusion import fuse_pool_global, pool_t_cost, anf_pool


def anf_of(f, n):
    """快速 Möbius 变换：真值表 → ANF 系数表（int，bit m = 单项式系数）。"""
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
    """BDD 图阶段（reorder 可关）→ 全局条目。"""
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
        out_lines = [f'# 分解路径探针 n={n} cases={len(cases)}']
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
                    assert x == ttobj.bitstr, 'ANF 池 XOR 不变量破坏'
                kept, st = fuse_pool_global(entries, nn)
                t = pool_t_cost(kept, st)
                dt = time.perf_counter() - t0
                stats[key][0] += t
                stats[key][1] += dt
                stats[key][2] += len(entries)
            out_lines.append(f'{ci} {tt} A={stats["A"][0]}(p{stats["A"][2]}) '
                             f'B={stats["B"][0]}(p{stats["B"][2]}) C={stats["C"][0]}(p{stats["C"][2]})')
        summary = [f'# n={n} 汇总:']
        for key, name in (('A', 'BDD+reorder+融合(当前)'), ('B', 'BDD无reorder+融合'), ('C', 'ANF+融合(纯代数)')):
            summary.append(f'  {name}: ΣT {stats[key][0]}，耗时 {stats[key][1]:.2f}s，'
                           f'初始池规模均值 {stats[key][2]/len(cases):.0f}')
        summary.append(f'  B vs A: {1 - stats["B"][0]/max(1,stats["A"][0]):+.1%}；'
                       f'C vs A: {1 - stats["C"][0]/max(1,stats["A"][0]):+.1%}')
        print('\n'.join(summary))
        out = os.path.join(EXPERIMENTS, 'phase0_results', f'decomposition_probe_{n}.txt')
        with open(out, 'w') as f:
            f.write('\n'.join(out_lines + summary) + '\n')
        print(f'[已写入 {out}]')


if __name__ == '__main__':
    main()
