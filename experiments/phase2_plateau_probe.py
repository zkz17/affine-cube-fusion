# -*- coding: utf-8 -*-
"""
阶段 2 假设检验探针(G-1a / G-1b):平台移动是否有价值?

对照口径(与生产后端逐行一致):
  A 池 = LimBDDOracleSynthesizer._bdd_entries(tt)(图阶段产物,含根掩码);
  C 池 = anf_pool(tt.bitstr, n)(纯代数产物)。

每池三变体:
  1. strict       : fuse_pool_global(entries, n)                  —— 现状(gain > 0)
  2. neutral      : fuse_pool_global(entries, n, accept_neutral=True)   —— G-1a(gain >= 0)
  3. neutral+cat  : 同上,但池中先注入零成本催化剂三元组 {[x_j=1], [x_j=0], 1} × 所有 j
                    (ΣT=0,XOR=0 ⇒ 池不变量保持)                        —— G-1b

生产级指标:base = min(A_strict, C_strict);improved = min(A_best, C_best)。
判定:improved 相对 base 的 ΣT 缩减是否可观(参照决策惯例阈值 ~0.5%)。

输出:experiments/phase0_results/phase2_plateau_probe.txt
用法:./venv/bin/python experiments/phase2_plateau_probe.py [--cases 200] [4 5 6 7 8]
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.CompleteBranch import CompleteBranch
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'phase2_plateau_probe.txt')


def catalyst_entries(n):
    """零成本催化剂三元组:[x_j=1] ⊕ [x_j=0] ⊕ 1 = 0(t(1)+t(1)+t(0) = 0)。"""
    out = []
    for j in range(n):
        m = 1 << j
        out.append((CompleteBranch(n, mask=m, bitstr=m), 0, ('cat', j, 1)))
        out.append((CompleteBranch(n, mask=m, bitstr=0), 0, ('cat', j, 0)))
        out.append((CompleteBranch(n, mask=0, bitstr=0), 0, ('cat', j, -1)))
    return out


def xor_pool(kept, standalone):
    """池 XOR 不变量,在有效函数空间上验:kept 段先按根掩码 r 平移。"""
    x = 0
    for cb, r, _ in kept:
        x ^= shift_fn(cb.fn(), r, cb.n)
    for cb in standalone:
        x ^= cb.fn()
    return x


def path_variants(entries, n, tt_bitstr, cats):
    """一个池的三个变体;返回 (costs, best_cost, best_name)。校验每变体 XOR 不变量。"""
    results = {}
    k0, s0 = fuse_pool_global(entries, n)
    assert xor_pool(k0, s0) == tt_bitstr, 'strict XOR 不变量被破坏'
    results['strict'] = pool_t_cost(k0, s0)

    k1, s1 = fuse_pool_global(entries, n, accept_neutral=True)
    assert xor_pool(k1, s1) == tt_bitstr, 'neutral XOR 不变量被破坏'
    results['neutral'] = pool_t_cost(k1, s1)

    k2, s2 = fuse_pool_global(entries + cats, n, accept_neutral=True)
    assert xor_pool(k2, s2) == tt_bitstr, 'neutral+cat XOR 不变量被破坏'
    results['neutral+cat'] = pool_t_cost(k2, s2)

    best_name = min(results, key=results.get)
    return results, results[best_name], best_name


def run_case(synth, n, tt, cats):
    entries_a = synth._bdd_entries(tt, None)
    entries_c = anf_pool(tt.bitstr, n)

    ra, ca_a, na = path_variants(entries_a, n, tt.bitstr, cats)
    rc, ca_c, nc = path_variants(entries_c, n, tt.bitstr, cats)

    base = min(ra['strict'], rc['strict'])
    improved = min(ca_a, ca_c)
    assert improved <= base, 'min 组合下不得劣化'
    return base, improved, na if ca_a <= ca_c else nc, ra, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ns', type=int, nargs='*', default=[4, 5, 6, 7, 8])
    ap.add_argument('--cases', type=int, default=300, help='n=8 自动减半')
    args = ap.parse_args()

    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    lines = ['阶段 2 假设检验:G-1a(gain>=0)/ G-1b(催化剂三元组) 的端到端收益测量', '']
    for n in args.ns:
        cases = args.cases if n <= 7 else max(1, args.cases // 3)
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            data = [int(line.rstrip('\n')) for line in f.readlines()[:cases]]
        cats = catalyst_entries(n)

        stat = Counter()
        s_base = s_imp = 0
        t0 = time.perf_counter()
        for tt_int in data:
            tt = TruthTable(tt_int, n)
            base, improved, winner, ra, rc = run_case(synth, n, tt, cats)
            s_base += base
            s_imp += improved
            stat['better'] += improved < base
            stat['same'] += improved == base
            for name in ('strict', 'neutral', 'neutral+cat'):
                if ra[name] <= min(ra.values()):
                    stat[f'A_win_{name}'] += 1
                if rc[name] <= min(rc.values()):
                    stat[f'C_win_{name}'] += 1
            stat['A_best_neutral'] += (ra['neutral'] < ra['strict'])
            stat['A_best_cat'] += (ra['neutral+cat'] < min(ra['strict'], ra['neutral']))
            stat['C_best_neutral'] += (rc['neutral'] < rc['strict'])
            stat['C_best_cat'] += (rc['neutral+cat'] < min(rc['strict'], rc['neutral']))
        wall = time.perf_counter() - t0

        msg = [
            f'====== n = {n}({cases} cases,{wall:.1f}s)======',
            f'ΣT: base={s_base} improved={s_imp}(Δ {s_imp - s_base:+d}, {(s_base - s_imp) / max(s_base, 1):.2%})',
            f'cases: better={stat["better"]} same={stat["same"]}',
            f'A 池最佳变体分布: strict={stat["A_win_strict"]} neutral={stat["A_win_neutral"]} neutral+cat={stat["A_win_neutral+cat"]}',
            f'C 池最佳变体分布: strict={stat["C_win_strict"]} neutral={stat["C_win_neutral"]} neutral+cat={stat["C_win_neutral+cat"]}',
            f'neutral 严格优于 strict: A {stat["A_best_neutral"]} 例 / C {stat["C_best_neutral"]} 例;'
            f' cat 再优于 neutral: A {stat["A_best_cat"]} / C {stat["C_best_cat"]}',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
