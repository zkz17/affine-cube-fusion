# -*- coding: utf-8 -*-
"""
P3:BDD 路径结构拆解探针(仅 n ≤ 7,支持理论分析)。

对照四个变体的「路径级」池 ΣT(不取 min,各自独立评估):
  a : BDD 完整路径(split 模式 + try_absorb_branch 开启,现状)
  b : BDD 去掉 split 模式与 absorb(仅 Shannon/LIM 构造 + split_neg + reorder)
  c : ANF 池(c=0)
  f : FPRM 全枚举最优(平移)
回答:
  Q1: a vs b —— 图重写是否被融合引擎完全覆盖(相等/更好 ⇒ 可移除;更差 ⇒ 引擎有空间)
  Q2: a vs c 在 n 较小处是否真的占优、a vs f 当前后端下 BDD 是否还有贡献;
      占优用例的特征(deg、|ANF|、|BDD 池|)。

输出:experiments/phase0_results/bdd_path_probe.txt
"""
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from limbdd.LimBDDManager import LimBDDManager
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn, anf_degree

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'bdd_path_probe.txt')

PROFILE = {4: (150,), 5: (100,), 6: (60,), 7: (30,)}


def bdd_pool_cost(tt, patterns, absorb):
    """BDD 路径池 ΣT。patterns/absorb=False 时打补丁关闭对应图重写。"""
    orig_split = LimBDDManager.split_with_pattern
    orig_absorb = LimBDDManager.try_absorb_branch
    if not patterns:
        LimBDDManager.split_with_pattern = lambda self, limbdd, pattern: None
    if not absorb:
        LimBDDManager.try_absorb_branch = staticmethod(lambda *a, **k: None)
    try:
        synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
        entries = synth._bdd_entries(tt, None)
        kept, standalone = fuse_pool_global(entries, tt.num_var)
        return pool_t_cost(kept, standalone), len(entries)
    finally:
        LimBDDManager.split_with_pattern = orig_split
        LimBDDManager.try_absorb_branch = orig_absorb


def main():
    lines = ['P3:BDD 路径结构拆解(a=完整, b=去模式/吸收, c=ANF, f=FPRM)', '']
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
            f'Q1(a vs b): 更优 {stat["a<b"]} / 相同 {cases - stat["a<b"] - stat["a>b"]} / 更差 {stat["a>b"]}',
            f'Q2: a 优于 c(ANF) {stat["a<c"]}/{cases};a 优于 f(FPRM) {stat["a<f"]}/{cases};'
            f'b 优于 f {stat["b<f"]}/{cases}',
            f'池规模: BDD(a) 均值 {stat["m_a"]/cases:.1f} vs ANF 均值 {stat["m_c"]/cases:.1f};'
            f'平均 deg {stat["deg"]/cases:.1f}',
        ]
        if w:
            msg.append(f'a<c 的 {w} 例特征: deg {stat["win_deg"]/w:.1f} | '
                       f'BDD 池 {stat["win_m_a"]/w:.1f} vs ANF 池 {stat["win_m_c"]/w:.1f}')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
