# -*- coding: utf-8 -*-
"""
P5(诊断):SplitLv2Pattern 类型 3「借 0 边」的确切代数快照。

流程:构造随机小函数 → 定位 type-3 候选 → 抽取拆分前分支池 → 执行拆分 →
抽取拆分后分支池(原图 + 新片) → 找出只存在于前/后的段,打印函数与代价,
验证:恒等式、代价变化、「0 = 1⊕1」的借用结构。

输出:experiments/phase0_results/split_type3_snapshot.txt
"""
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from limbdd.LimBDD import LimBDD
from limbdd.LimBDDManager import LimBDDManager
from limbdd.CompleteBranch import extract_complete_branches, abs_lim
from limbdd.patterns.SplitLv2Pattern import SplitLv2Pattern
from limbdd.fusion.BranchFusion import anf_degree, shift_fn
from limbdd.CompleteBranch import t_cost

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'split_type3_snapshot.txt')


def pool_of(limbdd, n):
    """整图分支池(有效函数:根掩码平移已换算):{(fn, cost): count}。"""
    r = abs_lim(limbdd.root.lim, limbdd.var_order)
    c = Counter()
    for cb in extract_complete_branches(limbdd, n, anomalies=Counter()):
        fn = shift_fn(cb.fn(), r, n)
        c[(fn, cb.t_cost(), cb.nctrl())] += 1
    return c


def pool_of_pieces(pieces, n):
    c = Counter()
    for g in pieces:
        r = abs_lim(g.root.lim, g.var_order)
        for cb in extract_complete_branches(g, n, anomalies=Counter()):
            fn = shift_fn(cb.fn(), r, n)
            c[(fn, cb.t_cost(), cb.nctrl())] += 1
    return c


def find_type3(mgr):
    """返回 (node, level) 或 None。"""
    pat = SplitLv2Pattern()
    for level in range(2, mgr.limbdd.root.level + 1):
        cands = pat.identify_candidates(mgr.limbdd, level)
        for cand in cands:
            node, connect_list, succ_l, succ_r, ptype, _ = cand
            if ptype == 3:
                return cand, level
    return None, None


def main():
    out = []
    found = 0
    t0 = time.perf_counter()
    for n in (4, 5):
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            tts = [int(l.rstrip('\n')) for l in f.readlines()[:400]]
        for tt_int in tts:
            tt = TruthTable(tt_int, n)
            mgr = LimBDDManager()
            mgr.initialize(tt, list(range(n)))
            cand, level = find_type3(mgr)
            if cand is None:
                continue

            node, connect_list, succ_l, succ_r, ptype, succ_info = cand
            before = pool_of(mgr.limbdd, n)

            # 执行拆分(只对主图;新片进入 mgr.temp_list)
            mgr.perform_split(mgr.limbdd)
            mgr.append_temp_to_list()
            pieces = [mgr.limbdd]
            pieces += [g for gs in mgr.limbdd_list.values() for g in gs]
            after = pool_of_pieces(pieces, n)

            only_before = before - after
            only_after = after - before
            s_before = sum(k[1] * v for k, v in only_before.items())
            s_after = sum(k[1] * v for k, v in only_after.items())

            x_b = 0
            for (fn, _, _), cnt in before.items():
                x_b ^= fn * (cnt & 1)
            x_a = 0
            for (fn, _, _), cnt in after.items():
                x_a ^= fn * (cnt & 1)

            msg = [f'--- 例 n={n} tt={tt_int}(deg={anf_degree(tt_int, n)}, 触发层 {level}) ---',
                   f'全池 XOR 不变: {x_b == x_a}({x_b:0{1 << n}b});'
                   f'池段数 {sum(before.values())} → {sum(after.values())}',
                   f'拆分前独有段({sum(only_before.values())} 条, Σcost={s_before}):']
            for (fn, cost, k), cnt in only_before.items():
                msg.append(f'  fn={fn:0{1 << n}b} cost={cost} nctrl={k} ×{cnt}')
            msg.append(f'拆分后独有段({sum(only_after.values())} 条, Σcost={s_after}):')
            for (fn, cost, k), cnt in only_after.items():
                msg.append(f'  fn={fn:0{1 << n}b} cost={cost} nctrl={k} ×{cnt}')
            print('\n'.join(msg), flush=True)
            out += msg + ['']
            found += 1
            if found >= 3:
                break
        if found >= 3:
            break
    print(f'\n耗时 {time.perf_counter()-t0:.1f}s, 找到 {found} 例')
    with open(OUT, 'w') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
