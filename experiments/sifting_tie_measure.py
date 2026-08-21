"""
sifting 结点数平局占比量测（支撑 A1 方向：平局时用融合 ΣT 决胜的收益上限）。

复刻 reorder_proxy_measure 的决策循环（无融合评分），统计每个决策段（第二段）
pass2 中：
  - 结点数并列候选（node_num == min node）占比；
  - 结点数 + 分支数双并列候选（现行平局规则下真正可能改变选择的候选）占比。

用法: ./venv/bin/python experiments/sifting_tie_measure.py [n ...]   默认 8 9 10
"""
import os
import random
import sys

# 项目根目录（脚本位于 experiments/ 下，故上溯一级）；加入 sys.path 保证 limbdd/utils 可导入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from utils.TruthTable import TruthTable

CASES = {8: 5, 9: 5, 10: 3}


def swap_pass(mgr, limbdd, current_level, target_level, records):
    """复刻 swap_to_level 的移动与 min 跟踪（分支数平局），records 记录决策段位置。"""
    current_node_num, min_node_num, min_level = 0, 0, current_level
    min_branch_count = limbdd.branch_count(limbdd.root.node, cache={})

    def observe(level):
        nonlocal min_node_num, min_level, min_branch_count
        bc = limbdd.branch_count(limbdd.root.node, cache={})
        records.append((level, current_node_num, bc))
        if current_node_num < min_node_num:
            min_node_num, min_level, min_branch_count = current_node_num, level, bc
        elif current_node_num == min_node_num and bc <= min_branch_count:
            min_node_num, min_level, min_branch_count = current_node_num, level, bc

    while current_level < target_level:
        ok, delta = limbdd.swap_neighbor_levels(current_level, current_level + 1)
        if not ok:
            break
        current_level += 1
        current_node_num += delta
        observe(current_level)

    while current_level > target_level:
        ok, delta = limbdd.swap_neighbor_levels(current_level - 1, current_level)
        if not ok:
            break
        current_level -= 1
        current_node_num += delta
        observe(current_level)

    return current_level, min_level


def measure_case(tt, n):
    syn = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    syn.construct(tt)
    mgr = syn.limbdd_mgr
    mgr.try_neg(mgr.limbdd)
    mgr.split_neg(mgr.limbdd)

    limbdd = mgr.limbdd
    if limbdd.root.level <= 2 or len(limbdd.node_table) <= limbdd.root.level + 1:
        return None

    top_level = limbdd.root.level
    while top_level > 1 and len(limbdd.level2nodes[top_level - 1]) == 1:
        top_level -= 1
    if all(len(limbdd.level2nodes[l]) == 1 for l in range(1, top_level + 1)):
        return None

    mid_level = top_level >> 1
    cands = sorted(range(1, top_level + 1), key=lambda l: len(limbdd.level2nodes[l]))
    stats = []  # (决策段位置数, 结点数并列数, 结点+分支双并列数)
    while cands:
        start = cands.pop()
        recs = []
        if start <= mid_level:
            cur, _ = swap_pass(mgr, limbdd, start, 1, recs)
            p2 = len(recs)
            cur, min_level = swap_pass(mgr, limbdd, cur, top_level, recs)
        else:
            cur, _ = swap_pass(mgr, limbdd, start, top_level, recs)
            p2 = len(recs)
            cur, min_level = swap_pass(mgr, limbdd, cur, 1, recs)
        pass2 = recs[p2:]
        mn = min(r[1] for r in pass2) if pass2 else 0
        mn_bc = min(r[2] for r in pass2 if r[1] == mn) if pass2 else 0
        if pass2:
            node_ties = sum(1 for r in pass2 if r[1] == mn)
            both_ties = sum(1 for r in pass2 if r[1] == mn and r[2] == mn_bc)
            stats.append((len(pass2), node_ties, both_ties))
        # 第三段回位（维持管线状态，与 proxy 相同）
        final, _ = swap_pass(mgr, limbdd, cur, min_level, recs)
        if final > start:
            for i in range(len(cands)):
                if cands[i] <= final and cands[i] > start:
                    cands[i] -= 1
        elif final < start:
            for i in range(len(cands)):
                if cands[i] >= final and cands[i] < start:
                    cands[i] += 1
        mgr.split_neg(limbdd)

    return stats


def main():
    ns = [int(a) for a in sys.argv[1:]] or [8, 9, 10]
    for n in ns:
        tot_pos = tot_node = tot_both = 0
        for s in range(CASES[n]):
            rng = random.Random(10 ** 9 + n * 1000 + s)
            tt = TruthTable(rng.getrandbits(1 << n), n)
            stats = measure_case(tt, n)
            if stats is None:
                print(f'n={n} case {s}: skip', flush=True)
                continue
            npos = sum(x[0] for x in stats)
            nnode = sum(x[1] for x in stats)
            nboth = sum(x[2] for x in stats)
            tot_pos += npos
            tot_node += nnode
            tot_both += nboth
            print(f'n={n} case {s}: 决策点 {npos}, 结点并列 {nnode} ({nnode / max(1, npos):.0%}), '
                  f'结点+分支双并列 {nboth} ({nboth / max(1, npos):.0%})', flush=True)
        print(f'n={n} SUM: {tot_pos} 决策点, 结点并列 {tot_node / max(1, tot_pos):.1%}, '
              f'双并列 {tot_both / max(1, tot_pos):.1%}\n', flush=True)


if __name__ == '__main__':
    main()
