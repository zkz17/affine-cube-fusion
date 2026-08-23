# -*- coding: utf-8 -*-
"""
P7 探针:beam search 兑现「结构承诺」的引擎级收益(仅 n ≤ 7)。

动机(P3/P4 + P6a 解释修正):BDD 小 n 优势来自 split 模式的结构承诺(构图期
δ=2 重写);引擎侧替代 = 对匹配结构做承诺的 beam search——每轮保留 width 个
最优池状态(单步移动分支),而非贪心的极大匹配一步到位。

对照口径:
  池 A = BDD 图阶段产物(_bdd_entries,含根掩码);
  池 F = FPRM 最优平移的 ANF 池(融合前)。
  变体:greedy(生产 fuse_pool_global)、beam1/beam4/beam16(单步移动 beam);
  池 F 另附 GL 搜索成本(衡量 beam 相对整坐标重分解的回收率)。

实现为探针脚本,不动生产代码。状态去重键 = (ΣT, 池内 (cb, r) 集合)。
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from limbdd.CompleteBranch import CompleteBranch, plan_from_affine
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from limbdd.fusion.ClosedForm import canonical_affine, fuse_closed
from limbdd.fusion.GLSearch import fprm_best, gl_search

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'beam_search_probe.txt')
PROFILE = {5: (30, (1, 4, 16)), 6: (20, (1, 4, 16)), 7: (8, (1, 4))}


def successors(state, n):
    """一个池状态的单步移动后继:全部抵消/正收益融合各一个后继。"""
    cost, pool = state
    aff = [canonical_affine(cb, r) for cb, r, _ in pool]
    out = []
    seen = set()
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            res = fuse_closed(aff[i], aff[j], n)
            if not res.ok:
                continue
            if res.cancel:
                new_pool = [e for k, e in enumerate(pool) if k not in (i, j)]
                new_cost = cost - (pool[i][0].t_cost() + pool[j][0].t_cost())
            else:
                fused = CompleteBranch(n, synthetic=plan_from_affine(res.base, res.basis, n),
                                       origin=pool[i][0].origin)
                # 只接受抵消或非负收益(结构承诺口径:允许中性移动)
                if pool[i][0].t_cost() + pool[j][0].t_cost() < fused.t_cost():
                    continue
                new_pool = [e for k, e in enumerate(pool) if k not in (i, j)] + [(fused, 0, None)]
                new_cost = cost - (pool[i][0].t_cost() + pool[j][0].t_cost()
                                   - fused.t_cost())
                assert new_cost <= cost
            key = frozenset((cb, r) for cb, r, _ in new_pool)
            if key in seen:
                continue
            seen.add(key)
            out.append((new_cost, new_pool))
    return out


def beam_fuse(entries, n, width, max_rounds=None):
    """单步移动 beam search;返回 (best_cost, best_pool)。"""
    cost0 = sum(cb.t_cost() for cb, _, _ in entries)
    states = [(cost0, list(entries))]
    m = len(entries)
    if max_rounds is None:
        max_rounds = m + 2
    for _ in range(max_rounds):
        nxt = {}
        for st in states:
            for s in successors(st, n):
                key = (s[0], frozenset((cb, r) for cb, r, _ in s[1]))
                if key not in nxt or s[0] < nxt[key][0]:
                    nxt[key] = s
        if not nxt:
            break
        states = sorted(nxt.values(), key=lambda s: s[0])[:width]
    return min(states, key=lambda s: s[0])


def run_case(fn, n, widths):
    """对 BDD 池与 FPRM 池各跑 greedy/beam 变体,返回统计字典。"""
    from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    tt = TruthTable(fn, n)
    bdd = synth._bdd_entries(tt, None)
    vF, cF = fprm_best(fn, n)
    frm = anf_pool(shift_fn(fn, cF, n), n)

    out = Counter()
    for label, entries in (('bdd', bdd), ('frm', frm)):
        t0 = time.perf_counter()
        g = pool_t_cost(*fuse_pool_global(entries, n))
        out[f'{label}_greedy'] = g
        out[f'{label}_greedy_wall'] = time.perf_counter() - t0
        for w in widths:
            t0 = time.perf_counter()
            cost, _ = beam_fuse(entries, n, w)
            out[f'{label}_beam{w}'] = cost
            out[f'{label}_beam{w}_wall'] = time.perf_counter() - t0
            out[f'{label}_beam{w}_better'] = 1 if cost < g else 0
    t0 = time.perf_counter()
    kept_g, st_g, _, _ = gl_search(fn, n)
    out['frm_gl'] = pool_t_cost(kept_g, st_g)
    out['frm_gl_wall'] = time.perf_counter() - t0
    return out


def main():
    lines = ['P7 探针:beam search(引擎级结构承诺)vs 贪心融合', '']
    for n, (cases, widths) in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = Counter()
        for fn in data:
            r = run_case(fn, n, widths)
            for k, v in r.items():
                acc[k] += v
        msg = [f'====== n = {n}({cases} cases)======']
        for label in ('bdd', 'frm'):
            g = acc[f'{label}_greedy']
            msg.append(f'{label} 池: greedy ΣT={g} ({acc[f"{label}_greedy_wall"]:.1f}s)')
            for w in widths:
                b = acc[f'{label}_beam{w}']
                msg.append(f'  beam{w}: ΣT={b} ({(b/g-1):+.2%}, 更优 {acc[f"{label}_beam{w}_better"]}/{cases}, '
                           f'{acc[f"{label}_beam{w}_wall"]:.1f}s)')
        gl = acc['frm_gl']
        msg.append(f'frm 池对照: GL ΣT={gl} ({(gl/acc["frm_greedy"]-1):+.2%}, {acc["frm_gl_wall"]:.1f}s)')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
