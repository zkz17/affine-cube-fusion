# -*- coding: utf-8 -*-
"""
P6a 探针:exorlink-δ 的机会密度与 δ=2 符号预判的批量验证(n ≤ 7)。

三段快照,逐池统计所有段对的经典距离 δ:
  ANF 池     : anf_pool(原始 f)                      —— 全为普通立方;
  BDD 池     : 图阶段产物(_bdd_entries,含根掩码)     —— 一般仿射段;
  不动点池   : FPRM 最优平移的融合不动点             —— 走线/合成混合。
一般仿射段用 canonical_affine 的 RREF 约束行 (mask, const) 作「字面量」:
同 mask 同 const = 一致;同 mask 异 const = 极性相反(分歧);单侧 = 分歧。

指标:
  δ 分布直方图;δ=2 对中「存在正收益候选」的比例;δ≥3 对的正收益候选命中率;
  修正版符号预判(正收益 ⇔ 存在极性相反变量,收益 = t(k)−t(k−1))的批量验证
  (实际增益 vs 公式,统计不符数);
  不动点池的一次性改进:全局最佳正收益 δ=2/≥3 重写 + 全池重融的 ΣT 增量。
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import config
from limbdd.CompleteBranch import CompleteBranch, plan_from_affine
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from limbdd.fusion.ClosedForm import canonical_affine, fuse_closed
from limbdd.fusion.GLSearch import fprm_best

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'exorlink_p6a_probe.txt')
PROFILE = {5: 60, 6: 40, 7: 20}
TC = config['t_cost']


def rows_of(cb, r, n):
    """一般仿射段的规范约束行:((mask, const), ...),含根掩码平移(base ^= r)。

    注意:affine_rows() 的行掩码去掉了主元位(主元隐含),须加回 (1 << pivot)。
    """
    base, basis = canonical_affine(cb, r)
    plan = plan_from_affine(base, basis, n)
    tmp = CompleteBranch(n, synthetic=plan)
    return tuple(((1 << col) | mask, const) for col, mask, const in tmp.affine_rows())


def pair_info(ra, rb, n):
    """
    两段约束行集合的经典距离 δ 与候选 h(全部单行分歧候选)。
    返回 (δ, agree_rows, opp_indices, oneside_indices)。
    行集合按 (mask, const) 规范化比较。
    """
    da = dict((m, c) for m, c in ra)
    db = dict((m, c) for m, c in rb)
    agree = []
    opp = []
    oneside_a = []
    oneside_b = []
    for m in da:
        if m in db:
            if da[m] == db[m]:
                agree.append((m, da[m]))
            else:
                opp.append(m)
        else:
            oneside_a.append((m, da[m]))
    for m in db:
        if m not in da:
            oneside_b.append((m, db[m]))
    return len(opp) + len(oneside_a) + len(oneside_b), agree, opp, oneside_a, oneside_b


def h_candidates(agree, opp, oneside_a, oneside_b):
    """候选 h 的约束行列表:每个分歧变量一行(opp 取两侧各一种)。"""
    out = []
    for m in opp:
        out.append(agree + [(m, 0)])
        out.append(agree + [(m, 1)])
    for m, c in oneside_a:
        out.append(agree + [(m, c)])
    for m, c in oneside_b:
        out.append(agree + [(m, c)])
    return out


def cb_from_rows(rows, n):
    """约束行集合 → CompleteBranch(合成表示)。行须线性无关。"""
    pivots = {}
    for m, c in rows:
        mm, cc = m, c
        while mm:
            low = mm & (-mm)
            if low in pivots:
                mm ^= pivots[low][0]
                cc ^= pivots[low][1]
            else:
                pivots[low] = (mm, cc)
                break
        if mm == 0 and cc:
            return None          # 不可满足,非立方
    # 构造 (base, basis)
    basis = []
    base = 0
    pivot_cols = {v.bit_length() - 1 for v in pivots}
    for f in range(n):
        if f in pivot_cols:
            continue
        v = 1 << f
        for m, c in pivots.values():
            if (m >> f) & 1:
                v |= 1 << (m & (-m)).bit_length() - 1
        basis.append(v)
    for m, c in pivots.values():
        if c:
            base |= 1 << ((m & (-m)).bit_length() - 1)
    return CompleteBranch(n, synthetic=plan_from_affine(base, basis, n))


def move_gain(cb_a, cb_b, rows_h, n):
    """移动 {A,B} → {A⊕h, B⊕h} 的实际收益;不可融返回 (None, None)。

    返回 (gain, (kind1, kind2)) 供结构断言:正收益 ⇔ 一侧 kind1(C2)
    且另一侧 kind3/4(C3/C4 嵌套,代价不变)。
    """
    ch = cb_from_rows(rows_h, n)
    if ch is None:
        return None, None
    r1 = fuse_closed(canonical_affine(cb_a), canonical_affine(ch), n)
    r2 = fuse_closed(canonical_affine(cb_b), canonical_affine(ch), n)
    if not r1.ok or not r2.ok or r1.cancel or r2.cancel:
        return None, None
    g = TC(cb_a.nctrl()) + TC(cb_b.nctrl()) - TC(n - r1.dim) - TC(n - r2.dim)
    return g, (r1.kind, r2.kind)


def analyze_pool(pool, n, label, acc):
    """pool: [(cb, r)]。统计 δ 分布、δ=2/≥3 正收益命中、符号公式验证。"""
    m = len(pool)
    hist = Counter()
    pos2 = pos3 = tot2 = tot3 = 0
    formula_bad = 0
    best_move = None
    for i in range(m):
        ri = rows_of(pool[i][0], pool[i][1], n)
        for j in range(i + 1, m):
            rj = rows_of(pool[j][0], pool[j][1], n)
            d, agree, opp, oneside_a, oneside_b = pair_info(ri, rj, n)
            hist[min(d, 5)] += 1
            if d < 2:
                continue
            cands = h_candidates(agree, opp, oneside_a, oneside_b)
            gains = []                       # [(gain, kinds)]
            for rows_h in cands:
                g, kinds = move_gain(pool[i][0], pool[j][0], rows_h, n)
                if g is not None:
                    gains.append((g, kinds))
            pos = max((g for g, _ in gains), default=0)
            has_pos = pos > 0
            if d == 2:
                tot2 += 1
                pos2 += has_pos
                if has_pos:
                    gi = next(k for k, (g, _) in enumerate(gains) if g == pos)
                    if best_move is None or pos > best_move[0]:
                        best_move = (pos, i, j, cands[gi])
            else:
                tot3 += 1
                pos3 += has_pos
                if has_pos:
                    gi = next(k for k, (g, _) in enumerate(gains) if g == pos)
                    if best_move is None or pos > best_move[0]:
                        best_move = (pos, i, j, cands[gi])
            # 符号预判验证(结构版):对每个候选,正收益 ⇔ 一侧 kind1(C2)且另一侧
            # kind3/4(C3/C4 嵌套,代价不变)。
            for g, kinds in gains:
                is_c2 = 1 in kinds
                is_nest = any(k in (3, 4) for k in kinds)
                if (g > 0) != (is_c2 and is_nest):
                    formula_bad += 1
    acc['hist'] += hist
    acc['pairs'] += m * (m - 1) // 2
    acc['tot2'] += tot2
    acc['pos2'] += pos2
    acc['tot3'] += tot3
    acc['pos3'] += pos3
    acc['formula_bad'] += formula_bad
    return best_move


def apply_move(pool, best_move, n):
    """执行一次最佳重写并全池重融;返回 (新不动点成本, 原成本)。"""
    pos, i, j, rows_h = best_move
    ch = cb_from_rows(rows_h, n)
    r1 = fuse_closed(canonical_affine(pool[i][0], pool[i][1]), canonical_affine(ch), n)
    r2 = fuse_closed(canonical_affine(pool[j][0], pool[j][1]), canonical_affine(ch), n)
    p1 = CompleteBranch(n, synthetic=plan_from_affine(r1.base, r1.basis, n))
    p2 = CompleteBranch(n, synthetic=plan_from_affine(r2.base, r2.basis, n))
    new_pool = [(cb, r, key) for key, (cb, r) in enumerate(pool)]
    new_pool[i] = (p1, pool[i][1], i)
    new_pool[j] = (p2, pool[j][1], j)
    kept, st = fuse_pool_global(new_pool, n)
    return pool_t_cost(kept, st)


def main():
    lines = ['P6a 探针:exorlink-δ 机会密度与 δ=2 符号预判批量验证', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = {k: 0 for k in ('hist', 'pairs', 'tot2', 'pos2', 'tot3', 'pos3', 'formula_bad')}
        acc['hist'] = Counter()
        one_step = Counter()
        t0 = time.perf_counter()
        for fn in data:
            # ANF 池
            pool_anf = [(cb, 0) for cb, _, _ in anf_pool(fn, n)]
            analyze_pool(pool_anf, n, 'anf', acc)
            # BDD 池
            from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
            synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
            pool_bdd = [(cb, r) for cb, r, _ in synth._bdd_entries(__import__('utils.TruthTable', fromlist=['TruthTable']).TruthTable(fn, n), None)]
            analyze_pool(pool_bdd, n, 'bdd', acc)
            # 不动点池
            vF, cF = fprm_best(fn, n)
            kept, st = fuse_pool_global(anf_pool(shift_fn(fn, cF, n), n), n)
            pool_fp = [(cb, r) for cb, r, _ in kept] + [(cb, 0) for cb in st]
            best_move = analyze_pool(pool_fp, n, 'fp', acc)
            one_step['cases'] += 1
            if best_move is not None:
                c0 = pool_t_cost(kept, st)
                c1 = apply_move(pool_fp, best_move, n)
                one_step['gain_sum'] += c0 - c1
                one_step['gain_max'] = max(one_step['gain_max'], c0 - c1)
                one_step['nonzero'] += c1 < c0
        wall = time.perf_counter() - t0
        h = acc['hist']
        msg = [
            f'====== n = {n}({cases} cases,{wall:.1f}s)======',
            f'段对总数 {acc["pairs"]};δ 分布: ' +
            ', '.join(f'δ={k}:{h[k]}({h[k]/max(acc["pairs"],1):.0%})' for k in sorted(h)),
            f'δ=2: 共 {acc["tot2"]} 对,正收益 {acc["pos2"]} ({acc["pos2"]/max(acc["tot2"],1):.1%})',
            f'δ≥3: 共 {acc["tot3"]} 对,正收益 {acc["pos3"]} ({acc["pos3"]/max(acc["tot3"],1):.2%})',
            f'符号预判不符数: {acc["formula_bad"]}',
            f'不动点一次性改进: 有效 {one_step["nonzero"]}/{one_step["cases"]},'
            f' Σ 增益 {one_step["gain_sum"]}, 最大 {one_step["gain_max"]}',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
