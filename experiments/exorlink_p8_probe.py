# -*- coding: utf-8 -*-
"""
P8 探针:统一交织贪心——δ=1 融合 + δ=2 立即正收益 + 宏 M2(带归还者的承诺重写)。

宏 M2(用户提议,理论见 Optimization_roadmap.md 或本文件 doc):
  对 δ=2 的非可融/非正收益对 (A,B),取候选 h(A⊕h、B⊕h 均为立方,重写 T-中性),
  若存在第三段 C 使 gain(A′, C) > 0(A′ = A⊕h),则整体宏
  {A,B,C} → {B′, A′⊕C} 的 ΣT 严格下降(宏收益 = gain(A′, C)),且**当场归还**:
  不引入未承诺的中性状态(对比 P7 beam 的漂移缺陷)。B 侧对称。

变体(同一单步贪心骨架,仅移动类不同):
  V1 = 仅 δ=1 融合;
  V2 = V1 + δ=2 立即正收益重写;
  V3 = V2 + 宏 M2(完整)。
对照参考:生产 fuse_pool_global(极大匹配贪心)。

四个池(每用例):ANF 原始 / BDD 原始(split 后)/ **BDD 去 split** / FPRM 最优。
关键指标:BDD 去 split 池上 V3 相对 V1 的回收率
  recovery = [V1(ds) − V3(ds)] / [V1(ds) − V1(so)]
(split 增益 = V1(ds) − V1(so);≥30% 为进入生产化讨论的阈值)。

隔离声明:只读 data/test_cases,只写 experiments/phase0_results/;
不修改生产代码,不影响 data/our_cost/ 下运行中的其它实验。
"""
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import config
from utils.TruthTable import TruthTable
from limbdd.CompleteBranch import CompleteBranch, plan_from_affine
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from limbdd.fusion.ClosedForm import canonical_affine, fuse_closed
from limbdd.fusion.GLSearch import fprm_best
import experiments.exorlink_p6a_probe as P6

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'exorlink_p8_probe.txt')
TC = config['t_cost']
PROFILE = {5: 20, 6: 12, 7: 4}


def seg_from(res, root, n):
    """闭式结果 → 分支段(保持根掩码 root 的平移口径)。"""
    return CompleteBranch(n, synthetic=plan_from_affine(res.base ^ root, res.basis, n))


def pool_xor(pool, n):
    """池 XOR 不变量(有效函数空间:kept 段按根掩码平移)。"""
    x = 0
    for cb, r, _ in pool:
        x ^= shift_fn(cb.fn(), r, n)
    return x


def unified_engine(entries, n, allow_d2, allow_m2, max_rounds=None):
    """
    单步全局最大下降贪心。返回 (cost, stats)。
    移动类:
      d1 : 抵消 / 正收益融合;
      d2 : δ=2 双侧重写(立即正收益);
      m2 : 宏 M2(∃C: gain(A′,C) > 0,宏收益 = 该融合收益)。
    全程校验池 XOR 不变量(每轮)。
    """
    pool = [(cb, r, key) for cb, r, key in entries]
    expected_xor = pool_xor(pool, n)
    cost = sum(cb.t_cost() for cb, _, _ in pool)
    stats = Counter()
    if max_rounds is None:
        max_rounds = len(pool) + 2

    for _ in range(max_rounds):
        aff = [canonical_affine(cb, r) for cb, r, _ in pool]
        m = len(pool)
        best = None                       # (gain, kind, payload)

        # ---- d1 ----
        for i in range(m):
            ti = pool[i][0].t_cost()
            for j in range(i + 1, m):
                res = fuse_closed(aff[i], aff[j], n)
                if not res.ok:
                    continue
                gain = ti + pool[j][0].t_cost() if res.cancel else \
                    ti + pool[j][0].t_cost() - TC(n - res.dim)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, 'd1', (i, j, res))

        # ---- d2 / m2(只扫 δ=2 对)----
        if allow_d2 or allow_m2:
            rows = [P6.rows_of(pool[i][0], pool[i][1], n) for i in range(m)]
            for i in range(m):
                ti = pool[i][0].t_cost()
                for j in range(i + 1, m):
                    d, agree, opp, oa, ob = P6.pair_info(rows[i], rows[j], n)
                    if d != 2:
                        continue
                    for rows_h in P6.h_candidates(agree, opp, oa, ob):
                        ch = P6.cb_from_rows(rows_h, n)
                        if ch is None:
                            continue
                        ah = canonical_affine(ch)
                        r1 = fuse_closed(aff[i], ah, n)
                        r2 = fuse_closed(aff[j], ah, n)
                        if not (r1.ok and r2.ok and not r1.cancel and not r2.cancel):
                            continue
                        imm = ti + pool[j][0].t_cost() - TC(n - r1.dim) - TC(n - r2.dim)
                        if allow_d2 and imm > 0 and (best is None or imm > best[0]):
                            best = (imm, 'd2', (i, j, r1, r2))
                        if allow_m2:
                            for side, rA, rB, idxA, idxB in (('A', r1, r2, i, j),
                                                             ('B', r2, r1, j, i)):
                                tA = TC(n - rA.dim)
                                affA = (rA.base, rA.basis)
                                for k in range(m):
                                    if k in (i, j):
                                        continue
                                    rc = fuse_closed(affA, aff[k], n)
                                    if not rc.ok or rc.cancel:
                                        continue
                                    g = tA + pool[k][0].t_cost() - TC(n - rc.dim)
                                    if g > 0 and (best is None or g > best[0]):
                                        best = (g, 'm2',
                                                (idxA, idxB, k, rA, rB, rc, side))

        if best is None:
            break
        gain, kind, pay = best
        if kind == 'd1':
            i, j, res = pay
            if res.cancel:
                pool = [e for q, e in enumerate(pool) if q not in (i, j)]
                cost -= gain
            else:
                fused = seg_from(res, 0, n)
                pool = [e for q, e in enumerate(pool) if q not in (i, j)] + [(fused, 0, None)]
                cost -= gain
        elif kind == 'd2':
            i, j, r1, r2 = pay
            a1 = seg_from(r1, pool[i][1], n)
            b1 = seg_from(r2, pool[j][1], n)
            pool = [e for q, e in enumerate(pool) if q not in (i, j)] + \
                   [(a1, pool[i][1], i), (b1, pool[j][1], j)]
            cost -= gain
        else:  # m2
            idxA, idxB, k, rA, rB, rc, side = pay
            b1 = seg_from(rB, pool[idxB][1], n)
            fused = seg_from(rc, 0, n)
            pool = [e for q, e in enumerate(pool) if q not in (idxA, idxB, k)] + \
                   [(b1, pool[idxB][1], idxB), (fused, 0, None)]
            cost -= gain
        stats[kind] += 1
        stats[f'{kind}_gain'] += gain
        assert pool_xor(pool, n) == expected_xor, f'{kind} 移动破坏池 XOR 不变量'
    return cost, stats


def bdd_pool(tt, patterns):
    from limbdd.LimBDDManager import LimBDDManager
    from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
    orig = LimBDDManager.split_with_pattern
    if not patterns:
        LimBDDManager.split_with_pattern = lambda self, limbdd, pattern: None
    try:
        synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
        return synth._bdd_entries(tt, None)
    finally:
        LimBDDManager.split_with_pattern = orig


def run_case(fn, n, acc):
    tt = TruthTable(fn, n)
    vF, cF = fprm_best(fn, n)
    pools = {
        'anf': anf_pool(fn, n),
        'bdd': bdd_pool(tt, True),
        'bdd_ds': bdd_pool(tt, False),
        'frm': anf_pool(shift_fn(fn, cF, n), n),
    }
    for pname, entries in pools.items():
        ref = pool_t_cost(*fuse_pool_global(entries, n))
        acc[f'{pname}_ref'] += ref
        for vname, (d2, m2) in (('V1', (False, False)), ('V2', (True, False)),
                                ('V3', (True, True))):
            t0 = time.perf_counter()
            cost, st = unified_engine(entries, n, d2, m2)
            acc[f'{pname}_{vname}_wall'] += time.perf_counter() - t0
            acc[f'{pname}_{vname}'] += cost
            for k, v in st.items():
                acc[f'{pname}_{vname}_{k}'] += v
            if vname == 'V1':
                base = cost
            else:
                acc[f'{pname}_{vname}_better'] += cost < base
    acc['cases'] += 1


def main():
    lines = ['P8 探针:统一交织贪心(δ=1 / δ=2 / 宏 M2)', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = Counter()
        for fn in data:
            run_case(fn, n, acc)
        c = acc['cases']
        msg = [f'====== n = {n}({cases} cases)======']
        for pname in ('anf', 'bdd', 'bdd_ds', 'frm'):
            v1, v2, v3 = acc[f'{pname}_V1'], acc[f'{pname}_V2'], acc[f'{pname}_V3']
            msg.append(f'{pname:>7} 池: ref(生产)={acc[f"{pname}_ref"]}  '
                       f'V1={v1}  V2={v2}  V3={v3}')
            msg.append(f'          相对 V1: V2 {(v2/v1-1):+.2%}(更优 {acc[f"{pname}_V2_better"]}/{c})  '
                       f'V3 {(v3/v1-1):+.2%}(更优 {acc[f"{pname}_V3_better"]}/{c})')
            if acc[f'{pname}_V3_m2']:
                msg.append(f'          M2 接受 {acc[f"{pname}_V3_m2"]} 次(均增益 '
                           f'{acc[f"{pname}_V3_m2_gain"]/max(acc[f"{pname}_V3_m2"],1):.1f}),'
                           f' d2 接受 {acc[f"{pname}_V3_d2"]} 次')
        # 回收率:BDD 去 split 池上 V3 vs V1,分母 = split 增益
        v1so, v1ds, v3ds = acc['bdd_V1'], acc['bdd_ds_V1'], acc['bdd_ds_V3']
        denom = v1ds - v1so
        rec = (v1ds - v3ds) / denom if denom > 0 else float('nan')
        msg.append(f'split 增益(V1)={denom},V3 在去 split 池上的额外收益={v1ds - v3ds},'
                   f'回收率={rec:.1%}')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
