# -*- coding: utf-8 -*-
"""
P9 探针:宏 M2 的生产式相间调度验证(五项路径分别量测边际)。

调度(生产式):pool → fuse_pool_global(极大匹配,到不动点)→ 枚举最佳宏 M2 →
应用 → 重融,循环;上限 interleave 轮数。对照 V0 = 纯生产融合(无 M2)。

五项路径的**融合前**池:
  A  = BDD 图阶段产物(split 开,absorb 已移除);
  C  = anf_pool(f)(c=0);
  F  = anf_pool(f∘τ_cF)(FPRM 最优平移);
  FP = anf_pool(f∘τ_cFp)(fprm_proxy 最优平移);
  G  = gl_search 结果池(kept+standalone,池本体为 g=f(Ax⊕c) 的分解)。

验证:
  - 每轮融合/宏应用后断言池 XOR 不变量;
  - A/C/F/FP 路径的最终池按生产发射管线(平移三明治 + 根三明治 + B-4 重排 +
    simplify)发射并 compute_tt == f;G 路径的线性 wrapper 属发射层,
    只验证池 XOR == g(发射层不参与 M2,故 G 只做池级边际)。
隔离:只读 data/test_cases,只写 experiments/phase0_results/。
"""
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import config
from utils.Qubit import QubitManager
from utils.TruthTable import TruthTable
from limbdd.CompleteBranch import CompleteBranch, _bits, plan_from_affine
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from limbdd.fusion.ClosedForm import canonical_affine, fuse_closed
from limbdd.fusion.FprmProxy import fprm_proxy_best
from limbdd.fusion.GLSearch import (apply_linear, compose_transvection_right,
                                    fprm_best, gl_search)
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from circuit.MCXCircuit import MCXCircuit, order_segments_greedy
import experiments.exorlink_p6a_probe as P6

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'exorlink_p9_probe.txt')
TC = config['t_cost']
PROFILE = {5: 25, 6: 15, 7: 6}
MAX_INTERLEAVE = 30


def seg_from(res, root, n):
    return CompleteBranch(n, synthetic=plan_from_affine(res.base ^ root, res.basis, n))


def pool_xor(pool, n):
    x = 0
    for cb, r, _ in pool:
        x ^= shift_fn(cb.fn(), r, n)
    return x


def best_m2(pool, n):
    """
    不动点处的最佳宏 M2(真实总收益 = 对子移动 imm + A′+C 融合收益,严格 > 0)。

    返回 (total, imm, g_fuse, idxA, idxB, k, rA, rB, rc, side) 或 None。
    注:不动点处 imm ≤ 0(否则 d2 移动早已被采用);Π 中性对即 imm == 0,
    imm < 0 属「有界恶化 + 融合补偿」推广(总收益仍严格为正)。
    """
    aff = [canonical_affine(cb, r) for cb, r, _ in pool]
    rows = [P6.rows_of(pool[i][0], pool[i][1], n) for i in range(len(pool))]
    m = len(pool)
    best = None
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
                for side, (rA, rB, idxA, idxB) in (('A', (r1, r2, i, j)),
                                                   ('B', (r2, r1, j, i))):
                    tA = TC(n - rA.dim)
                    affA = (rA.base, rA.basis)
                    for k in range(m):
                        if k in (i, j):
                            continue
                        rc = fuse_closed(affA, aff[k], n)
                        if not rc.ok or rc.cancel:
                            continue
                        g_fuse = tA + pool[k][0].t_cost() - TC(n - rc.dim)
                        total = imm + g_fuse
                        if total > 0 and (best is None or total > best[0]):
                            best = (total, imm, g_fuse, idxA, idxB, k, rA, rB, rc, side)
    return best


def interleaved(entries, n, expected_xor):
    """生产式相间调度:融合到不动点 → 最佳 M2 → 重融,循环。返回 (cost, pool, stats)。"""
    pool = [(cb, r, key) for cb, r, key in entries]
    stats = Counter()
    rounds = 0
    while rounds < MAX_INTERLEAVE:
        kept, st = fuse_pool_global(pool, n)
        pool = [(cb, r, key) for cb, r, key in kept] + [(cb, 0, None) for cb in st]
        assert pool_xor(pool, n) == expected_xor, '融合后 XOR 不变量破坏'
        mv = best_m2(pool, n)
        if mv is None:
            break
        total, imm, g_fuse, idxA, idxB, k, rA, rB, rc, side = mv
        b1 = seg_from(rB, pool[idxB][1], n)
        fused = seg_from(rc, 0, n)
        pool = [e for q, e in enumerate(pool) if q not in (idxA, idxB, k)] + \
               [(b1, pool[idxB][1], pool[idxB][2]), (fused, 0, None)]
        assert pool_xor(pool, n) == expected_xor, 'M2 后 XOR 不变量破坏'
        stats['m2'] += 1
        stats['m2_gain'] += total
        stats['m2_neutral'] += imm == 0
        rounds += 1
    return sum(cb.t_cost() for cb, _, _ in pool), pool, stats


def emit_and_verify(pool, fn, n, shift_c=0):
    """按生产发射管线(平移三明治 + 根三明治 + B-4 重排 + simplify)发射并验证。"""
    qm = QubitManager()
    for v in range(n):
        qm.get_input_qubit(v)
    circ = MCXCircuit(qm)
    circ.add_lim_operator([qm.get_input_qubit(v) for v in _bits(shift_c)])
    kept_by_r = {}
    for cb, r, key in pool:
        if key is not None:
            kept_by_r.setdefault(r, []).append(cb)
    for r, segs in sorted(kept_by_r.items()):
        rq = [qm.get_input_qubit(v) for v in _bits(r)]
        circ.add_lim_operator(rq)
        for sc in order_segments_greedy([cb.to_circuit(qm) for cb in segs],
                                        circ if not circ.empty() else None):
            circ.merge(sc)
        circ.add_lim_operator(rq)
    stand = [cb for cb, r, key in pool if key is None]
    for sc in order_segments_greedy([cb.to_circuit(qm) for cb in stand],
                                    circ if not circ.empty() else None):
        circ.merge(sc)
    circ.add_lim_operator([qm.get_input_qubit(v) for v in _bits(shift_c)])
    circ.simplify_terms()
    assert circ.compute_tt() == fn, '发射级 compute_tt 验证失败'
    return circ


def g_of_cnots(fn, shift_c, cnots, n):
    """由 gl_search 的 cnots(生成顺序)重建 A,返回 g = f(Ax ⊕ c)。"""
    A = [1 << i for i in range(n)]
    for c, t in cnots:
        A = compose_transvection_right(A, c, t, n)
    return apply_linear(shift_fn(fn, shift_c, n) if shift_c else fn, A, n)


def path_pools(fn, n, synth, tt):
    _, cF = fprm_best(fn, n)
    _, _, cFp = fprm_proxy_best(fn, n)
    kept_g, st_g, c_g, cnots_g = gl_search(fn, n)
    g_pool = [(cb, r, key) for cb, r, key in kept_g] + [(cb, 0, None) for cb in st_g]
    return {
        'A': (synth._bdd_entries(tt, None), fn, 0),
        'C': (anf_pool(fn, n), fn, 0),
        'F': (anf_pool(shift_fn(fn, cF, n), n), shift_fn(fn, cF, n), cF),
        'FP': (anf_pool(shift_fn(fn, cFp, n), n), shift_fn(fn, cFp, n), cFp),
        'G': (g_pool, g_of_cnots(fn, c_g, cnots_g, n), None),
    }


def run_case(fn, n, acc):
    tt = TruthTable(fn, n)
    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    pools = path_pools(fn, n, synth, tt)
    for pname, (entries, expect, shift_c) in pools.items():
        t0 = time.perf_counter()
        v0 = pool_t_cost(*fuse_pool_global(entries, n))
        acc[f'{pname}_V0_wall'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        v1, pool1, st = interleaved(entries, n, expect)
        acc[f'{pname}_V1_wall'] += time.perf_counter() - t0
        acc[f'{pname}_V0'] += v0
        acc[f'{pname}_V1'] += v1
        acc[f'{pname}_V1_better'] += v1 < v0
        acc[f'{pname}_m2'] += st['m2']
        acc[f'{pname}_m2_gain'] += st['m2_gain']
        acc[f'{pname}_m2_neutral'] += st['m2_neutral']
        if pname != 'G':
            emit_and_verify(pool1, fn, n, shift_c)
            acc[f'{pname}_emit_ok'] += 1
    acc['cases'] += 1


def main():
    lines = ['P9 探针:宏 M2 生产式相间调度(五项路径边际)', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = Counter()
        for fn in data:
            run_case(fn, n, acc)
        c = acc['cases']
        msg = [f'====== n = {n}({cases} cases)======']
        for pname in ('A', 'C', 'F', 'FP', 'G'):
            v0, v1 = acc[f'{pname}_V0'], acc[f'{pname}_V1']
            msg.append(f'{pname:>3} 池: V0={v0}  V1={v1}  ({(v1/v0-1):+.2%},更优 '
                       f'{acc[f"{pname}_V1_better"]}/{c}, M2 {acc[f"{pname}_m2"]} 次/Σgain '
                       f'{acc[f"{pname}_m2_gain"]},中性对 '
                       f'{acc[f"{pname}_m2_neutral"]})')
        msg.append(f'A/C/F/FP 发射级 compute_tt 验证: {acc["A_emit_ok"]}/{c} 通过')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
