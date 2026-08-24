# -*- coding: utf-8 -*-
"""
n≥8 时间冒烟:生产默认后端(F / FP)每例时间构成 + M2 边际;C 池顺带量
M2 枚举上限(段最多)。A/G 不测(A 非生产默认且 BDD 构图在大 n 慢;
G 是小 n 强化选项)。只读 data/test_cases,只写 experiments/phase0_results/。

用法: ./venv/bin/python experiments/m2_n8n9_smoke.py <n> <cases>
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_costs
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from limbdd.fusion.GLSearch import fprm_best
from limbdd.fusion.FprmProxy import fprm_proxy_best
from limbdd.fusion.MacroM2 import m2_interleaved

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'm2_n8n9_smoke.txt')


def fmt(x):
    return f'{x*1e3:8.2f}'


def main():
    n = int(sys.argv[1])
    cases = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    data = [int(l.rstrip('\n')) for l in
            open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]

    f_off = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                    m2_post=False)
    f_on = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                   m2_post=True)
    fp_off = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                     fprm_proxy=True, m2_post=False)
    fp_on = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                    fprm_proxy=True, m2_post=True)

    acc = dict(F_prep=0.0, F_off=0.0, F_on=0.0, F_T0=0, F_T1=0, F_CX0=0, F_CX1=0,
               F_gain=0, F_m2wall=0.0,
               FP_prep=0.0, FP_off=0.0, FP_on=0.0, FP_T0=0, FP_T1=0, FP_CX0=0, FP_CX1=0,
               FP_gain=0, FP_m2wall=0.0,
               C_off=0.0, C_on=0.0, C_gain=0, C_m2wall=0.0)
    lines = [f'====== n = {n}({cases} cases), 每例 ms ======', '']

    for idx, fn in enumerate(data):
        tt = TruthTable(fn, n)
        row = [f'--- 例 {idx + 1}/{cases}(tt={fn})']

        t0 = time.perf_counter()
        _, cF = fprm_best(fn, n)
        tp = time.perf_counter() - t0
        acc['F_prep'] += tp
        row.append(f'  F  prep(fprm_best)      {fmt(tp)} ms')

        g = shift_fn(fn, cF, n) if cF else fn
        kept, st = fuse_pool_global(anf_pool(g, n), n)
        base = pool_t_cost(kept, st)
        t0 = time.perf_counter()
        k2, s2 = m2_interleaved(kept, st, n)
        mw = time.perf_counter() - t0
        acc['F_m2wall'] += mw
        acc['F_gain'] += base - pool_t_cost(k2, s2)
        row.append(f'  F  M2 后处理           {fmt(mw)} ms(池T收益 {base - pool_t_cost(k2, s2)})')

        t0 = time.perf_counter()
        f_off.synthesize(tt)
        toff = time.perf_counter() - t0
        acc['F_off'] += toff
        t0f, cx0 = count_costs(to_qiskit(f_off.circ))
        acc['F_T0'] += t0f
        acc['F_CX0'] += cx0

        t0 = time.perf_counter()
        f_on.synthesize(tt)
        ton = time.perf_counter() - t0
        acc['F_on'] += ton
        t1f, cx1 = count_costs(to_qiskit(f_on.circ))
        acc['F_T1'] += t1f
        acc['F_CX1'] += cx1
        assert f_on.circ.verify(tt)
        row.append(f'  F  整例: off {fmt(toff)}  on {fmt(ton)} ms  '
                   f'(T {t0f}→{t1f}, CX {cx0}→{cx1})')

        t0 = time.perf_counter()
        _, _, cFp = fprm_proxy_best(fn, n)
        tp = time.perf_counter() - t0
        acc['FP_prep'] += tp
        row.append(f'  FP prep(proxy_best)    {fmt(tp)} ms(c={cFp})')

        g = shift_fn(fn, cFp, n) if cFp else fn
        kept, st = fuse_pool_global(anf_pool(g, n), n)
        t0 = time.perf_counter()
        k2, s2 = m2_interleaved(kept, st, n)
        mw = time.perf_counter() - t0
        acc['FP_m2wall'] += mw
        base = pool_t_cost(kept, st)
        acc['FP_gain'] += base - pool_t_cost(k2, s2)
        row.append(f'  FP M2 后处理           {fmt(mw)} ms(池T收益 {base - pool_t_cost(k2, s2)})')

        t0 = time.perf_counter()
        fp_off.synthesize(tt)
        toff = time.perf_counter() - t0
        acc['FP_off'] += toff
        t0f, cx0 = count_costs(to_qiskit(fp_off.circ))
        acc['FP_T0'] += t0f
        acc['FP_CX0'] += cx0

        t0 = time.perf_counter()
        fp_on.synthesize(tt)
        ton = time.perf_counter() - t0
        acc['FP_on'] += ton
        t1f, cx1 = count_costs(to_qiskit(fp_on.circ))
        acc['FP_T1'] += t1f
        acc['FP_CX1'] += cx1
        assert fp_on.circ.verify(tt)
        row.append(f'  FP 整例: off {fmt(toff)}  on {fmt(ton)} ms  '
                   f'(T {t0f}→{t1f}, CX {cx0}→{cx1})')

        kept, st = fuse_pool_global(anf_pool(fn, n), n)
        base = pool_t_cost(kept, st)
        t0 = time.perf_counter()
        k2, s2 = m2_interleaved(kept, st, n)
        mw = time.perf_counter() - t0
        acc['C_m2wall'] += mw
        acc['C_gain'] += base - pool_t_cost(k2, s2)
        row.append(f'  C  M2 后处理           {fmt(mw)} ms(池T收益 {base - pool_t_cost(k2, s2)},'
                   f'池 m={len(kept) + len(st)})')

        print('\n'.join(row), flush=True)
        lines += row

    c = len(data)
    msg = [f'--- n={n} 平均(ms/例) ---',
           f'F  : prep={fmt(acc["F_prep"]/c)}  整例 off={fmt(acc["F_off"]/c)}  '
           f'on={fmt(acc["F_on"]/c)}  M2={fmt(acc["F_m2wall"]/c)}  池T收益 Σ={acc["F_gain"]}',
           f'FP : prep={fmt(acc["FP_prep"]/c)}  整例 off={fmt(acc["FP_off"]/c)}  '
           f'on={fmt(acc["FP_on"]/c)}  M2={fmt(acc["FP_m2wall"]/c)}  池T收益 Σ={acc["FP_gain"]}',
           f'C  : M2={fmt(acc["C_m2wall"]/c)}  池T收益 Σ={acc["C_gain"]}',
           f'ΣT: F {acc["F_T0"]}→{acc["F_T1"]}({(acc["F_T1"]/acc["F_T0"]-1):+.2%})  '
           f'FP {acc["FP_T0"]}→{acc["FP_T1"]}({(acc["FP_T1"]/acc["FP_T0"]-1):+.2%})',
           f'ΣCX: F {acc["F_CX0"]}→{acc["F_CX1"]}({(acc["F_CX1"]/acc["F_CX0"]-1):+.2%})  '
           f'FP {acc["FP_CX0"]}→{acc["FP_CX1"]}({(acc["FP_CX1"]/acc["FP_CX0"]-1):+.2%})']
    print('\n'.join(msg), flush=True)
    lines += msg + ['']
    with open(OUT, 'a') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
