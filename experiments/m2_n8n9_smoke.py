# -*- coding: utf-8 -*-
"""
n≥8 timing smoke test: per-case time breakdown for the production default backends (FS / FP)
plus the M2 margin; the F pool is measured as a side effect of the M2 enumeration upper bound
(maximum number of segments). A/FG are not measured (A is not the production default and BDD
graph construction is slow at large n; FG is a small-n strengthening option). Reads only
data/test_cases and writes only experiments/phase0_results/.

Usage: ./venv/bin/python experiments/m2_n8n9_smoke.py <n> <cases>
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_costs
from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.GLSearch import fprm_best
from backend.fusion.FprmProxy import fprm_proxy_best
from backend.fusion.MacroM2 import m2_interleaved

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'm2_n8n9_smoke.txt')


def fmt(x):
    return f'{x*1e3:8.2f}'


def main():
    n = int(sys.argv[1])
    cases = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    data = [int(l.rstrip('\n')) for l in
            open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]

    f_off = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                    m2_post=False)
    f_on = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                   m2_post=True)
    fp_off = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                     fprm_proxy=True, m2_post=False)
    fp_on = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                    fprm_proxy=True, m2_post=True)

    acc = dict(FS_prep=0.0, FS_off=0.0, FS_on=0.0, FS_T0=0, FS_T1=0, FS_CX0=0, FS_CX1=0,
               FS_gain=0, FS_m2wall=0.0,
               FP_prep=0.0, FP_off=0.0, FP_on=0.0, FP_T0=0, FP_T1=0, FP_CX0=0, FP_CX1=0,
               FP_gain=0, FP_m2wall=0.0,
               F_off=0.0, F_on=0.0, F_gain=0, F_m2wall=0.0)
    lines = [f'====== n = {n}({cases} cases), ms per case ======', '']

    for idx, fn in enumerate(data):
        tt = TruthTable(fn, n)
        row = [f'--- case {idx + 1}/{cases}(tt={fn})']

        t0 = time.perf_counter()
        _, cF = fprm_best(fn, n)
        tp = time.perf_counter() - t0
        acc['FS_prep'] += tp
        row.append(f'  FS  prep(fprm_best)      {fmt(tp)} ms')

        g = shift_fn(fn, cF, n) if cF else fn
        kept, st = fuse_pool_global(anf_pool(g, n), n)
        base = pool_t_cost(kept, st)
        t0 = time.perf_counter()
        k2, s2 = m2_interleaved(kept, st, n)
        mw = time.perf_counter() - t0
        acc['FS_m2wall'] += mw
        acc['FS_gain'] += base - pool_t_cost(k2, s2)
        row.append(f'  FS  M2 post-process      {fmt(mw)} ms(pool-T gain {base - pool_t_cost(k2, s2)})')

        t0 = time.perf_counter()
        f_off.synthesize(tt)
        toff = time.perf_counter() - t0
        acc['FS_off'] += toff
        t0f, cx0 = count_costs(to_qiskit(f_off.circ))
        acc['FS_T0'] += t0f
        acc['FS_CX0'] += cx0

        t0 = time.perf_counter()
        f_on.synthesize(tt)
        ton = time.perf_counter() - t0
        acc['FS_on'] += ton
        t1f, cx1 = count_costs(to_qiskit(f_on.circ))
        acc['FS_T1'] += t1f
        acc['FS_CX1'] += cx1
        assert f_on.circ.verify(tt)
        row.append(f'  FS  whole case: off {fmt(toff)}  on {fmt(ton)} ms  '
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
        row.append(f'  FP M2 post-process      {fmt(mw)} ms(pool-T gain {base - pool_t_cost(k2, s2)})')

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
        row.append(f'  FP whole case: off {fmt(toff)}  on {fmt(ton)} ms  '
                   f'(T {t0f}→{t1f}, CX {cx0}→{cx1})')

        kept, st = fuse_pool_global(anf_pool(fn, n), n)
        base = pool_t_cost(kept, st)
        t0 = time.perf_counter()
        k2, s2 = m2_interleaved(kept, st, n)
        mw = time.perf_counter() - t0
        acc['F_m2wall'] += mw
        acc['F_gain'] += base - pool_t_cost(k2, s2)
        row.append(f'  F  M2 post-process      {fmt(mw)} ms(pool-T gain {base - pool_t_cost(k2, s2)},'
                   f'pool m={len(kept) + len(st)})')

        print('\n'.join(row), flush=True)
        lines += row

    c = len(data)
    msg = [f'--- n={n} averages (ms/case) ---',
           f'FS  : prep={fmt(acc["FS_prep"]/c)}  whole off={fmt(acc["FS_off"]/c)}  '
           f'on={fmt(acc["FS_on"]/c)}  M2={fmt(acc["FS_m2wall"]/c)}  pool-T gain Σ={acc["FS_gain"]}',
           f'FP : prep={fmt(acc["FP_prep"]/c)}  whole off={fmt(acc["FP_off"]/c)}  '
           f'on={fmt(acc["FP_on"]/c)}  M2={fmt(acc["FP_m2wall"]/c)}  pool-T gain Σ={acc["FP_gain"]}',
           f'F  : M2={fmt(acc["F_m2wall"]/c)}  pool-T gain Σ={acc["F_gain"]}',
           f'ΣT: FS {acc["FS_T0"]}→{acc["FS_T1"]}({(acc["FS_T1"]/acc["FS_T0"]-1):+.2%})  '
           f'FP {acc["FP_T0"]}→{acc["FP_T1"]}({(acc["FP_T1"]/acc["FP_T0"]-1):+.2%})',
           f'ΣCX: FS {acc["FS_CX0"]}→{acc["FS_CX1"]}({(acc["FS_CX1"]/acc["FS_CX0"]-1):+.2%})  '
           f'FP {acc["FP_CX0"]}→{acc["FP_CX1"]}({(acc["FP_CX1"]/acc["FP_CX0"]-1):+.2%})']
    print('\n'.join(msg), flush=True)
    lines += msg + ['']
    with open(OUT, 'a') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
