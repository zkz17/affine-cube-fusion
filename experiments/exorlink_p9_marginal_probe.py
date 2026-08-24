# -*- coding: utf-8 -*-
"""
P9-M 边际探针:m2_post 开关的生产端到端 T/CX 对比(五项路径)。

直接用生产合成器(m2_post=False / True)综合同一批测例,发射后按
evaluate.py 的 count_costs(T/CX 双口径)计数——与最终生产路径完全一致
(含 G 的线性 wrapper、B-4 段重排、simplify)。规模 n=5/6/7(25/15/6),
与 P9 探针同口径,只写 experiments/phase0_results/。
"""
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_costs
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'exorlink_p9_marginal_probe.txt')
PROFILE = {5: 25, 6: 15, 7: 6}
CONFIGS = [('F', False, False), ('A', False, False), ('C', False, False),
           ('FP', False, True), ('G', True, False)]


def main():
    lines = ['P9-M 边际探针:生产端到端 m2_post 开关 T/CX 对比(五项路径)', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = Counter()
        for cfg, gl, proxy in CONFIGS:
            kwargs = dict(reorder=True, ancilla_num=0, fused_backend=True,
                          gl_search=gl, fprm_proxy=proxy,
                          paths=('A',) if cfg == 'A' else ('C',) if cfg == 'C' else ('F',))
            off = LimBDDOracleSynthesizer(m2_post=False, **kwargs)
            on = LimBDDOracleSynthesizer(m2_post=True, **kwargs)
            for fn in data:
                tt = TruthTable(fn, n)
                t0 = time.perf_counter()
                off.synthesize(tt)
                acc[f'{cfg}_off_wall'] += time.perf_counter() - t0
                t_off, cx_off = count_costs(to_qiskit(off.circ))

                t0 = time.perf_counter()
                on.synthesize(tt)
                acc[f'{cfg}_on_wall'] += time.perf_counter() - t0
                t_on, cx_on = count_costs(to_qiskit(on.circ))
                assert on.circ.verify(tt), f'[{cfg}] M2 发射验证失败: tt={fn}'

                acc[f'{cfg}_T_off'] += t_off
                acc[f'{cfg}_T_on'] += t_on
                acc[f'{cfg}_CX_off'] += cx_off
                acc[f'{cfg}_CX_on'] += cx_on
                acc[f'{cfg}_T_better'] += t_on < t_off
                acc[f'{cfg}_CX_better'] += cx_on < cx_off
                acc[f'{cfg}_CX_worse'] += cx_on > cx_off
        msg = [f'====== n = {n}({cases} cases)======']
        for cfg, gl, proxy in CONFIGS:
            t0, t1 = acc[f'{cfg}_T_off'], acc[f'{cfg}_T_on']
            c0, c1 = acc[f'{cfg}_CX_off'], acc[f'{cfg}_CX_on']
            dt = f'{(t1/t0-1):+.2%}({acc[f"{cfg}_T_better"]}/{cases})' if t0 else 'n/a'
            dc = f'{(c1/c0-1):+.2%}(优{acc[f"{cfg}_CX_better"]}劣{acc[f"{cfg}_CX_worse"]}/{cases})' if c0 else 'n/a'
            wall = (acc[f'{cfg}_on_wall'] - acc[f'{cfg}_off_wall']) / cases * 1e3
            msg.append(f'{cfg:>3}  ΣT: {t0}→{t1} ({dt})   ΣCX: {c0}→{c1} ({dc})   '
                       f'M2增量 {max(wall,0):.2f} ms/例')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
