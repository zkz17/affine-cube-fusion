# -*- coding: utf-8 -*-
"""
P9-M marginal probe: production end-to-end T/CX comparison for the m2_post switch (five paths).

Synthesize the same batch of test cases directly with the production synthesizer (m2_post=False / True), then after emission count with evaluate.py's count_costs (T/CX dual conventions) —— exactly matching the final production path
(including FG's linear wrapper, B-4 segment reordering, simplify). Sizes n=5/6/7 (25/15/6),
same convention as the P9 probe, only writes experiments/phase0_results/.
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
from backend.OracleSynthesizer import OracleSynthesizer

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'exorlink_p9_marginal_probe.txt')
PROFILE = {5: 25, 6: 15, 7: 6}
CONFIGS = [('FS', False, False), ('A', False, False), ('F', False, False),
           ('FP', False, True), ('FG', True, False)]


def main():
    lines = ['P9-M marginal probe: production end-to-end T/CX comparison of the m2_post switch (five paths)', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = Counter()
        for cfg, gl, proxy in CONFIGS:
            kwargs = dict(reorder=True, ancilla_num=0, fused_backend=True,
                          gl_search=gl, fprm_proxy=proxy,
                          paths=('A',) if cfg == 'A' else ('F',) if cfg == 'F' else ('FS',))
            off = OracleSynthesizer(m2_post=False, **kwargs)
            on = OracleSynthesizer(m2_post=True, **kwargs)
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
                assert on.circ.verify(tt), f'[{cfg}] M2 emission verification failed: tt={fn}'

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
            dc = f'{(c1/c0-1):+.2%}(better{acc[f"{cfg}_CX_better"]}worse{acc[f"{cfg}_CX_worse"]}/{cases})' if c0 else 'n/a'
            wall = (acc[f'{cfg}_on_wall'] - acc[f'{cfg}_off_wall']) / cases * 1e3
            msg.append(f'{cfg:>3}  ΣT: {t0}→{t1} ({dt})   ΣCX: {c0}→{c1} ({dc})   '
                       f'M2 delta {max(wall,0):.2f} ms/case')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nresults written to {OUT}')


if __name__ == '__main__':
    main()
