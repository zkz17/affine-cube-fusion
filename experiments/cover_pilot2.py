# -*- coding: utf-8 -*-
"""
Pilot run for the PKS/PKP cover routes (branch exp/starting-cover-routes).

Measures T / CX / wall per configuration (PK, PKS, PKP with M2 on; FS and FP as
references) and the degree bound, plus a small MT translation-variance diagnostic
(16 fixed shifts per case at n=7/8) to justify skipping shift search on minterm pools.

Writes the table to experiments/phase0_results/cover_pilot2.txt.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import config
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_costs
from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import anf_degree, fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.CoverGenerators import minterm_entries

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'cover_pilot2.txt')
PROFILE = {6: 5, 7: 5, 8: 5, 9: 5, 10: 2, 11: 2}


def run_one(fn, n, route, m2=True):
    s = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, m2_post=m2)
    t0 = time.perf_counter()
    s.clear()
    s.allocate_input_qubit(n)
    s.process_cover_route(TruthTable(fn, n), route)
    wall = time.perf_counter() - t0
    assert s.circ.verify(TruthTable(fn, n)), f'verify failed: {route} n={n} tt={fn}'
    t, cx = count_costs(to_qiskit(s.circ))
    return t, cx, wall


def main():
    lines = ['PKS/PKP pilot (T / CX / wall; PK, FS, FP references; MT shift-variance diagnostic)', '']
    for n, cases in PROFILE.items():
        fns = [int(l) for l in open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = {r: [0, 0, 0.0] for r in ('PK', 'PKS', 'PKP', 'FS', 'FP')}
        bound_sum = 0
        for fn in fns:
            bound_sum += config['t_cost'](anf_degree(fn, n))
            for r in ('PK', 'PKS', 'PKP'):
                t, cx, w = run_one(fn, n, r)
                acc[r][0] += t
                acc[r][1] += cx
                acc[r][2] += w
            for r, kwargs in (('FS', {}), ('FP', dict(fprm_proxy=True))):
                s = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, **kwargs)
                t0 = time.perf_counter()
                s.synthesize(TruthTable(fn, n))
                acc[r][2] += time.perf_counter() - t0
                t, cx = count_costs(to_qiskit(s.circ))
                acc[r][0] += t
                acc[r][1] += cx
        msg = [f'====== n = {n}({cases} cases)======']
        msg.append(f'{"config":<8}{"ΣT":>10}{"ΣCX":>10}{"wall(ms/例)":>14}')
        for r in ('PK', 'PKS', 'PKP', 'FS', 'FP'):
            msg.append(f'{r:<8}{acc[r][0]:>10}{acc[r][1]:>10}{acc[r][2]/cases*1e3:>13.1f}')
        msg.append(f'{"bound":<8}{bound_sum:>10}')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']

    # ---- MT translation-variance diagnostic ----
    lines.append('=== MT translation-variance diagnostic (16 fixed shifts per case) ===')
    for n in (7, 8):
        fns = [int(l) for l in open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:3]]
        msg = [f'-- n = {n} --']
        for fn in fns:
            costs = []
            for c in range(16):
                kept, st = fuse_pool_global(minterm_entries(shift_fn(fn, c, n), n), n)
                costs.append(pool_t_cost(kept, st))
            span = (max(costs) - min(costs)) / max(costs)
            msg.append(f'  tt={fn}: min={min(costs)} max={max(costs)} '
                       f'range={span:+.2%} (16 shifts)')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nresults written to {OUT}')


if __name__ == '__main__':
    main()
