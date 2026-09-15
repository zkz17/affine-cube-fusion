# -*- coding: utf-8 -*-
"""
Pilot run for the experimental starting-cover routes (branch exp/starting-cover-routes).

Measures, per test case and per configuration (MT / PK, each with M2 on/off):
T, CX, wall, and verify; prints per-n aggregates together with the F-route reference
and the degree lower bound. Small case counts (5 per n for n=6..9, 2 for n=10/11).

Writes the table to experiments/phase0_results/cover_pilot.txt.
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
from backend.fusion.BranchFusion import anf_degree

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'cover_pilot.txt')
PROFILE = {6: 5, 7: 5, 8: 5, 9: 5, 10: 2, 11: 2}
ROUTES = (('MT', True), ('MT', False), ('PK', True), ('PK', False))


def main():
    lines = ['Cover-route pilot (MT/PK x M2 on/off, T / CX / wall, with F reference and degree bound)', '']
    for n, cases in PROFILE.items():
        fns = [int(l) for l in open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = {f'{r}{m2}': [0, 0, 0.0] for r, m2 in [('MT', 'on'), ('MT', 'off'), ('PK', 'on'), ('PK', 'off')]}
        f_ref = [0, 0, 0.0]
        bound_sum = 0
        for fn in fns:
            tt = TruthTable(fn, n)
            bound_sum += config['t_cost'](anf_degree(fn, n))
            for r, m2 in ROUTES:
                s = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, m2_post=m2)
                t0 = time.perf_counter()
                s.clear()
                s.allocate_input_qubit(n)
                s.process_cover_route(tt, r)
                wall = time.perf_counter() - t0
                assert s.circ.verify(tt), f'verify failed: {r} m2={m2} n={n} tt={fn}'
                t, cx = count_costs(to_qiskit(s.circ))
                key = f'{r}{"on" if m2 else "off"}'
                acc[key][0] += t
                acc[key][1] += cx
                acc[key][2] += wall
            sf = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
            t0 = time.perf_counter()
            sf.synthesize(tt)
            f_ref[2] += time.perf_counter() - t0
            t, cx = count_costs(to_qiskit(sf.circ))
            f_ref[0] += t
            f_ref[1] += cx
        msg = [f'====== n = {n}({cases} cases)======']
        hdr = f'{"config":<10}{"ΣT":>10}{"ΣCX":>10}{"wall(ms/例)":>14}'
        msg.append(hdr)
        for r, m2 in ROUTES:
            key = f'{r}{"on" if m2 else "off"}'
            msg.append(f'{key:<10}{acc[key][0]:>10}{acc[key][1]:>10}{acc[key][2]/cases*1e3:>13.1f}')
        msg.append(f'{"F(ref)":<10}{f_ref[0]:>10}{f_ref[1]:>10}{f_ref[2]/cases*1e3:>13.1f}')
        msg.append(f'{"deg bound":<10}{bound_sum:>10}')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nresults written to {OUT}')


if __name__ == '__main__':
    main()
