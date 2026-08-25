# -*- coding: utf-8 -*-
"""
Phase 2 hypothesis-testing probe (G-1a / G-1b): are plateau moves worthwhile?

Control setup (line-by-line consistent with the production backend):
  A pool = OracleSynthesizer._bdd_entries(tt) (graph-stage product, with root masks);
  C pool = anf_pool(tt.bitstr, n) (pure algebraic product).

Three variants per pool:
  1. strict       : fuse_pool_global(entries, n)                      -- current state (gain > 0)
  2. neutral      : fuse_pool_global(entries, n, accept_neutral=True) -- G-1a (gain >= 0)
  3. neutral+cat  : same as above, but first inject zero-cost catalyst triples {[x_j=1], [x_j=0], 1} for all j
                    (ΣT=0, XOR=0 ⇒ pool invariant preserved)          -- G-1b

Production-grade metrics: base = min(A_strict, C_strict); improved = min(A_best, C_best).
Judgment: whether the ΣT reduction of improved over base is significant (refer to the decision-convention threshold ~0.5%).

Output: experiments/phase0_results/phase2_plateau_probe.txt
Usage: ./venv/bin/python experiments/phase2_plateau_probe.py [--cases 200] [4 5 6 7 8]
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from backend.OracleSynthesizer import OracleSynthesizer
from backend.CompleteBranch import CompleteBranch
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'phase2_plateau_probe.txt')


def catalyst_entries(n):
    """Zero-cost catalyst triples: [x_j=1] ⊕ [x_j=0] ⊕ 1 = 0 (t(1)+t(1)+t(0) = 0)."""
    out = []
    for j in range(n):
        m = 1 << j
        out.append((CompleteBranch(n, mask=m, bitstr=m), 0, ('cat', j, 1)))
        out.append((CompleteBranch(n, mask=m, bitstr=0), 0, ('cat', j, 0)))
        out.append((CompleteBranch(n, mask=0, bitstr=0), 0, ('cat', j, -1)))
    return out


def xor_pool(kept, standalone):
    """Pool XOR invariant, checked in effective-function space: kept segments are first shifted by root mask r."""
    x = 0
    for cb, r, _ in kept:
        x ^= shift_fn(cb.fn(), r, cb.n)
    for cb in standalone:
        x ^= cb.fn()
    return x


def path_variants(entries, n, tt_bitstr, cats):
    """Three variants of one pool; returns (costs, best_cost, best_name). Verifies the XOR invariant of each variant."""
    results = {}
    k0, s0 = fuse_pool_global(entries, n)
    assert xor_pool(k0, s0) == tt_bitstr, 'strict XOR invariant violated'
    results['strict'] = pool_t_cost(k0, s0)

    k1, s1 = fuse_pool_global(entries, n, accept_neutral=True)
    assert xor_pool(k1, s1) == tt_bitstr, 'neutral XOR invariant violated'
    results['neutral'] = pool_t_cost(k1, s1)

    k2, s2 = fuse_pool_global(entries + cats, n, accept_neutral=True)
    assert xor_pool(k2, s2) == tt_bitstr, 'neutral+cat XOR invariant violated'
    results['neutral+cat'] = pool_t_cost(k2, s2)

    best_name = min(results, key=results.get)
    return results, results[best_name], best_name


def run_case(synth, n, tt, cats):
    entries_a = synth._bdd_entries(tt, None)
    entries_c = anf_pool(tt.bitstr, n)

    ra, ca_a, na = path_variants(entries_a, n, tt.bitstr, cats)
    rc, ca_c, nc = path_variants(entries_c, n, tt.bitstr, cats)

    base = min(ra['strict'], rc['strict'])
    improved = min(ca_a, ca_c)
    assert improved <= base, 'must not degrade under the min combination'
    return base, improved, na if ca_a <= ca_c else nc, ra, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ns', type=int, nargs='*', default=[4, 5, 6, 7, 8])
    ap.add_argument('--cases', type=int, default=300, help='auto-halved for n=8')
    args = ap.parse_args()

    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    lines = ['Stage-2 hypothesis test: end-to-end gain measurement of G-1a (gain>=0) / G-1b (catalyst triples)', '']
    for n in args.ns:
        cases = args.cases if n <= 7 else max(1, args.cases // 3)
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            data = [int(line.rstrip('\n')) for line in f.readlines()[:cases]]
        cats = catalyst_entries(n)

        stat = Counter()
        s_base = s_imp = 0
        t0 = time.perf_counter()
        for tt_int in data:
            tt = TruthTable(tt_int, n)
            base, improved, winner, ra, rc = run_case(synth, n, tt, cats)
            s_base += base
            s_imp += improved
            stat['better'] += improved < base
            stat['same'] += improved == base
            for name in ('strict', 'neutral', 'neutral+cat'):
                if ra[name] <= min(ra.values()):
                    stat[f'A_win_{name}'] += 1
                if rc[name] <= min(rc.values()):
                    stat[f'C_win_{name}'] += 1
            stat['A_best_neutral'] += (ra['neutral'] < ra['strict'])
            stat['A_best_cat'] += (ra['neutral+cat'] < min(ra['strict'], ra['neutral']))
            stat['C_best_neutral'] += (rc['neutral'] < rc['strict'])
            stat['C_best_cat'] += (rc['neutral+cat'] < min(rc['strict'], rc['neutral']))
        wall = time.perf_counter() - t0

        msg = [
            f'====== n = {n}({cases} cases,{wall:.1f}s)======',
            f'ΣT: base={s_base} improved={s_imp}(Δ {s_imp - s_base:+d}, {(s_base - s_imp) / max(s_base, 1):.2%})',
            f'cases: better={stat["better"]} same={stat["same"]}',
            f'A-pool best-variant distribution: strict={stat["A_win_strict"]} neutral={stat["A_win_neutral"]} neutral+cat={stat["A_win_neutral+cat"]}',
            f'C-pool best-variant distribution: strict={stat["C_win_strict"]} neutral={stat["C_win_neutral"]} neutral+cat={stat["C_win_neutral+cat"]}',
            f'neutral strictly better than strict: A {stat["A_best_neutral"]} cases / C {stat["C_best_neutral"]} cases;'
            f' cat better still than neutral: A {stat["A_best_cat"]} / C {stat["C_best_cat"]}',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nresults written to {OUT}')


if __name__ == '__main__':
    main()
