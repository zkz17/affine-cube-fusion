# -*- coding: utf-8 -*-
"""
G-2.1 probe: gain measurement of FPRM shift-subgroup search (phase 3 preliminary study).

Applying the shift x ↦ x⊕c to f: the circuit implementation of c is an X(c) sandwich (T-free), hence
  ΣT (ANF-pool fusion of the optimal c) = the real T cost of the realizable circuit.
c=0 is the current ANF path ⇒ the min combination is structurally a zero regression.

Metrics:
  base     = min(A_BDD, F_{c=0})                  -- current production backend
  improved = min(A_BDD, F pool best over c)       -- FPRM search backend
  Also record the t(deg f) lower bound (G-3) and the F-path time per c (which decides the full-enumeration/sampling budget).

Setup: reuse OracleSynthesizer._bdd_entries (A pool) and anf_pool + fuse_pool_global (F pool);
the XOR invariant is fully checked for n≤7 (function space after shifting).

Output: experiments/phase0_results/fprm_probe.txt
Usage: ./venv/bin/python experiments/fprm_probe.py
"""
import argparse
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from backend.OracleSynthesizer import OracleSynthesizer
from backend.CompleteBranch import t_cost
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'fprm_probe.txt')

# n -> (cases, shift-set size; None = full enumeration of 2^n)
PROFILE = {
    5: (80, None),
    6: (60, None),
    7: (50, None),
    8: (20, None),
    9: (10, 64),
    10: (5, 64),
}


def xor_pool(kept, standalone):
    x = 0
    for cb, r, _ in kept:
        x ^= shift_fn(cb.fn(), r, cb.n)
    for cb in standalone:
        x ^= cb.fn()
    return x


def anf_degree(bitstr, n):
    """ANF degree of f = maximum popcount of a monomial mask (input to the G-3 lower bound t(deg f))."""
    deg = 0
    for cb, _, _ in anf_pool(bitstr, n):
        deg = max(deg, cb.mask.bit_count())
    return deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20250821)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    lines = ['G-2.1 probe: FPRM shift-subgroup search gain measurement', '']
    for n, (cases, n_shifts) in PROFILE.items():
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            data = [int(line.rstrip('\n')) for line in f.readlines()[:cases]]

        all_c = list(range(1 << n))
        shifts = all_c if n_shifts is None else [0] + rng.sample(all_c[1:], n_shifts - 1)

        stat = Counter()
        s_base = s_imp = s_lb = 0
        c_time = 0.0
        for tt_int in data:
            tt = TruthTable(tt_int, n)

            entries_a = synth._bdd_entries(tt, None)
            kept_a, st_a = fuse_pool_global(entries_a, n)
            cost_a = pool_t_cost(kept_a, st_a)

            t0 = time.perf_counter()
            cost_c0, best_c = None, None
            for c in shifts:
                g = shift_fn(tt.bitstr, c, n)
                kept, standalone = fuse_pool_global(anf_pool(g, n), n)
                if n <= 7:
                    assert xor_pool(kept, standalone) == g, f'XOR invariant violated n={n} c={c}'
                cost = pool_t_cost(kept, standalone)
                if c == 0:
                    cost_c0 = cost
                if best_c is None or cost < best_c[0]:
                    best_c = (cost, c)
            c_time += time.perf_counter() - t0

            base = min(cost_a, cost_c0)
            improved = min(cost_a, best_c[0])
            lb = t_cost(anf_degree(tt.bitstr, n))

            s_base += base
            s_imp += improved
            s_lb += lb
            stat['better'] += improved < base
            stat['same'] += improved == base
            stat['cases'] += 1

        wall_c = c_time
        per_c_ms = wall_c / (cases * len(shifts)) * 1000
        msg = [
            f'====== n = {n} ======',
            f'cases={cases} shifts={len(shifts)}/2^{n}  F-path total time {wall_c:.1f}s(mean per c {per_c_ms:.1f} ms)',
            f'ΣT: base={s_base} improved={s_imp}  Δ {s_imp - s_base:+d}({(s_base - s_imp) / max(s_base, 1):.2%})',
            f'cases: better={stat["better"]} same={stat["same"]}',
            f'G-3 lower bound Σ t(deg f)={s_lb}: {s_lb / max(s_base, 1):.2%} of base;'
            f'headroom base→bound {(s_base - s_lb) / max(s_base, 1):.2%}, improved→bound {(s_imp - s_lb) / max(s_imp, 1):.2%}',
            f'estimated F-path time for full 2^{n} shifts: {per_c_ms * (1 << n) / 1000:.1f}s/case',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nresults written to {OUT}')


if __name__ == '__main__':
    main()
