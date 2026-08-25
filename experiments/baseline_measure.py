# -*- coding: utf-8 -*-
"""
Acceptance baseline for phase 0.3: 1.1 (T-1 closed-form switch).

Records three kinds of quantities, all **non-intrusive** (no production code changes; only monkeypatch counting/timing inside this script):

1. **per-case ΣT** (quality invariant) -- 1.1 is a pure performance change and must stay **bit-identical** to this baseline after the switch;
2. **fusion phase wall time** -- the target that 1.1 optimizes, timed separately to quantify the speedup factor;
3. **call volume of the fn path** -- `is_affine_cube` / `shift_fn` call counts and their O(2^n) work,
   used to check whether the "eliminated 2^n factor" matches the measured speedup.

Output:
  experiments/phase0_results/baseline_tcost_{n}.txt   per-case `tt:T` (same format as evaluate.py, for easy diff)
  experiments/phase0_results/baseline_summary.txt     summary table

Usage:
  ./venv/bin/python experiments/baseline_measure.py            # default n=4..7
  ./venv/bin/python experiments/baseline_measure.py 4 5 6 7 --cases 200
"""
import argparse
import os
import sys
import time

# Project root (script lives under experiments/, so go up one level); must be added to sys.path before importing limbdd/evaluate
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENTS = os.path.join(ROOT, 'experiments')

import backend.fusion.BranchFusion as BF
import backend.OracleSynthesizer as SYN
from backend.OracleSynthesizer import OracleSynthesizer
from backend.CompleteBranch import t_cost as mcx_t_cost
from backend.fusion.BranchFusion import anf_degree
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_t_cost

OUT_DIR = os.path.join(EXPERIMENTS, 'phase0_results')


class Probe:
    """Timing and call counting for the fusion phase (monkeypatch; zero production code changes)."""

    def __init__(self):
        self.fuse_time = 0.0
        self.affine_calls = 0
        self.affine_work = 0      # Σ 2^n, i.e. the total enumeration size of is_affine_cube
        self.shift_calls = 0
        self.shift_work = 0
        self._orig = {}

    def install(self):
        # Note: Synthesizer binds the name via `from ... import fuse_pool_global`,
        # so we must replace the name in the **Synthesizer module namespace**, not the one in BF.
        orig_pool = SYN.fuse_pool_global
        self._orig = {'pool': orig_pool}

        # From 1.1 onward the production fusion path is closed-form and no longer goes through is_affine_cube / shift_fn:
        # if these names still exist in BranchFusion (e.g. when falling back to the old implementation), keep counting them;
        # otherwise stay at 0, letting the output columns directly prove the 2^n path has vanished from the production chain.
        if hasattr(BF, 'is_affine_cube'):
            orig_affine = BF.is_affine_cube
            self._orig['affine'] = orig_affine

            def affine(fn, n):
                self.affine_calls += 1
                self.affine_work += 1 << n
                return orig_affine(fn, n)

            BF.is_affine_cube = affine

        if hasattr(BF, 'shift_fn'):
            orig_shift = BF.shift_fn
            self._orig['shift'] = orig_shift

            def shift(fn, r, n):
                self.shift_calls += 1
                self.shift_work += 1 << n
                return orig_shift(fn, r, n)

            BF.shift_fn = shift

        def pool(entries, n, max_rounds=None):
            t0 = time.perf_counter()
            try:
                return orig_pool(entries, n, max_rounds)
            finally:
                self.fuse_time += time.perf_counter() - t0

        SYN.fuse_pool_global = pool

    def uninstall(self):
        for name, orig in self._orig.items():
            if name == 'pool':
                SYN.fuse_pool_global = orig
            else:
                setattr(BF, name, orig)


def load_cases(n, count):
    """New data layout (2025-08-22): test cases in data/test_cases/test_cases_{n} (one integer per line),
    Exorcism T baseline in data/exorcism_cost/exorcism_t_cost_{n} (aligned by line number; missing entries count as 0)."""
    path = os.path.join(ROOT, f'data/test_cases/test_cases_{n}')
    with open(path) as f:
        lines = f.readlines()
    count = min(count, len(lines))
    tts = [int(line.rstrip('\n')) for line in lines[:count]]

    exor_path = os.path.join(ROOT, f'data/exorcism_cost/exorcism_t_cost_{n}')
    exor = [0] * count
    try:
        with open(exor_path) as f:
            elines = f.readlines()
        for i, line in enumerate(elines[:count]):
            exor[i] = int(line.split(':')[1])
    except FileNotFoundError:
        pass
    return list(zip(tts, exor))


def measure(n, count, verify):
    cases = load_cases(n, count)
    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    probe = Probe()
    probe.install()

    rows = []
    t_sum, exor_sum, bound_sum, wall = 0, 0, 0, 0.0
    failures = []
    try:
        for tt_int, t_exor in cases:
            tt = TruthTable(tt_int, n)
            t0 = time.perf_counter()
            synth.synthesize(tt)
            wall += time.perf_counter() - t0

            t_cost = count_t_cost(to_qiskit(synth.circ))
            rows.append((tt_int, t_cost))
            t_sum += t_cost
            exor_sum += t_exor
            # G-3: free quality lower bound t(deg f)
            bound_sum += mcx_t_cost(anf_degree(tt.bitstr, n))

            if verify and not synth.circ.verify(tt):
                failures.append(tt_int)
    finally:
        probe.uninstall()

    with open(os.path.join(OUT_DIR, f'baseline_tcost_{n}.txt'), 'w') as f:
        for tt_int, t_cost in rows:
            f.write(f'{tt_int}:{t_cost}\n')

    return {
        'n': n, 'cases': len(cases), 'sum_t': t_sum, 'sum_exor': exor_sum,
        'sum_bound': bound_sum,
        'wall': wall, 'fuse_time': probe.fuse_time,
        'affine_calls': probe.affine_calls, 'affine_work': probe.affine_work,
        'shift_calls': probe.shift_calls, 'shift_work': probe.shift_work,
        'failures': failures,
    }


def main():
    ap = argparse.ArgumentParser(description='Phase 0.3 baseline measurement')
    ap.add_argument('ns', type=int, nargs='*', default=[4, 5, 6, 7])
    ap.add_argument('--cases', type=int, default=200, help='How many leading cases to take per n')
    ap.add_argument('--no-verify', action='store_true', help='Skip per-case compute_tt verification')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for n in args.ns:
        print(f'--- n={n} ({args.cases} cases) ---', flush=True)
        r = measure(n, args.cases, not args.no_verify)
        results.append(r)
        print(f'  ΣT={r["sum_t"]} (exorcism {r["sum_exor"]})  G-3 lower bound Σt(deg f)={r["sum_bound"]} '
              f'({r["sum_bound"] / max(r["sum_t"], 1):.0%} of ΣT)  '
              f'wall={r["wall"]:.2f}s  fusion={r["fuse_time"]:.2f}s '
              f'({r["fuse_time"] / max(r["wall"], 1e-9):.0%} of wall)', flush=True)
        print(f'  is_affine_cube: {r["affine_calls"]} calls / {r["affine_work"]:,} enum steps;  '
              f'shift_fn: {r["shift_calls"]} calls / {r["shift_work"]:,} steps', flush=True)
        if r['failures']:
            print(f'  !!! VERIFY FAILED for {len(r["failures"])} cases: {r["failures"][:5]}', flush=True)

    lines = []
    lines.append('Phase 0.3 baseline (acceptance reference for the 1.1 closed-form switch)')
    lines.append(f'Command: {" ".join(sys.argv)}')
    lines.append('')
    header = f'{"n":>3} {"cases":>6} {"ΣT":>10} {"ΣT_exor":>10} {"Σbound":>8} {"wall(s)":>9} {"fusion(s)":>10} {"fus%":>5} {"affine_calls":>13} {"affine_steps":>14} {"shift_calls":>12} {"verify":>7}'
    lines.append(header)
    lines.append('-' * len(header))
    for r in results:
        lines.append(f'{r["n"]:>3} {r["cases"]:>6} {r["sum_t"]:>10} {r["sum_exor"]:>10} {r["sum_bound"]:>8} '
                     f'{r["wall"]:>9.2f} {r["fuse_time"]:>10.2f} '
                     f'{r["fuse_time"] / max(r["wall"], 1e-9) * 100:>4.0f}% '
                     f'{r["affine_calls"]:>13,} {r["affine_work"]:>14,} {r["shift_calls"]:>12,} '
                     f'{"FAIL" if r["failures"] else "ok":>7}')
    lines.append('')
    lines.append('Per-case T is in experiments/phase0_results/baseline_tcost_{n}.txt.')
    lines.append('1.1 acceptance: after switching to closed-form, rerun this script; baseline_tcost_{n}.txt must be bit-identical (empty diff).')
    text = '\n'.join(lines)

    with open(os.path.join(OUT_DIR, 'baseline_summary.txt'), 'w') as f:
        f.write(text + '\n')
    print()
    print(text)


if __name__ == '__main__':
    main()
