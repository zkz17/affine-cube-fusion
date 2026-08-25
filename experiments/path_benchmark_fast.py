"""
Path-level performance benchmark (single-start fusion): no-reorder BDD / BDD (node-count sifting) / ANF.

Report per path:
  T: fused-pool ΣT (single-start fuse_pool_global, matches production, no circuit emission);
  m: pool entry count (decomposition size);
  t_dec: decomposition-stage time (BDD: construct + try_neg/split_neg + sifting + piece extraction;
                  ANF: Möbius transform + monomial pool);
  t_tot: decomposition + fusion total time.
prod = min(BDD, ANF) (current production combination), time = bn + anf.

Usage: ./venv/bin/python experiments/path_benchmark_fast.py [n ...]   default 8 9 10 11
"""
import os
import random
import sys
import time

# Project root (the script lives under experiments/, so go up one level); add to sys.path so limbdd/utils can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost
from utils.TruthTable import TruthTable

CASES = {8: 5, 9: 5, 10: 3, 11: 2}


def bdd_path(syn, tt, n, reorder):
    syn.clear()
    syn.reorder = reorder
    t0 = time.perf_counter()
    entries = syn._bdd_entries(tt, None)
    t_dec = time.perf_counter() - t0
    kept, st = fuse_pool_global(entries, n)
    t_tot = time.perf_counter() - t0
    return pool_t_cost(kept, st), len(entries), t_dec, t_tot


def anf_path(tt, n):
    t0 = time.perf_counter()
    entries = anf_pool(tt.bitstr, n)
    t_dec = time.perf_counter() - t0
    kept, st = fuse_pool_global(entries, n)
    t_tot = time.perf_counter() - t0
    return pool_t_cost(kept, st), len(entries), t_dec, t_tot


def run(n, cnt):
    syn = OracleSynthesizer(n, 0, fused_backend=True)
    keys = ('nr', 'bn', 'anf')
    sums = {k: [0, 0.0] for k in keys}  # T, t_tot
    print(f'=== n={n} ({cnt} cases) ===', flush=True)
    for s in range(cnt):
        rng = random.Random(10 ** 9 + n * 1000 + s)
        tt = TruthTable(rng.getrandbits(1 << n), n)
        row = {
            'nr': bdd_path(syn, tt, n, reorder=False),
            'bn': bdd_path(syn, tt, n, reorder=True),
            'anf': anf_path(tt, n),
        }
        prod_t = min(row['bn'][0], row['anf'][0])
        prod_time = row['bn'][3] + row['anf'][3]
        parts = [f'case {s} tt={tt.bitstr:0{n}x}']
        for k in keys:
            T, m, t_dec, t_tot = row[k]
            sums[k][0] += T
            sums[k][1] += t_tot
            parts.append(f'{k}:T={T}/m={m}/t={t_dec:.1f}+{t_tot - t_dec:.1f}s')
        parts.append(f'prod:T={prod_t}/t={prod_time:.1f}s')
        print('  ' + ' | '.join(parts), flush=True)
    print('  SUM ' + ' | '.join(f'{k}:T={sums[k][0]}/t={sums[k][1]:.1f}s' for k in keys), flush=True)


def main():
    ns = [int(a) for a in sys.argv[1:]] or [8, 9, 10, 11]
    for n in ns:
        run(n, CASES[n])


if __name__ == '__main__':
    main()
