"""
路径级性能基准（单起点融合）：no-reorder BDD / BDD(结点数 sifting) / ANF。

每路径报告：
  T：融合池 ΣT（单起点 fuse_pool_global，与生产一致，不发射电路）；
  m：池条目数（分解规模）；
  t_dec：分解阶段耗时（BDD：construct + try_neg/split_neg + sifting + 分片抽取；
                  ANF：Möbius 变换 + 单项式池）；
  t_tot：分解 + 融合总耗时。
prod = min(BDD, ANF)（当前生产组合），耗时 = bn + anf。

用法: ./venv/bin/python experiments/path_benchmark_fast.py [n ...]   默认 8 9 10 11
"""
import os
import random
import sys
import time

# 项目根目录（脚本位于 experiments/ 下，故上溯一级）；加入 sys.path 保证 limbdd/utils 可导入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost
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
    syn = LimBDDOracleSynthesizer(n, 0, fused_backend=True)
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
