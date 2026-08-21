# -*- coding: utf-8 -*-
"""
G-2.1 探针:FPRM 平移子群搜索的收益测量(阶段 3 预研)。

对 f 施加平移 x ↦ x⊕c:c 的电路实现是 X(c) 三明治(T 免费),故
  ΣT(最优 c 的 ANF 池融合) = 可实现电路的真实 T 代价。
c=0 即现状 ANF 路径 ⇒ min 组合结构性零回归。

指标:
  base     = min(A_BDD, C_{c=0})                  —— 现状生产后端
  improved = min(A_BDD, best over c 的 C 池)      —— FPRM 搜索后端
  另记录 t(deg f) 下界(G-3)与每 c 的 C 路径耗时(决定全枚举/采样预算)。

口径:复用 LimBDDOracleSynthesizer._bdd_entries(A 池)与 anf_pool + fuse_pool_global(C 池);
XOR 不变量在 n≤7 全量校验(平移后函数空间)。

输出:experiments/phase0_results/fprm_probe.txt
用法:./venv/bin/python experiments/fprm_probe.py
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
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.CompleteBranch import t_cost
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'fprm_probe.txt')

# n -> (cases, 平移集合规模;None=全枚举 2^n)
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
    """f 的 ANF 次数 = 单项式掩码的最大 popcount(G-3 下界 t(deg f) 的输入)。"""
    deg = 0
    for cb, _, _ in anf_pool(bitstr, n):
        deg = max(deg, cb.mask.bit_count())
    return deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20250821)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    lines = ['G-2.1 探针:FPRM 平移子群搜索收益测量', '']
    for n, (cases, n_shifts) in PROFILE.items():
        with open(os.path.join(ROOT, f'data/exorcism_rand_tcost_{n}.txt')) as f:
            data = [int(line.split(':')[0]) for line in f.readlines()[:cases]]

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
                    assert xor_pool(kept, standalone) == g, f'XOR 不变量破坏 n={n} c={c}'
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
            f'cases={cases} shifts={len(shifts)}/2^{n}  C 路径总耗时 {wall_c:.1f}s(每 c 均值 {per_c_ms:.1f} ms)',
            f'ΣT: base={s_base} improved={s_imp}  Δ {s_imp - s_base:+d}({(s_base - s_imp) / max(s_base, 1):.2%})',
            f'cases: better={stat["better"]} same={stat["same"]}',
            f'G-3 下界 Σ t(deg f)={s_lb}:为 base 的 {s_lb / max(s_base, 1):.2%};'
            f'余量 base→下界 {(s_base - s_lb) / max(s_base, 1):.2%},improved→下界 {(s_imp - s_lb) / max(s_imp, 1):.2%}',
            f'全枚举 2^{n} 个 c 的预估 C 路径耗时: {per_c_ms * (1 << n) / 1000:.1f}s/例',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
