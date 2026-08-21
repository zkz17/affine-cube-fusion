# -*- coding: utf-8 -*-
"""
成对完备性猜想（CompleteBranch_algebra.md §4 开问题）的随机反例搜索。

猜想（k=3 基例）：若三个仿射立方（分支段）的 XOR 是仿射立方 D，
则必存在一对 (i,j) 其 XOR 为仿射立方（C1–C4，即 2-to-1 可融合）。

搜索：随机生成三元组 CompleteBranch（随机 SyntheticPlan），
筛出 d = f1^f2^f3 为仿射立方、且三对两两均不可融合（is_fusable=False）者 = 反例。

用法：python counterexample_search.py n examples [seed]
输出：stdout + experiments/phase0_results/cex_search_{n}_{seed}.txt
"""
import argparse
import os
import random
import sys
import time

# 项目根目录（脚本位于 experiments/ 下，故上溯一级）；加入 sys.path 保证 limbdd/utils 可导入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENTS = os.path.join(ROOT, 'experiments')

from limbdd.CompleteBranch import CompleteBranch, SyntheticPlan, is_affine_cube
from limbdd.fusion.BranchFusion import is_fusable


def random_cube(n, rng):
    """随机仿射立方：随机 X 平移 + 随机 CNOT 链 + 随机控制子集/极性。"""
    x_trans = rng.getrandbits(n) & ((1 << n) - 1)
    cnot_ops = []
    for _ in range(rng.randrange(0, n + 1)):
        c = rng.randrange(n)
        t = rng.randrange(n)
        if c != t:
            cnot_ops.append((c, t))
    controls = tuple((v, rng.randrange(2)) for v in range(n) if rng.randrange(2))
    return CompleteBranch(n, synthetic=SyntheticPlan(x_trans, tuple(cnot_ops), controls))


def search(n, examples, seed):
    rng = random.Random(seed)
    total = 0          # 三元组总数
    cand = 0           # d 为仿射立方的候选数
    cex = 0            # 反例数（候选且三对均不可融合）
    dim_hist = {}
    t0 = time.time()
    lines = [f'# 成对完备性反例搜索  n={n}  examples={examples}  seed={seed}']

    for i in range(examples):
        a = random_cube(n, rng)
        b = random_cube(n, rng)
        c = random_cube(n, rng)
        d = a.fn() ^ b.fn() ^ c.fn()
        total += 1
        ok, dim = is_affine_cube(d, n)
        if not ok:
            continue
        cand += 1
        dim_hist[dim] = dim_hist.get(dim, 0) + 1
        if is_fusable(a, b) or is_fusable(a, c) or is_fusable(b, c):
            continue
        cex += 1
        pab = is_affine_cube(a.fn() ^ b.fn(), n)
        pac = is_affine_cube(a.fn() ^ c.fn(), n)
        pbc = is_affine_cube(b.fn() ^ c.fn(), n)
        msg = (f'COUNTEREXAMPLE #{cex} at example {i}:\n'
               f'  A = {a.synthetic!r}\n  B = {b.synthetic!r}\n  C = {c.synthetic!r}\n'
               f'  D dim = {dim}\n'
               f'  pair (affine?, dim): AB=({pab[0]}, {pab[1]}), AC=({pac[0]}, {pac[1]}), BC=({pbc[0]}, {pbc[1]})')
        print(msg)
        lines.append(msg)

    lines += [
        '',
        f'# 统计: 总数 {total}, d-仿射候选 {cand} ({cand/max(1,total):.2%}), 反例 {cex}',
        f'# 候选 dim 分布: {dict(sorted(dim_hist.items()))}',
        f'# 耗时 {time.time()-t0:.1f}s',
    ]
    print('\n'.join(lines))
    out_dir = os.path.join(EXPERIMENTS, 'phase0_results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'cex_search_{n}_{seed}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[已写入 experiments/phase0_results/cex_search_{n}_{seed}.txt]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('n', type=int)
    ap.add_argument('examples', type=int)
    ap.add_argument('seed', type=int, nargs='?', default=20250815)
    args = ap.parse_args()
    search(args.n, args.examples, args.seed)


if __name__ == '__main__':
    main()
