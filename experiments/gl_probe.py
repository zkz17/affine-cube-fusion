# -*- coding: utf-8 -*-
"""
G-2.2 探针:GL(n,2) 线性子群搜索的收益测量(阶段 3 预研,仅 n ≤ 7)。

原理(Optimization_roadmap.md §3.3):融合引擎对 AGL(n,2) 等变,但 ANF/Möbius
分解绑定标准坐标基 ⇒ 必须在换坐标后**重新生成池**才有新起点。线性变换 A 的
电路是 CNOT 网络(**T 免费**,代价在 CX:wrapper = 2 × #CNOT)。

变体:
  c0 : 恒等(现状 ANF 路径)
  cF : FPRM 平移最优(G-2.1)
  L  : GL 采样 + CNOT 邻域爬山(c=0)
  LF : 先平移 cF 再 GL 采样 + 爬山(仿射组合)
指标:融合 ΣT、最佳映射的 wrapper CX(2×#CNOT)、评估次数。

输出:experiments/phase0_results/gl_probe.txt
用法:./venv/bin/python experiments/gl_probe.py
"""
import argparse
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'gl_probe.txt')

# n -> (cases, GL 采样数, 爬山步数上限)
PROFILE = {
    5: (30, 32, 10),
    6: (12, 32, 8),
    7: (4, 24, 6),
}


def apply_linear(fn, A, n):
    """g(x) = f(Ax)。A 以行掩码表示:(Ax)_i = parity(A[i] & x)。"""
    out = 0
    for x in range(1 << n):
        y = 0
        for i, r in enumerate(A):
            if (r & x).bit_count() & 1:
                y |= 1 << i
        if (fn >> y) & 1:
            out |= 1 << x
    return out


def compose_transvection_right(A, c, t, n):
    """A ∘ E,E = CNOT(c → t) 作用于输入:行 i 在 bit t 置位时异或 bit c。"""
    out = A[:]
    for i in range(n):
        if (out[i] >> t) & 1:
            out[i] ^= 1 << c
    return out


def rand_gl(n, k, rng):
    """k 个随机转vections 之积(必可逆),返回 (A, cnot 列表)。"""
    A = [(1 << i) for i in range(n)]
    cnots = []
    for _ in range(k):
        c = rng.randrange(n)
        t = rng.randrange(n)
        if c == t:
            continue
        A = compose_transvection_right(A, c, t, n)
        cnots.append((c, t))
    return A, cnots


def make_cost(fn, n, cache):
    def cost(A):
        key = tuple(A)
        if key not in cache:
            kept, standalone = fuse_pool_global(anf_pool(apply_linear(fn, A, n), n), n)
            cache[key] = pool_t_cost(kept, standalone)
        return cache[key]
    return cost


def climb(A0, cnots0, cost, n, max_steps):
    """CNOT 邻域最陡上升(严格改进);返回 (A, cnots, value)。"""
    A, cnots = A0[:], list(cnots0)
    cur = cost(tuple(A))
    while True:
        best, best_A, best_move = cur, None, None
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                A2 = compose_transvection_right(A, c, t, n)
                v = cost(tuple(A2))
                if v < best:
                    best, best_A, best_move = v, A2, (c, t)
        if best_A is None:
            break
        A = best_A
        cnots.append(best_move)
        cur = best
        if max_steps and len(cnots) - len(cnots0) >= max_steps:
            break
    return A, cnots, cur


def fprm_best(fn, n):
    """平移子群全枚举的最优 c(与生产 _fprm_entries 的 n≤8 口径一致)。"""
    best, best_c = None, 0
    for c in range(1 << n):
        g = shift_fn(fn, c, n)
        kept, standalone = fuse_pool_global(anf_pool(g, n), n)
        v = pool_t_cost(kept, standalone)
        if best is None or v < best:
            best, best_c = v, c
    return best, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20250822)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    lines = ['G-2.2 探针:GL(n,2) 线性子群搜索收益测量(仅 n ≤ 7)', '']
    for n, (cases, n_sample, max_steps) in PROFILE.items():
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            data = [int(line.rstrip('\n')) for line in f.readlines()[:cases]]

        stat = Counter()
        s_c0 = s_cF = s_L = s_LF = 0
        wrap_cx = 0
        t0 = time.perf_counter()
        for fn in data:
            cache = {}
            cost = make_cost(fn, n, cache)

            c0 = cost(tuple([1 << i for i in range(n)]))
            vF, cF = fprm_best(fn, n)   # vF=代价, cF=平移下标(供 LF 变体使用)

            # ---- 变体 L:c=0,GL 采样 + 爬山 ----
            best_A, best_cnots, best_v = None, None, None
            for _ in range(n_sample):
                k = rng.choice([n // 2, n, 3 * n // 2])
                A, cnots = rand_gl(n, max(k, 1), rng)
                A, cnots, v = climb(A, cnots, cost, n, max_steps)
                if best_v is None or v < best_v:
                    best_A, best_cnots, best_v = A, cnots, v
            tL = best_v

            # ---- 变体 LF:先平移 cF,再 GL ----
            fn_pre = shift_fn(fn, cF, n)
            cache2 = {}
            cost2 = make_cost(fn_pre, n, cache2)
            best_v2, best_cnots2 = None, None
            for _ in range(n_sample):
                k = rng.choice([n // 2, n, 3 * n // 2])
                A, cnots = rand_gl(n, max(k, 1), rng)
                A, cnots, v = climb(A, cnots, cost2, n, max_steps)
                if best_v2 is None or v < best_v2:
                    best_v2, best_cnots2 = v, cnots
            tLF = best_v2

            s_c0 += c0
            s_cF += vF
            s_L += tL
            s_LF += tLF
            wrap_cx += 2 * len(best_cnots) + 2 * len(best_cnots2)
            stat['L_better_than_cF'] += tL < vF
            stat['LF_better_than_cF'] += tLF < vF
            stat['LF_better_than_L'] += tLF < tL
            stat['evals'] += len(cache) + len(cache2)

        wall = time.perf_counter() - t0
        msg = [
            f'====== n = {n}({cases} cases,{wall:.1f}s)======',
            f'ΣT: c0={s_c0} cF={s_cF} L={s_L} LF={s_LF}',
            f'相对 c0: cF {(s_cF/s_c0-1):+.2%}  L {(s_L/s_c0-1):+.2%}  LF {(s_LF/s_c0-1):+.2%}',
            f'相对 cF: L {(s_L/s_cF-1):+.2%}  LF {(s_LF/s_cF-1):+.2%}',
            f'L 优于 cF: {stat["L_better_than_cF"]}/{cases};LF 优于 cF: {stat["LF_better_than_cF"]}/{cases};'
            f'LF 优于 L: {stat["LF_better_than_L"]}/{cases}',
            f'最优映射 wrapper CX 合计: {wrap_cx}(平均 {wrap_cx/max(cases,1):.1f}/例;'
            f'仅对获胜者计,若纳入生产需计入 CX 口径)',
            f'平均评估次数: {stat["evals"]/max(cases,1):.0f}/例',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
