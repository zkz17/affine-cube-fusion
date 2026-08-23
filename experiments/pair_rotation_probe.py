# -*- coding: utf-8 -*-
"""
P2 探针(方案 B 入场门槛):不动点段对旋转 vs GL 搜索的兑现率。

定理 B(docs/fprm_proxy_and_pair_rotation.md §2.1):可融合对 (A,B)(d=A⊕B ≠ 0),
对任意 w ∈ K_d:{A,B} → {A∘τ_w, B∘τ_w} 保持池 XOR 与各段 T 精确不变,
且改变与第三方段的陪集位置(融合关系真实改变)。

口径:
  (a) FPRM 最优平移的融合不动点(现状 F 路径终值);
  (b) 在 (a) 的不动点池上做对旋转 + 全池重融的局部搜索(T 单调下降,严格改进);
  (c) GL 搜索(生产 gl_search,确定性)。

输出:ΣT(a/b/c)、兑现率 = (a−b)/(a−c)、移动/墙时统计。
决策规则:兑现率 ≥ 30% 且墙时 ≤ GL/10 ⇒ 落地;15%~30% 交裁决;<15% 放弃。
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from limbdd.CompleteBranch import CompleteBranch, plan_from_affine
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from limbdd.fusion.ClosedForm import canonical_affine, fuse_closed
from limbdd.fusion.GLSearch import fprm_best, gl_search

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'pair_rotation_probe.txt')
PROFILE = {5: 20, 6: 20, 7: 20}


def w_candidates(basis, cap=12):
    """K_d 的确定性 w 候选:规范基向量 + 全部两两和(截断)。"""
    out = list(basis)
    k = len(basis)
    for i in range(k):
        for j in range(i + 1, k):
            out.append(basis[i] ^ basis[j])
    return out[:cap]


def rotate_pair(pool, i, j, w, n):
    """第 i/j 段在有效函数空间上同时平移 w(合成表示,控制数精确不变)。"""
    out = list(pool)
    for idx in (i, j):
        cb, r, key = pool[idx]
        base, basis = canonical_affine(cb, r)
        new_cb = CompleteBranch(n, synthetic=plan_from_affine(base ^ w, basis, n))
        out[idx] = (new_cb, r, key)
    return out


def pair_rotation_search(pool0, n, max_moves=200, verbose=False):
    """
    对旋转 + 全池重融的确定性局部搜索(探针口径,非增量)。

    pool0: 融合不动点的条目列表 [(cb, r, key)]。
    每步枚举可融合且 d≠0 的对 × w 候选,取**第一个**严格改进(重融后 ΣT 下降),
    接受后新池 = 重融输出的不动点;无改进即停。T 严格单调下降 ⇒ 自然终止。
    返回 (ΣT, moves, wall, 统计)。
    """
    cost = pool_t_cost(*fuse_pool_global(pool0, n))
    pool = [(cb, r, key) for cb, r, key in pool0]
    moved = 0
    stat = Counter()
    t0 = time.perf_counter()
    while moved < max_moves:
        aff = [canonical_affine(cb, r) for cb, r, _ in pool]
        m = len(pool)
        found = None
        for i in range(m):
            for j in range(i + 1, m):
                res = fuse_closed(aff[i], aff[j], n)
                if not res.ok or res.cancel:
                    continue
                stat['fusable_pairs'] += 1
                for w in w_candidates(res.basis):
                    new_pool = rotate_pair(pool, i, j, w, n)
                    kept, standalone = fuse_pool_global(new_pool, n)
                    c = pool_t_cost(kept, standalone)
                    if c < cost:
                        found = (c, kept, standalone, w)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            break
        cost = found[0]
        pool = [(cb, r, key) for cb, r, key in found[1]] + [(cb, 0, None) for cb in found[2]]
        moved += 1
        stat['moves'] += 1
    return cost, moved, time.perf_counter() - t0, stat


def main():
    lines = ['P2 探针:不动点段对旋转 vs GL 搜索的兑现率', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        s_a = s_b = s_c = 0
        t_b = t_c = 0.0
        moves_sum = pairs_sum = 0
        better_b = better_c = 0
        for fn in data:
            vF, cF = fprm_best(fn, n)
            pool0 = fuse_pool_global(anf_pool(shift_fn(fn, cF, n), n), n)
            a = pool_t_cost(*pool0)
            entries0 = [(cb, r, key) for cb, r, key in pool0[0]] + \
                       [(cb, 0, None) for cb in pool0[1]]

            t0 = time.perf_counter()
            b, moves, tb, stat = pair_rotation_search(entries0, n)
            t_b += time.perf_counter() - t0

            t0 = time.perf_counter()
            kept_g, st_g, _, _ = gl_search(fn, n)
            t_c += time.perf_counter() - t0
            c = pool_t_cost(kept_g, st_g)

            s_a += a
            s_b += b
            s_c += c
            moves_sum += moves
            pairs_sum += stat['fusable_pairs']
            better_b += b < a
            better_c += c < a

        msg = [
            f'====== n = {n}({cases} cases)======',
            f'ΣT: (a) FPRM 不动点={s_a}  (b) 对旋转={s_b}  (c) GL={s_c}',
            f'相对 (a): b {(s_b/s_a-1):+.2%}  c {(s_c/s_a-1):+.2%}',
            f'兑现率 = (a−b)/(a−c) = {(s_a-s_b)/max(s_a-s_c,1):.2%}  (b 更优 {better_b}/{cases}, c 更优 {better_c}/{cases})',
            f'旋转: 平均移动 {moves_sum/cases:.1f} 次, 每轮可融对均值 {pairs_sum/max(moves_sum,1):.1f}, '
            f'墙时 {t_b/cases:.2f}s/例 vs GL {t_c/cases:.2f}s/例',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n结果已写入 {OUT}')


if __name__ == '__main__':
    main()
