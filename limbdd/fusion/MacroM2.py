# -*- coding: utf-8 -*-
"""
宏 M2 生产实现(P9 定稿形态,2025-08-22)。

理论:docs/macro_m2_theory.md §2-3。对 δ=2 对 (A,B) 与候选 h(经典 exorlink
分歧字面量族,P6a 机制),先做双侧重写 {A,B} → {A⊕h, B⊕h}(对子移动 imm,
不动点处必 ≤ 0,否则 δ=2 立即型早已被采用),再挑第三段 C 与 A′ 融合
(g_fuse > 0 且总收益 imm + g_fuse > 0),应用:

    {A, B, C} → { B⊕h, (A⊕h)⊕C }

B 侧对称孪生一并枚举。宏严格下降(XOR 不变量由 h⊕h=0 保证)⇒ 逐例 T 零回归;
时间成本 n≤7 实测 0.1~2.9 ms/例(P9),生产接入带开关(默认关闭)。

生产化要点:
  - C 预筛:gain(A′,C)>0 的必要条件是双方 dim(K) 差 ≤1(理论文档推广 #6),
    按 dim 分桶把 C 扫描从 O(m) 降为桶内;`m2_best_move(pool, n, prescreen=…)`
    可用全扫描对照(tests/test_macro_m2.py 做差分验证);
  - 轮次上限 max_rounds(默认 30,与 P9 探针口径一致);
  - XOR 不变量断言仅在 LIMBDD_FUSION_DEBUG=1 下执行(O(m·2^n),n≥8 生产关闭)。
"""
from collections import defaultdict

from limbdd.CompleteBranch import CompleteBranch, plan_from_affine
from limbdd.fusion.BranchFusion import fuse_pool_global, shift_fn, FUSION_DEBUG
from limbdd.fusion.ClosedForm import canonical_affine, fuse_closed
from config import config

TC = config['t_cost']


# ------------------------------------------------------------- P6a 机制移植 ----

def rows_of(cb, r, n):
    """一般仿射段的规范约束行:((mask, const), ...),含根掩码平移(base ^= r)。

    affine_rows() 的行掩码去掉了主元位(主元隐含),须加回 (1 << pivot)。
    """
    base, basis = canonical_affine(cb, r)
    plan = plan_from_affine(base, basis, n)
    tmp = CompleteBranch(n, synthetic=plan)
    return tuple(((1 << col) | mask, const) for col, mask, const in tmp.affine_rows())


def pair_info(ra, rb, n):
    """两段约束行集合的经典距离 δ 与候选 h 原料(全部单行分歧候选)。"""
    da = dict((m, c) for m, c in ra)
    db = dict((m, c) for m, c in rb)
    agree, opp, oneside_a, oneside_b = [], [], [], []
    for m in da:
        if m in db:
            if da[m] == db[m]:
                agree.append((m, da[m]))
            else:
                opp.append(m)
        else:
            oneside_a.append((m, da[m]))
    for m in db:
        if m not in da:
            oneside_b.append((m, db[m]))
    return len(opp) + len(oneside_a) + len(oneside_b), agree, opp, oneside_a, oneside_b


def h_candidates(agree, opp, oneside_a, oneside_b):
    """候选 h 的约束行列表:每个分歧变量一行(opp 取两侧各一种)。"""
    out = []
    for m in opp:
        out.append(agree + [(m, 0)])
        out.append(agree + [(m, 1)])
    for m, c in oneside_a:
        out.append(agree + [(m, c)])
    for m, c in oneside_b:
        out.append(agree + [(m, c)])
    return out


def cb_from_rows(rows, n):
    """约束行集合 → CompleteBranch(合成表示)。行须线性无关。"""
    pivots = {}
    for m, c in rows:
        mm, cc = m, c
        while mm:
            low = mm & (-mm)
            if low in pivots:
                mm ^= pivots[low][0]
                cc ^= pivots[low][1]
            else:
                pivots[low] = (mm, cc)
                break
        if mm == 0 and cc:
            return None          # 不可满足,非立方
    basis, base = [], 0
    pivot_cols = {v.bit_length() - 1 for v in pivots}
    for f in range(n):
        if f in pivot_cols:
            continue
        v = 1 << f
        for m, c in pivots.values():
            if (m >> f) & 1:
                v |= 1 << (m & (-m)).bit_length() - 1
        basis.append(v)
    for m, c in pivots.values():
        if c:
            base |= 1 << ((m & (-m)).bit_length() - 1)
    return CompleteBranch(n, synthetic=plan_from_affine(base, basis, n))


# ------------------------------------------------------------- 宏 M2 ---------

def seg_from(res, root, n):
    return CompleteBranch(n, synthetic=plan_from_affine(res.base ^ root, res.basis, n))


def _pool_xor(pool, n):
    x = 0
    for cb, r, _ in pool:
        x ^= shift_fn(cb.fn(), r, n)
    return x


def m2_best_move(pool, n, prescreen=True):
    """
    不动点池上的最佳宏 M2。返回 (total, idxA, idxB, k, rA, rB, rc) 或 None。

    total = imm + g_fuse > 0 为真实总收益:
      imm     = t(A)+t(B) − t(A⊕h) − t(B⊕h)(对子移动,不动点处 ≤ 0);
      g_fuse  = t(A⊕h)+t(C) − t((A⊕h)⊕C)(A′+C 融合收益)。
    prescreen=True 时 C 只扫 dim(K) 与 rA.dim 差 ≤1 的桶(理论推广 #6);
    差分正确性由 tests/test_macro_m2.py 全扫描对照验证。
    """
    aff = [canonical_affine(cb, r) for cb, r, _ in pool]
    dims = [len(basis) for _, basis in aff]
    rows = [rows_of(pool[i][0], pool[i][1], n) for i in range(len(pool))]
    if prescreen:
        buckets = defaultdict(list)
        for k, d in enumerate(dims):
            buckets[d].append(k)

        def c_candidates(d_a):
            out = []
            for d in (d_a - 1, d_a, d_a + 1):
                out.extend(buckets[d])
            return out
    else:
        def c_candidates(d_a):
            return range(len(pool))

    m = len(pool)
    best = None
    for i in range(m):
        ti = pool[i][0].t_cost()
        for j in range(i + 1, m):
            d, agree, opp, oa, ob = pair_info(rows[i], rows[j], n)
            if d != 2:
                continue
            for rows_h in h_candidates(agree, opp, oa, ob):
                ch = cb_from_rows(rows_h, n)
                if ch is None:
                    continue
                ah = canonical_affine(ch)
                r1 = fuse_closed(aff[i], ah, n)
                r2 = fuse_closed(aff[j], ah, n)
                if not (r1.ok and r2.ok and not r1.cancel and not r2.cancel):
                    continue
                imm = ti + pool[j][0].t_cost() - TC(n - r1.dim) - TC(n - r2.dim)
                for rA, rB, idxA, idxB in ((r1, r2, i, j), (r2, r1, j, i)):
                    tA = TC(n - rA.dim)
                    affA = (rA.base, rA.basis)
                    for k in c_candidates(rA.dim):
                        if k == idxA or k == idxB:
                            continue
                        rc = fuse_closed(affA, aff[k], n)
                        if not rc.ok or rc.cancel:
                            continue
                        g_fuse = tA + pool[k][0].t_cost() - TC(n - rc.dim)
                        total = imm + g_fuse
                        if total > 0 and (best is None or total > best[0]):
                            best = (total, idxA, idxB, k, rA, rB, rc)
    return best


def m2_interleaved(kept, standalone, n, max_rounds=30):
    """
    宏 M2 相间调度(生产式,P9 口径):池须已融合到不动点。

    循环:最佳宏 M2 → 应用 → fuse_pool_global 重融,至多 max_rounds 次;
    返回 (kept, standalone),结构与 fuse_pool_global 相同(发射归组语义不变)。
    每步严格下降 ⇒ 逐例 T 零回归;LIMBDD_FUSION_DEBUG=1 时逐轮断言 XOR 不变量。
    """
    pool = [(cb, r, key) for cb, r, key in kept] + [(cb, 0, None) for cb in standalone]
    if FUSION_DEBUG:
        expected_xor = _pool_xor(pool, n)
    for _ in range(max_rounds):
        mv = m2_best_move(pool, n)
        if mv is None:
            break
        total, idxA, idxB, k, rA, rB, rc = mv
        b1 = seg_from(rB, pool[idxB][1], n)
        fused = seg_from(rc, 0, n)
        pool = [e for q, e in enumerate(pool) if q not in (idxA, idxB, k)] + \
               [(b1, pool[idxB][1], pool[idxB][2]), (fused, 0, None)]
        if FUSION_DEBUG:
            assert _pool_xor(pool, n) == expected_xor, 'M2 移动破坏池 XOR 不变量'
        kept2, st2 = fuse_pool_global(pool, n)
        pool = [(cb, r, key) for cb, r, key in kept2] + [(cb, 0, None) for cb in st2]
        if FUSION_DEBUG:
            assert _pool_xor(pool, n) == expected_xor, 'M2 后重融破坏池 XOR 不变量'
    kept = [(cb, r, key) for cb, r, key in pool if key is not None]
    standalone = [cb for cb, r, key in pool if key is None]
    return kept, standalone
