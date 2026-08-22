# -*- coding: utf-8 -*-
"""
G-2.2:确定性 GL(n,2) 局部搜索(生产路径的**可选**开关,默认关闭)。

设计(gl_det_probe 实测定稿,去随机化):
  - 基线:FPRM 最优平移 cF(含 c=0;n≤8 全枚举,n≥9 双起点平移爬山);
  - 起点集:{恒等 on f, 恒等 on f∘τ_cF} ∪ {全部单 CNOT E_ij on f∘τ_cF},
    共 2 + n(n−1) 个,**无任何随机性**;
  - 爬山:CNOT 邻域最陡上升、严格改进、步数上限 max_steps;
  - 候选含 cF 本身(零 wrapper);T 并列时优先零 wrapper ⇒ 结果 ≤ FPRM 最优
    (支配链 G ≤ F ≤ C 保持),且 CX 不因并列而多付。

代价构成(n=7 实测):~2400 次评估 × 2.55ms,其中 **94% 是 fuse_pool_global**
(每次换坐标都要重新 Möbius + 融合,无法复用)——这是开关默认关闭的原因。

返回 (kept, standalone, shift_c, cnots):获胜池是 g(x) = f(Ax ⊕ c) 的池,
发射三明治 = X(c) · CNOT(A⁻¹) · [段] · CNOT(A) · X(c)(T 免费,CNOT 计入 CX)。
"""
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn


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


def make_cost(fn, n, cache):
    def cost(A):
        key = tuple(A)
        if key not in cache:
            kept, standalone = fuse_pool_global(anf_pool(apply_linear(fn, A, n), n), n)
            cache[key] = pool_t_cost(kept, standalone)
        return cache[key]
    return cost


def climb_best(A0, cnots0, cost, n, max_steps):
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
    """FPRM 最优平移(与生产 _fprm_entries 的预算口径完全一致)。返回 (代价, c)。"""
    full = 1 << n
    if full <= 256:
        shifts = range(full)
        best, best_c = None, 0
        for c in shifts:
            g = shift_fn(fn, c, n)
            kept, standalone = fuse_pool_global(anf_pool(g, n), n)
            v = pool_t_cost(kept, standalone)
            if best is None or v < best:
                best, best_c = v, c
        return best, best_c
    # n ≥ 9:双起点平移爬山(与生产 _fprm_entries 一致,确定性)
    cache = {}

    def eval_c(c):
        if c not in cache:
            g = shift_fn(fn, c, n)
            kept, standalone = fuse_pool_global(anf_pool(g, n), n)
            cache[c] = pool_t_cost(kept, standalone)
        return cache[c]

    for start in (0, full - 1):
        cur = start
        while True:
            b, bc = eval_c(cur), None
            for j in range(n):
                nb = cur ^ (1 << j)
                v = eval_c(nb)
                if v < b:
                    b, bc = v, nb
            if bc is None:
                break
            cur = bc
    best_c = min(cache, key=lambda c: cache[c])
    return cache[best_c], best_c


def gl_search(fn, n, max_steps=3):
    """
    确定性 GL 搜索(可选开关激活时调用)。

    返回 (kept, standalone, shift_c, cnots):
      shift_c : 获胜候选的平移(FPRM 基线);
      cnots   : 获胜线性映射 A 的 CNOT 分解(生成顺序,A = E1∘…∘Ek ⇒
                A⁻¹ 按 cnots 正序施加、A 按逆序施加)。
    """
    vF, cF = fprm_best(fn, n)
    ident = [1 << i for i in range(n)]
    best = (vF, cF, ident, [])          # (cost, shift_c, A, cnots)

    def consider(c, A, cnots, v):
        nonlocal best
        if v < best[0] or (v == best[0] and len(cnots) < len(best[3])):
            best = (v, c, list(A), list(cnots))

    cache = {}
    cost = make_cost(fn, n, cache)
    A, cnots, v = climb_best(ident, [], cost, n, max_steps)
    consider(0, A, cnots, v)

    fn_pre = shift_fn(fn, cF, n)
    cache2 = {}
    cost2 = make_cost(fn_pre, n, cache2)
    A, cnots, v = climb_best(ident, [], cost2, n, max_steps)
    consider(cF, A, cnots, v)
    for c in range(n):
        for t in range(n):
            if c == t:
                continue
            A0 = compose_transvection_right(ident, c, t, n)
            A, cnots, v = climb_best(A0, [(c, t)], cost2, n, max_steps)
            consider(cF, A, cnots, v)

    _, shift_c, A, cnots = best
    g = apply_linear(shift_fn(fn, shift_c, n) if shift_c else fn, A, n)
    kept, standalone = fuse_pool_global(anf_pool(g, n), n)
    return kept, standalone, shift_c, cnots
