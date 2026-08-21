# -*- coding: utf-8 -*-
"""
闭式融合判定与构造(Optimization_roadmap.md 的 T-1 实现)。

**本模块尚未接入生产路径**:它与 `limbdd/fusion/BranchFusion.py` 的 fn 路径并存,
等价性由 `tests/test_closed_form.py` 全量交叉验证(阶段 0.1)。确认零分歧后,
阶段 1.1 再把 `fuse_pool` / `fuse_pool_global` 切换到本模块。

数学依据
--------
- `CompleteBranch_algebra.md` §2.2 定理 2:两段可融合 ⟺ supp(f_A⊕f_B) 为仿射子空间,
  且**充要地**分为 C1–C4 四情形。故结构判定即权威判定。
- `CompleteBranch_algebra.md` §3.7:各情形融合结果 (e, B) 的代数闭式。
- `CompleteBranch_algebra.md` §1.4:跨片平移只平移 base,方向子空间 K 不变。

复杂度对比(n 为变量数,生产路径的 2^n 全部消失)
------------------------------------------------
| 步骤       | fn 路径                          | 本模块          |
|-----------|----------------------------------|----------------|
| 可融合判定 | is_affine_cube  O(2^n)+O(|S|^2)  | O(n^2) 位运算   |
| 结果维数   | 同上                              | O(1)           |
| 平移       | shift_fn        O(2^n)           | base ^= r  O(1)|
| 结果构造   | affine_from_fn  O(2^n)           | O(n^2)         |

规范形约定(重要)
----------------
所有 `basis` 必须是 `reduce_basis` 的产出(最低置位互异、按主元升序),否则 `in_span`
不成立。`CompleteBranch.affine().basis` **不保证**该形式——它由 RREF 的自由列构造,
多个基向量可能共享同一最低位(`BranchFusion.four_case_kind` 亦为此显式重新归约)。
因此一切入口都必须经 `canonical_affine()`。
"""
from collections import namedtuple

from limbdd.CompleteBranch import in_span, reduce_basis

# ok     : 是否可融合(含抵消)
# kind   : 1..4,与 BranchFusion.four_case_kind 编号一致(1=C1+C2, 2=C4, 3/4=C3 及其对称)
# cancel : d == 0,两段相互抵消
# dim    : 融合结果 F 的维数;抵消时为 -1(与 is_affine_cube 的空支撑约定一致)
# base/basis : F = base + span(basis);不可融合或抵消时无意义
ClosedFusion = namedtuple('ClosedFusion', 'ok kind cancel dim base basis')

NOT_FUSABLE = ClosedFusion(False, None, False, 0, 0, ())


# ---------------------------------------------------------------- 规范化 ----

def canonical_affine(cb, root_mask=0):
    """
    取分支的规范仿射表示 (base, basis),并按片根掩码 root_mask 做平移。

    平移依据(代数文档 §1.4):g(x) = f(x⊕r) ⇒ supp(g) = supp(f) ⊕ r,
    即 base ^= r、K 不变。这替代了 fn 路径里 O(2^n) 的 `shift_fn`。
    """
    aff = cb.affine()
    return aff.base ^ root_mask, reduce_basis(aff.basis)


# ---------------------------------------------------------------- 子空间 ----

def _subspace_leq(small, big):
    """span(small) ⊆ span(big)。两者均须为规范基。"""
    return all(in_span(v, big) for v in small)


def _subspace_eq(ba, bb):
    """
    span(ba) == span(bb)。

    **不能**用元组相等代替:`reduce_basis` 产出的是行阶梯形(各向量最低位互异)而非
    RREF,而行阶梯形一般不唯一(例:span{e0,e1} 的 (0b01,0b10) 与 (0b11,0b10) 都合法)。
    这里用「同维 + 单向包含」判定,不依赖任何规范性假设。
    """
    return len(ba) == len(bb) and _subspace_leq(ba, bb)


def _pick_outside(basis, sub_basis):
    """取 basis 中一个不属于 span(sub_basis) 的向量(调用方须保证存在)。"""
    for v in basis:
        if not in_span(v, sub_basis):
            return v
    raise AssertionError('span(basis) ⊆ span(sub_basis):调用方前置条件被违反')


def _insert(piv, vec, acc, w_out):
    """
    带来源标记的高斯消元插入。

    不变量:每个 pivot 项 (vec, acc) 满足 acc ∈ span(A) 且 vec⊕acc ∈ span(B)。
    当 vec 被消为 0 时,acc 同时属于两者 ⇒ acc ∈ A∩B,收集进 w_out。
    """
    while vec:
        low = vec & (-vec)
        hit = piv.get(low)
        if hit is None:
            piv[low] = (vec, acc)
            return
        vec ^= hit[0]
        acc ^= hit[1]
    if acc:
        w_out.append(acc)


def _build_joint(basis_a, basis_b):
    """
    构造 span(A)+span(B) 的 pivot 表与 A∩B 的规范基。

    A 的向量以 acc=v 插入(vec⊕acc = 0 ∈ span(B));B 的向量以 acc=0 插入。
    返回 (piv, w_basis)。
    """
    piv, w = {}, []
    for v in basis_a:
        _insert(piv, v, v, w)
    for v in basis_b:
        _insert(piv, v, 0, w)
    return piv, reduce_basis(w)


def _solve_split(d, piv):
    """
    求 x ∈ span(A) 使 d⊕x ∈ span(B);d ∉ span(A)+span(B) 时返回 None。

    用于 C4 取交点:t = a_A ⊕ x 同时落在 S_A 与 S_B 内。
    """
    x, r = 0, d
    while r:
        low = r & (-r)
        hit = piv.get(low)
        if hit is None:
            return None
        r ^= hit[0]
        x ^= hit[1]
    return x


# ---------------------------------------------------------------- 判定 ----

def fuse_closed(aff_a, aff_b, n):
    """
    闭式融合判定与构造。aff_* 为 canonical_affine() 的产出 (base, basis)。

    返回 ClosedFusion。四情形编号与 BranchFusion.four_case_kind 一致。
    """
    base_a, ka = aff_a
    base_b, kb = aff_b
    d = base_a ^ base_b

    # ---- kind 1:K_A = K_B → C1 抵消 / C2 融合 ----
    if _subspace_eq(ka, kb):
        if in_span(d, ka):
            return ClosedFusion(True, 1, True, -1, 0, ())
        basis = reduce_basis(ka + (d,))
        assert len(basis) == len(ka) + 1, 'C2:d ∉ K 时维数必须 +1'
        return ClosedFusion(True, 1, False, len(ka) + 1, base_a, basis)

    # ---- kind 3:K_A ⊊ K_B 余维 1 且同大陪集 → F = S_B ∖ S_A,方向 K_A ----
    if len(kb) == len(ka) + 1 and _subspace_leq(ka, kb) and in_span(d, kb):
        w = _pick_outside(kb, ka)
        return ClosedFusion(True, 3, False, len(ka), base_a ^ w, ka)

    # ---- kind 4:K_B ⊊ K_A 余维 1 且同大陪集 → F = S_A ∖ S_B,方向 K_B ----
    if len(ka) == len(kb) + 1 and _subspace_leq(kb, ka) and in_span(d, ka):
        w = _pick_outside(ka, kb)
        return ClosedFusion(True, 4, False, len(kb), base_b ^ w, kb)

    # ---- kind 2:同维、交余维 1、陪集相交 → C4 ----
    if len(ka) == len(kb):
        piv, wb = _build_joint(ka, kb)
        if len(wb) == len(ka) - 1:
            x = _solve_split(d, piv)
            if x is not None:
                t = base_a ^ x
                u1 = _pick_outside(ka, wb)
                u2 = _pick_outside(kb, wb)
                basis = reduce_basis(wb + (u1 ^ u2,))
                assert len(basis) == len(ka), 'C4:结果维数必须等于 dim K_A'
                return ClosedFusion(True, 2, False, len(ka), t ^ u1, basis)

    return NOT_FUSABLE


def closed_gain(cb_a, cb_b, res, t_cost_fn, n):
    """
    闭式收益(与 BranchFusion 的 gain 口径一致)。

    抵消:收益 = 两段代价之和(且恒接受,即使为 0 也减少段数与门数);
    融合:收益 = t(k_A) + t(k_B) − t(n − dim F)。
    """
    if not res.ok:
        return None
    base = cb_a.t_cost() + cb_b.t_cost()
    if res.cancel:
        return base
    return base - t_cost_fn(n - res.dim)
