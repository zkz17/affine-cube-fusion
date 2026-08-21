# -*- coding: utf-8 -*-
"""
Phase 3：分支融合原语。

数学依据：CompleteBranch_algebra.md（定理 2 四情形、§3.6 RREF 规范构造）；
工程规范：CompleteBranch_fusion_plan.md §2.4 / §7.9。

- is_fusable(a, b)   ：权威判定（d = f_A⊕f_B 为仿射立方，d==0 亦融合=抵消）；
- four_case_kind(a, b)：定理 2 的四情形结构分类（测试与桶预筛用）；
- fuse_pair(a, b)    ：构造融合段（SyntheticPlan）；抵消返回 (None, True)；
- fuse_pool(branches)：贪心融合池（XOR 不变量、段数单调减）。
"""
import collections

from limbdd.CompleteBranch import (AffineInfo, CompleteBranch, SyntheticPlan, t_cost,
                                   affine_from_fn, is_affine_cube, plan_from_affine,
                                   in_span, reduce_basis)


def is_fusable(a, b):
    """权威可融合判定：d = f_A⊕f_B 的支撑为仿射子空间（含 d==0 的抵消）。"""
    assert isinstance(a, CompleteBranch) and isinstance(b, CompleteBranch) and a.n == b.n
    d = a.fn() ^ b.fn()
    if d == 0:
        return True
    ok, _ = is_affine_cube(d, a.n)
    return ok


def four_case_kind(a, b):
    """定理 2 的四情形结构分类（CompleteBranch_algebra.md §2.2）。返回 1..4 或 None。"""
    # 先归约为主元升序的规范基（direction_key 的形式不保证该序），再作成员判定
    ka, kb = reduce_basis(a.direction_key()), reduce_basis(b.direction_key())
    ba, bb = a.affine().base, b.affine().base
    if ka == kb:
        return 1
    if len(kb) == len(ka) + 1 and all(in_span(v, kb) for v in ka) and in_span(ba ^ bb, kb):
        return 3  # K_A ⊂ K_B 余维 1 且同大陪集
    if len(ka) == len(kb) + 1 and all(in_span(v, ka) for v in kb) and in_span(ba ^ bb, ka):
        return 4  # K_B ⊂ K_A 余维 1 且同大陪集
    if len(ka) == len(kb):
        red = reduce_basis(ka + kb)
        inter_dim = 2 * len(ka) - len(red)
        if inter_dim == len(ka) - 1 and in_span(ba ^ bb, red):
            return 2  # 同维、交余维 1、陪集相交
    return None


def fuse_pair(a, b):
    """
    构造融合段。返回 (fused, cancelled)：
      cancelled=True  → f_A⊕f_B == 0，两者应从池中移除；
      fused 非 None   → 替换两者的合成段（规范 SyntheticPlan）；
      (None, False)   → 不可融合。
    """
    if not isinstance(a, CompleteBranch) or not isinstance(b, CompleteBranch) or a.n != b.n:
        return None, False
    d = a.fn() ^ b.fn()
    if d == 0:
        return None, True
    ok, _ = is_affine_cube(d, a.n)
    if not ok:
        return None, False
    base, basis = affine_from_fn(d, a.n)
    plan = plan_from_affine(base, basis, a.n)
    fused = CompleteBranch(a.n, synthetic=plan, origin=a.origin)
    assert fused.fn() == d, f'融合契约失败: {a!r} ⊕ {b!r}'
    return fused, False


def fuse_pool(branches, max_rounds=None):
    """
    贪心融合池：XOR 不变量、段数单调减，返回融合后的新列表（不修改输入）。

    策略：每轮收集全部正收益候选（抵消 / d 仿射融合），按 gain 降序贪心
    匹配（每分支至多用一次），融合段重入池；无正收益候选时停止。
    """
    if not branches:
        return []
    n = branches[0].n
    assert all(isinstance(b, CompleteBranch) and b.n == n for b in branches)

    pool = list(branches)
    if max_rounds is None:
        max_rounds = len(pool) + 1
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        cands = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                d = pool[i].fn() ^ pool[j].fn()
                if d == 0:
                    # 抵消恒有益（即使 T 收益为 0 也减少段数与门数），始终入候选
                    cands.append((pool[i].t_cost() + pool[j].t_cost(), i, j, True))
                    continue
                ok, k = is_affine_cube(d, n)
                if not ok:
                    continue
                gain = pool[i].t_cost() + pool[j].t_cost() - t_cost(n - k)
                if gain > 0:
                    cands.append((gain, i, j, False))
        if not cands:
            break
        cands.sort(reverse=True)
        used = set()
        fused_list = []
        for gain, i, j, cancel in cands:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            if not cancel:
                fused, _ = fuse_pair(pool[i], pool[j])
                assert fused is not None
                fused_list.append(fused)
        pool = [b for idx, b in enumerate(pool) if idx not in used] + fused_list
    return pool


def shift_fn(fn, r, n):
    """平移后的函数 g(x) = fn(x ⊕ r)（片根三明治 X(r) 使片内分支在平移输入上生效）。"""
    out = 0
    for x in range(1 << n):
        if (fn >> (x ^ r)) & 1:
            out |= 1 << x
    return out


def fuse_pool_global(entries, n, max_rounds=None):
    """
    跨片全局融合池（Phase 5）。

    entries: list[(CompleteBranch, root_mask, key)]，root_mask 为所属片的根边 Lim
             绝对掩码，key 为片身份（发射归组用；融合产物 key=None）。
    返回 (kept, standalone)：
      kept       : list[(CompleteBranch, root_mask, key)]，须在对应片（按 key）的
                   根三明治内发射；
      standalone : list[CompleteBranch]（key=None，自包含，可置于任意位置发射）。

    判定与构造使用平移后的有效函数 d' = shift(fn_A, r_A) ⊕ shift(fn_B, r_B)
    （平移不改变方向子空间；base 平移由 affine_from_fn 自然吸收，见代数文档 §1.4）。
    """
    if not entries:
        return [], []
    pool = [(cb, r, key) for cb, r, key in entries]
    eff = [shift_fn(cb.fn(), r, n) for cb, r, _ in pool]
    if max_rounds is None:
        max_rounds = len(pool) + 1
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        cands = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                d = eff[i] ^ eff[j]
                if d == 0:
                    cands.append((pool[i][0].t_cost() + pool[j][0].t_cost(), i, j, True))
                    continue
                ok, k = is_affine_cube(d, n)
                if not ok:
                    continue
                gain = pool[i][0].t_cost() + pool[j][0].t_cost() - t_cost(n - k)
                if gain > 0:
                    cands.append((gain, i, j, False))
        if not cands:
            break
        cands.sort(reverse=True)
        used = set()
        new_pool, new_eff = [], []
        for gain, i, j, cancel in cands:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            if cancel:
                continue
            d = eff[i] ^ eff[j]
            base, basis = affine_from_fn(d, n)
            plan = plan_from_affine(base, basis, n)
            fused = CompleteBranch(n, synthetic=plan, origin=pool[i][0].origin)
            assert fused.fn() == d, f'全局融合契约失败: {pool[i][0]!r} ⊕ {pool[j][0]!r}'
            new_pool.append((fused, 0, None))
            new_eff.append(fused.fn())
        pool = [pool[idx] for idx in range(len(pool)) if idx not in used] + new_pool
        eff = [eff[idx] for idx in range(len(eff)) if idx not in used] + new_eff
    kept = [(cb, r, key) for cb, r, key in pool if key is not None]
    standalone = [cb for cb, r, key in pool if key is None]
    return kept, standalone


def pool_t_cost(kept, standalone):
    """池（kept+standalone）的 ΣT。"""
    return sum(cb.t_cost() for cb, _, _ in kept) + sum(cb.t_cost() for cb in standalone)


def anf_pool(bitstr, n):
    """
    ANF 分解池（纯代数初始分解）：真值表 bitstr 经快速 Möbius 变换得到 ANF 系数，
    每个单项式 x^S（= 仿射立方）即一个 CompleteBranch（mask=bitstr=S，全正极性）。
    返回 [(CompleteBranch, root=0, key)]，可直接喂给 fuse_pool_global(_multistart)。
    池 XOR 不变量：⊕ 分支函数 == 原真值表（ANF 定义）。
    """
    a = bitstr
    size = 1 << n
    for j in range(n):
        step = 1 << j
        block = step << 1
        for base in range(0, size, block):
            for m in range(base, base + step):
                if (a >> m) & 1:
                    a ^= 1 << (m + step)
    entries = []
    for m in range(size):
        if (a >> m) & 1:
            entries.append((CompleteBranch(n, mask=m, bitstr=m), 0, m))
    return entries

