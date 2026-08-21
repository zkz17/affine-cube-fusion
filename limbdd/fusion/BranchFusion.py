# -*- coding: utf-8 -*-
"""
Phase 3：分支融合原语。

阶段 1.1（Optimization_roadmap.md T-1）起，判定与构造全部走**闭式路径**；
阶段 1.3（T-2）起，贪心候选维护改为**分桶 + 增量**（见 `_greedy_fusion`）。

- is_fusable(a, b)   ：权威判定（定理 2 的结构判定，O(n²)；d==0 亦融合=抵消）；
- four_case_kind(a, b)：定理 2 的四情形结构分类（测试与桶预筛用）；
- fuse_pair(a, b)    ：构造融合段（SyntheticPlan）；抵消返回 (None, True)；
- fuse_pool(branches)：贪心融合池（XOR 不变量、段数单调减）；
- fuse_pool_global   ：跨片全局融合池（根掩码平移 = base ^= r）。

数学依据：docs/CompleteBranch_algebra.md（定理 2 四情形、§3.6/§3.7 RREF 规范构造）；
工程规范：docs/CompleteBranch_fusion_plan.md §2.4 / §7.9。
闭式实现：limbdd/fusion/ClosedForm.py；其与旧 fn 路径的等价性已由
tests/test_closed_form.py 全量交叉验证（n=4..7），输出的逐位一致性由
experiments/phase0_results/baseline_tcost_{n}.txt 的 diff 验收。

分桶与增量（1.3，与旧轨迹逐位一致的理由）：
  - 定理 2 保证可融合对的 dim(K) 差 ≤ 1，故只枚举同桶与相邻桶对（分桶）；
  - 候选元组 (gain, i, j, cancel) 的键 (i, j) 全局唯一，且池重建时保留项的
    old→new 索引映射严格单调 ⇒ 重映射后的幸存候选**仍保持降序**，只需与新候选
    合并，无需重判（增量）；新候选仅涉及融合产物，O(|kept|·|fused| + |fused|²)。

旧 fn 路径（fn() / is_affine_cube / affine_from_fn / shift_fn）仍保留：
  - 供 counterexample_search.py、residual_measurement.py 等研究脚本使用；
  - 设环境变量 LIMBDD_FUSION_DEBUG=1 时，生产路径每次融合后与 fn 路径对照校验。
"""
import os

from limbdd.CompleteBranch import (CompleteBranch, t_cost, in_span, reduce_basis,
                                   plan_from_affine)
from limbdd.fusion.ClosedForm import ClosedFusion, NOT_FUSABLE, canonical_affine, fuse_closed

# 调试开关：每次融合后用 fn 路径重新校验闭式产物（慢，仅调试用）
FUSION_DEBUG = os.environ.get('LIMBDD_FUSION_DEBUG') == '1'


def is_fusable(a, b):
    """权威可融合判定：supp(f_A⊕f_B) 为仿射子空间（含 d==0 的抵消）。闭式，O(n²)。"""
    assert isinstance(a, CompleteBranch) and isinstance(b, CompleteBranch) and a.n == b.n
    return fuse_closed(canonical_affine(a), canonical_affine(b), a.n).ok


def four_case_kind(a, b):
    """定理 2 的四情形结构分类（docs/CompleteBranch_algebra.md §2.2）。返回 1..4 或 None。"""
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


def _fuse_from_result(res, n, origin):
    """由闭式判定结果构造融合段（规范 SyntheticPlan，无 2^n 枚举）。"""
    plan = plan_from_affine(res.base, res.basis, n)
    return CompleteBranch(n, synthetic=plan, origin=origin)


def fuse_pair(a, b):
    """
    构造融合段。返回 (fused, cancelled)：
      cancelled=True  → f_A⊕f_B == 0，两者应从池中移除；
      fused 非 None   → 替换两者的合成段（规范 SyntheticPlan）；
      (None, False)   → 不可融合。
    """
    if not isinstance(a, CompleteBranch) or not isinstance(b, CompleteBranch) or a.n != b.n:
        return None, False
    res = fuse_closed(canonical_affine(a), canonical_affine(b), a.n)
    if not res.ok:
        return None, False
    if res.cancel:
        return None, True
    fused = _fuse_from_result(res, a.n, a.origin)
    if FUSION_DEBUG:
        d = a.fn() ^ b.fn()
        assert fused.fn() == d, f'融合契约失败: {a!r} ⊕ {b!r}'
    return fused, False


# ------------------------------------------------------------------ 贪心引擎 ----

class _Entry:
    """贪心引擎的池条目：分支 + 已算好的规范仿射表示（全局池中已含根掩码平移）。"""

    __slots__ = ('cb', 'aff', 'cost', 'extra', 'plain')

    def __init__(self, cb, aff, extra=None):
        self.cb = cb
        self.aff = aff            # (base, basis),canonical_affine 的产出
        self.cost = cb.t_cost()
        self.extra = extra        # fuse_pool: None;global: (root_mask, key)
        # 纯立方快径:方向子空间为坐标子空间(基全为单位向量)时,记录约束集掩码
        # C = { 被约束的变量 } = 非自由列;ANF 单项式、多数 BDD 走线段都满足。
        basis = aff[1]
        if all(v & (v - 1) == 0 for v in basis):
            cols = 0
            for v in basis:
                cols |= v
            self.plain = ((1 << cb.n) - 1) ^ cols
        else:
            self.plain = None


def _fuse_plain(a, b, ca, cb, n):
    """
    两个「纯立方」(方向子空间为坐标子空间)的 O(popcount) 闭式判定。

    支撑 = {x : x_C = (base 在 C 上的投影)},C 为约束集掩码。
    与 fuse_closed 的定理 2 分类完全一致,构造亦给出同一仿射集
    (规范 SyntheticPlan 由 plan_from_affine 唯一决定,故输出逐位不变)。
    """
    d = a ^ b
    if ca == cb:
        if d & ca == 0:
            return ClosedFusion(True, 1, True, -1, 0, ())      # C1
        units = tuple(1 << v for v in range(n) if not ((ca >> v) & 1))
        return ClosedFusion(True, 1, False, n - ca.bit_count() + 1, a, units + (d & ca,))  # C2
    if ca.bit_count() == cb.bit_count() + 1 and cb & ~ca == 0:          # K_A ⊂ K_B(余维 1)
        if d & cb == 0:                                                 # 同大陪集
            w = ca & ~cb
            units = tuple(1 << v for v in range(n) if not ((ca >> v) & 1))
            return ClosedFusion(True, 3, False, n - ca.bit_count(), a ^ w, units)
    if cb.bit_count() == ca.bit_count() + 1 and ca & ~cb == 0:          # K_B ⊂ K_A(余维 1)
        if d & ca == 0:
            w = cb & ~ca
            units = tuple(1 << v for v in range(n) if not ((cb >> v) & 1))
            return ClosedFusion(True, 4, False, n - cb.bit_count(), b ^ w, units)
    if ca.bit_count() == cb.bit_count():
        inter = ca & cb
        if inter.bit_count() == ca.bit_count() - 1 and d & inter == 0:  # C4:交余维 1 且陪集相交
            j = (cb & ~ca).bit_length() - 1
            jp = (ca & ~cb).bit_length() - 1
            t = a ^ (d & ~ca)
            wb = tuple(1 << v for v in range(n) if not ((ca | cb) >> v) & 1)
            return ClosedFusion(True, 2, False, n - ca.bit_count(),
                                t ^ (1 << j), wb + ((1 << j) ^ (1 << jp),))
    return NOT_FUSABLE


def _fuse_two(e_i, e_j, n):
    """池内两段的闭式判定:纯立方快径优先,否则走通用 fuse_closed。"""
    if e_i.plain is not None and e_j.plain is not None:
        return _fuse_plain(e_i.aff[0], e_j.aff[0], e_i.plain, e_j.plain, n)
    return fuse_closed(e_i.aff, e_j.aff, n)


def _pair_candidate(e_i, e_j, i, j, n):
    """dim 预筛 + 闭式判定;返回候选元组 (gain, i, j, cancel) 或 None。"""
    if abs(len(e_i.aff[1]) - len(e_j.aff[1])) > 1:
        return None      # 定理 2:可融合 ⟹ dim(K) 差 ≤ 1
    res = _fuse_two(e_i, e_j, n)
    if not res.ok:
        return None
    if res.cancel:
        # 抵消恒有益（即使 T 收益为 0 也减少段数与门数），始终入候选
        return (e_i.cost + e_j.cost, i, j, True)
    gain = e_i.cost + e_j.cost - t_cost(n - res.dim)
    if gain > 0:
        return (gain, i, j, False)
    return None


def _bucket_pairs(pool):
    """
    按 dim(K) 分桶产出全部 |Δdim| ≤ 1 的 (i, j) 对（i < j），生成器实现避免物化。

    候选键 (gain, i, j, cancel) 的 (i, j) 全局唯一，故产出顺序不影响排序结果;
    与旧代码全量 (i<j) 枚举相比，唯一差异是跳过定理 2 保证不可融合的对。
    """
    buckets = {}
    for idx, e in enumerate(pool):
        buckets.setdefault(len(e.aff[1]), []).append(idx)
    for d in sorted(buckets):
        b = buckets[d]
        for a in range(len(b)):
            for c in range(a + 1, len(b)):
                yield b[a], b[c]
        b2 = buckets.get(d + 1)
        if b2:
            for i in b:
                for j in b2:
                    if i < j:
                        yield i, j
                    else:
                        yield j, i


def _merge_desc(a, b):
    """合并两个按 (gain, i, j, cancel) 降序排列的候选列表(元素互斥,故无平局歧义)。"""
    out = []
    p = q = 0
    while p < len(a) and q < len(b):
        if a[p] >= b[q]:
            out.append(a[p])
            p += 1
        else:
            out.append(b[q])
            q += 1
    out.extend(a[p:])
    out.extend(b[q:])
    return out


def _greedy_fusion(entries, n, max_rounds, fused_extra, debug_check=None):
    """
    分桶 + 增量的贪心融合引擎,与旧「每轮全量重枚举 + 重排序」**逐位一致**:

    1. 首轮按桶枚举全部 |Δdim|≤1 对,建候选并排序;
    2. 每轮按 (gain, i, j) 降序做极大匹配(每分支至多用一次);
    3. 池重建 = 保留项(原序)+ 融合产物(候选处理序);
    4. 幸存候选只重映射索引(严格单调映射保序),新候选只涉及融合产物,
       合并两个降序列表即得本轮候选——无需对保留对重新判定。

    fused_extra: 融合产物 _Entry 的 extra 字段(fuse_pool 用 None;global 用 (0, None))。
    返回最终池(_Entry 列表)。
    """
    pool = list(entries)
    if max_rounds is None:
        max_rounds = len(pool) + 1

    cands = []
    for i, j in _bucket_pairs(pool):
        c = _pair_candidate(pool[i], pool[j], i, j, n)
        if c is not None:
            cands.append(c)
    cands.sort(reverse=True)

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        if not cands:
            break

        used = set()
        fused_list = []
        for gain, i, j, cancel in cands:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            if cancel:
                continue
            res = _fuse_two(pool[i], pool[j], n)
            assert res.ok and not res.cancel
            fused = _fuse_from_result(res, n, pool[i].cb.origin)
            if FUSION_DEBUG and debug_check is not None:
                debug_check(pool[i], pool[j], fused)
            fused_list.append(fused)

        kept = [e for idx, e in enumerate(pool) if idx not in used]
        fused_entries = [_Entry(f, canonical_affine(f), extra=fused_extra)
                         for f in fused_list]
        pos = {}
        for new_idx, old_idx in enumerate(idx for idx in range(len(pool)) if idx not in used):
            pos[old_idx] = new_idx
        pool = kept + fused_entries

        # ---- 增量维护候选 ----
        remapped = []
        for g, i, j, c in cands:
            if i in used or j in used:
                continue
            remapped.append((g, pos[i], pos[j], c))
        # pos 严格单调 ⇒ remapped 仍降序

        n_kept = len(kept)
        m_new = len(pool)
        new_cands = []
        for i in range(n_kept):
            for j in range(n_kept, m_new):
                c = _pair_candidate(pool[i], pool[j], i, j, n)
                if c is not None:
                    new_cands.append(c)
        for i in range(n_kept, m_new):
            for j in range(i + 1, m_new):
                c = _pair_candidate(pool[i], pool[j], i, j, n)
                if c is not None:
                    new_cands.append(c)
        new_cands.sort(reverse=True)
        cands = _merge_desc(remapped, new_cands)

    return pool


def fuse_pool(branches, max_rounds=None):
    """
    贪心融合池：XOR 不变量、段数单调减，返回融合后的新列表（不修改输入）。

    策略：每轮收集全部正收益候选（抵消 / d 仿射融合），按 gain 降序贪心
    匹配（每分支至多用一次），融合段重入池；无正收益候选时停止。
    判定与构造全部走闭式；候选维护为分桶 + 增量，轨迹与旧实现逐位一致。
    """
    if not branches:
        return []
    n = branches[0].n
    assert all(isinstance(b, CompleteBranch) and b.n == n for b in branches)

    entries = [_Entry(b, canonical_affine(b)) for b in branches]
    if FUSION_DEBUG:
        def debug_check(e_i, e_j, fused):
            d = e_i.cb.fn() ^ e_j.cb.fn()
            assert fused.fn() == d, f'融合契约失败: {e_i.cb!r} ⊕ {e_j.cb!r}'
    else:
        debug_check = None
    pool = _greedy_fusion(entries, n, max_rounds, None, debug_check)
    return [e.cb for e in pool]


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

    判定与构造使用平移后的有效函数 d' = f_A(x⊕r_A) ⊕ f_B(x⊕r_B)。闭式上等价于
    base ^= r（K 平移不变），即 canonical_affine(cb, r)，见代数文档 §1.4。
    """
    if not entries:
        return [], []
    pool_entries = [_Entry(cb, canonical_affine(cb, r), extra=(r, key))
                    for cb, r, key in entries]
    if FUSION_DEBUG:
        def debug_check(e_i, e_j, fused):
            d = shift_fn(e_i.cb.fn(), e_i.extra[0], n) ^ shift_fn(e_j.cb.fn(), e_j.extra[0], n)
            assert fused.fn() == d, f'全局融合契约失败: {e_i.cb!r} ⊕ {e_j.cb!r}'
    else:
        debug_check = None
    pool = _greedy_fusion(pool_entries, n, max_rounds, (0, None), debug_check)
    kept = [(e.cb, e.extra[0], e.extra[1]) for e in pool if e.extra[1] is not None]
    standalone = [e.cb for e in pool if e.extra[1] is None]
    return kept, standalone


def pool_t_cost(kept, standalone):
    """池（kept+standalone）的 ΣT。"""
    return sum(cb.t_cost() for cb, _, _ in kept) + sum(cb.t_cost() for cb in standalone)


def anf_pool(bitstr, n):
    """
    ANF 分解池（纯代数初始分解）：真值表 bitstr 经快速 Möbius 变换得到 ANF 系数，
    每个单项式 x^S（= 仿射立方）即一个 CompleteBranch（mask=bitstr=S，全正极性）。
    返回 [(CompleteBranch, root=0, key)]，可直接喂给 fuse_pool_global。
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
