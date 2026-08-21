# -*- coding: utf-8 -*-
"""
阶段 0.1:闭式判定/构造 vs fn 判定/构造的全量一致性交叉验证。

被验证对象:`limbdd/fusion/ClosedForm.py`(闭式,O(n^2))
参照基准  :`limbdd/fusion/BranchFusion.py` + `limbdd/CompleteBranch.py` 的 fn 路径(O(2^n))

这是 Optimization_roadmap.md 阶段 1.1(T-1 切换)的**验收前提**:必须先在 n=4..7 上
确认零分歧,才允许把生产路径切到闭式。

核心等价判据
------------
`plan_from_affine(base, basis, n)` 只依赖仿射**集合**,与代表元/基的选择无关:
  - `rref` 对给定子空间唯一 ⇒ 主元/自由列划分唯一;
  - 若 e' = e ⊕ v (v ∈ K),则自由列常数 c_f 不变
    (因 v 满足齐次约束 v_f ⊕ Σ_p (r_p)_f v_p = 0)。
因此「两条路径给出同一个 SyntheticPlan」⟺「两条路径给出同一个仿射集合」,
可以直接用 `==` 做最强断言,无需逐一比对 (base, basis) 表示。
"""
from collections import Counter
from random import Random, randint

from utils.TruthTable import TruthTable
from limbdd.CompleteBranch import (CompleteBranch, abs_lim, affine_from_fn,
                                   extract_complete_branches, in_span, is_affine_cube,
                                   plan_from_affine, reduce_basis, t_cost)
from limbdd.fusion.BranchFusion import anf_pool, four_case_kind, shift_fn
from limbdd.fusion.ClosedForm import (_build_joint, _subspace_leq, canonical_affine,
                                      closed_gain, fuse_closed)
from tests.test_complete_branch import prepare_pieces


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def bdd_entries(tt, reorder):
    """BDD 图阶段后的全局条目 (cb, root_mask, key);非零 root_mask 用于检验平移闭式。"""
    entries = []
    for key, g in enumerate(prepare_pieces(tt, reorder)):
        r = abs_lim(g.root.lim, g.var_order)
        for cb in extract_complete_branches(g, tt.num_var, anomalies=Counter()):
            entries.append((cb, r, key))
    return entries


def _pairs(count, max_pairs):
    """全部对;超过上限时**均匀抽样**(直接截断会只覆盖 (0, j),偏置严重)。"""
    out = [(i, j) for i in range(count) for j in range(i + 1, count)]
    if len(out) <= max_pairs:
        return out
    step = len(out) / max_pairs
    return [out[int(k * step)] for k in range(max_pairs)]


def _pools(tt, i):
    """交替产出 BDD 池(带根掩码)与 ANF 池(根掩码恒 0),两类都要覆盖。"""
    yield bdd_entries(tt, reorder=(i % 2 == 1))
    yield anf_pool(tt.bitstr, tt.num_var)


# kind3 与 kind4 严格互镜(C3 及其对称情形);kind1/kind2 自镜。
#
# **为什么必须按正反两序各判一次**:C3/C4 要求两段方向子空间嵌套且余维 1,而池中条目的
# 排列顺序与维数是**系统性相关**的,单一取向会让某一侧永不出现——加大样本量无效。
#   ANF 池:x^S 的方向子空间为 span{e_j : j∉S},嵌套要求 T ⊊ S 且 |S|=|T|+1;
#           位掩码上 T ⊂ S 必然 T < S,而条目按掩码升序排列,故固定 a<b 时恒有
#           mask_a ⊂ mask_b,即 dim K_A = dim K_B + 1 ⇒ **恒为 kind4,kind3 永不出现**。
#   BDD 池:实测同样偏置(正序恒 kind4、反序恒 kind3,kind1/kind2/None 计数不变)。
MIRROR_KIND = {None: None, 1: 1, 2: 2, 3: 4, 4: 3}


# ------------------------------------------------------------------ 主测试 ----

def test_closed_form_agreement(num_vars=5, rounds=1200, max_pairs=600, verbose=False):
    """闭式与 fn 路径在「可融合性 / 结果维数 / 融合段 / 收益」四项上逐对一致。"""
    print(f'Testing closed-form vs fn agreement with {rounds} random {num_vars}-input functions...')

    n = num_vars
    stat = Counter()
    for i in range(rounds):
        tt = random_tt(n)
        for entries in _pools(tt, i):
            if len(entries) < 2:
                continue
            for a, b in _pairs(len(entries), max_pairs):
                ca, ra, _ = entries[a]
                cb, rb, _ = entries[b]

                # ---- 基准:fn 路径 ----
                d = shift_fn(ca.fn(), ra, n) ^ shift_fn(cb.fn(), rb, n)
                ok_fn, dim_fn = is_affine_cube(d, n)

                # ---- 被验证:闭式(正反两序,兼作对称性验证)----
                aff_a = canonical_affine(ca, ra)
                aff_b = canonical_affine(cb, rb)
                res = fuse_closed(aff_a, aff_b, n)
                res_rev = fuse_closed(aff_b, aff_a, n)

                stat['pairs'] += 1
                assert res.ok == ok_fn, (
                    f'可融合性分歧: 闭式={res.ok} fn={ok_fn}\n  A={ca!r} r={ra}\n  B={cb!r} r={rb}')

                # ---- 对称性:d = f_A⊕f_B 与顺序无关 ⇒ 除 kind 3↔4 互换外全部一致 ----
                assert res_rev.ok == res.ok, f'对称性破坏(ok): {res.ok} vs {res_rev.ok}'
                assert res_rev.cancel == res.cancel, f'对称性破坏(cancel)'
                assert res_rev.kind == MIRROR_KIND[res.kind], (
                    f'对称性破坏(kind): 正序={res.kind} 反序={res_rev.kind}\n'
                    f'  A={ca!r} r={ra}\n  B={cb!r} r={rb}')

                if not ok_fn:
                    stat['not_fusable'] += 1
                    continue

                if d == 0:
                    stat['cancel'] += 1
                    assert res.cancel, f'抵消未被闭式识别: A={ca!r} B={cb!r}'
                    assert res.dim == dim_fn == -1, f'抵消维数约定不一致: {res.dim} vs {dim_fn}'
                    continue

                assert not res.cancel, f'闭式误判为抵消(d != 0): A={ca!r} B={cb!r}'
                assert res.dim == dim_fn, (
                    f'结果维数分歧: 闭式={res.dim} fn={dim_fn}\n  A={ca!r} r={ra}\n  B={cb!r} r={rb}')
                assert res_rev.dim == res.dim, f'对称性破坏(dim): {res.dim} vs {res_rev.dim}'

                # ---- 融合段必须逐字段相同(等价于同一仿射集合),正反两序都要对 ----
                plan_cf = plan_from_affine(res.base, res.basis, n)
                plan_rev = plan_from_affine(res_rev.base, res_rev.basis, n)
                plan_fn = plan_from_affine(*affine_from_fn(d, n), n)
                assert plan_cf == plan_fn, (
                    f'融合段分歧:\n  闭式={plan_cf!r}\n  fn  ={plan_fn!r}\n'
                    f'  A={ca!r} r={ra}\n  B={cb!r} r={rb}')
                assert plan_rev == plan_fn, (
                    f'反序融合段分歧:\n  反序={plan_rev!r}\n  fn  ={plan_fn!r}\n'
                    f'  A={ca!r} r={ra}\n  B={cb!r} r={rb}')

                # ---- 融合契约:段函数精确等于 d ----
                assert CompleteBranch(n, synthetic=plan_cf).fn() == d, (
                    f'闭式融合契约失败: A={ca!r} ⊕ B={cb!r}')

                # ---- 收益口径一致 ----
                gain_fn = ca.t_cost() + cb.t_cost() - t_cost(n - dim_fn)
                assert closed_gain(ca, cb, res, t_cost, n) == gain_fn, '收益口径分歧'

                # 正反两序各计一次:kind3/kind4 互镜故计数必然相等,kind1/kind2 自镜故计两次
                stat[f'kind{res.kind}'] += 1
                stat[f'kind{res_rev.kind}'] += 1
                if gain_fn > 0:
                    stat['positive_gain'] += 1

    print(f'  checked pairs: {stat["pairs"]}, fusable: {stat["pairs"] - stat["not_fusable"]}, '
          f'cancel: {stat["cancel"]}, positive gain: {stat["positive_gain"]}')
    print(f'  kind distribution (正反两序各计一次): ' + ', '.join(
        f'{k}={stat[k]}' for k in ('kind1', 'kind2', 'kind3', 'kind4')))
    # 覆盖率告警:某一情形从未出现 ⇒ 该分支未被验证,不足以支撑阶段 1.1 的切换
    unseen = [k for k in ('kind1', 'kind2', 'kind3', 'kind4') if not stat[k]]
    if unseen:
        print(f'  [WARN] 以下情形在本轮未被覆盖,闭式对应分支未经验证: {", ".join(unseen)}')
    if verbose:
        print(f'  raw stats: {dict(stat)}')
    assert stat['pairs'] > 0, '未检查到任何分支对,测试无效'
    print('Closed-form agreement test passed!!!')
    return stat


def test_closed_form_translation(num_vars=5, rounds=120, samples=8, verbose=False):
    """平移闭式:canonical_affine(cb, r) 与 affine_from_fn(shift_fn(fn, r)) 描述同一仿射集。"""
    print(f'Testing closed-form translation with {rounds} random {num_vars}-input functions...')

    n = num_vars
    rng = Random(0)
    checked = 0
    for i in range(rounds):
        tt = random_tt(n)
        entries = bdd_entries(tt, reorder=(i % 2 == 1)) + anf_pool(tt.bitstr, n)
        for cb, _, _ in entries[:samples]:
            for r in (0, rng.randrange(1 << n), (1 << n) - 1):
                base_cf, basis_cf = canonical_affine(cb, r)
                aff = affine_from_fn(shift_fn(cb.fn(), r, n), n)
                # 子空间相等用「同维 + 双向包含」,不依赖行阶梯形的唯一性
                assert (len(basis_cf) == len(aff.basis)
                        and _subspace_leq(basis_cf, aff.basis)
                        and _subspace_leq(aff.basis, basis_cf)), (
                    f'平移后方向子空间分歧: 闭式={basis_cf} fn={aff.basis} (r={r}, {cb!r})')
                assert in_span(base_cf ^ aff.base, basis_cf), (
                    f'平移后 base 不在同一陪集: 闭式={base_cf} fn={aff.base} (r={r}, {cb!r})')
                # 最强判据:同一仿射集 ⟺ 同一规范 SyntheticPlan
                assert (plan_from_affine(base_cf, basis_cf, n)
                        == plan_from_affine(aff.base, aff.basis, n)), (
                    f'平移后仿射集分歧 (r={r}, {cb!r})')
                checked += 1

    print(f'  checked (branch, shift) pairs: {checked}')
    assert checked > 0, '未检查到任何分支,测试无效'
    print('Closed-form translation test passed!!!')


def test_intersection_basis(num_vars=5, rounds=80, max_pairs=40, verbose=False):
    """_build_joint 的交空间基:维数满足 dim(A∩B) = dimA + dimB − dim(A+B),且成员合法。"""
    print(f'Testing intersection basis with {rounds} random {num_vars}-input functions...')

    n = num_vars
    checked = 0
    for i in range(rounds):
        tt = random_tt(n)
        entries = bdd_entries(tt, reorder=(i % 2 == 1)) + anf_pool(tt.bitstr, n)
        if len(entries) < 2:
            continue
        for a, b in _pairs(len(entries), max_pairs):
            _, ka = canonical_affine(entries[a][0], entries[a][1])
            _, kb = canonical_affine(entries[b][0], entries[b][1])

            _, wb = _build_joint(ka, kb)
            expected = len(ka) + len(kb) - len(reduce_basis(ka + kb))
            assert len(wb) == expected, (
                f'交空间维数错误: 得到 {len(wb)}, 期望 {expected} (ka={ka}, kb={kb})')
            for v in wb:
                assert in_span(v, ka) and in_span(v, kb), f'交空间向量 {v} 不在两个子空间内'
            checked += 1

    print(f'  checked subspace pairs: {checked}')
    assert checked > 0, '未检查到任何子空间对,测试无效'
    print('Intersection basis test passed!!!')


def test_kind_matches_four_case(num_vars=5, rounds=120, max_pairs=60, verbose=False):
    """闭式的四情形编号与既有 BranchFusion.four_case_kind 一致(正反两序都比对)。"""
    print(f'Testing kind agreement with four_case_kind on {rounds} random {num_vars}-input functions...')

    n = num_vars
    checked = 0
    seen = Counter()
    for _ in range(rounds):
        tt = random_tt(n)
        entries = anf_pool(tt.bitstr, n)   # ANF 池 root_mask 恒为 0,可直接与 four_case_kind 比对
        if len(entries) < 2:
            continue
        for a, b in _pairs(len(entries), max_pairs):
            ca, cb = entries[a][0], entries[b][0]
            for x, y in ((ca, cb), (cb, ca)):   # 反序才能覆盖 kind3
                res = fuse_closed(canonical_affine(x), canonical_affine(y), n)
                expected = four_case_kind(x, y)
                assert res.kind == expected, (
                    f'情形编号分歧: 闭式={res.kind} four_case_kind={expected}\n  A={x!r}\n  B={y!r}')
                seen[res.kind] += 1
                checked += 1
            # 顺带校验生产实现自身的对称性
            assert four_case_kind(cb, ca) == MIRROR_KIND[four_case_kind(ca, cb)], (
                f'four_case_kind 对称性破坏:\n  A={ca!r}\n  B={cb!r}')

    print(f'  checked (pair, order) combos: {checked}, kinds seen: '
          + ', '.join(f'{k}={seen[k]}' for k in (1, 2, 3, 4)))
    assert checked > 0, '未检查到任何分支对,测试无效'
    print('Kind agreement test passed!!!')


def test_cancellation_closed_form(num_vars=5, rounds=60, samples=6, verbose=False):
    """
    C1(抵消)分支与 kind1 内部的 C1/C2 判别。

    初始池天然不含重复段(实测各 n 均 cancel=0,重复只在融合产物重入池后出现),
    故必须**显式构造同函数段**才能覆盖 `fuse_closed` 的 cancel 分支。
    """
    print(f'Testing cancellation branch with {rounds} random {num_vars}-input functions...')

    n = num_vars
    checked = Counter()
    for i in range(rounds):
        tt = random_tt(n)
        entries = bdd_entries(tt, reorder=(i % 2 == 1)) + anf_pool(tt.bitstr, n)
        for cb, r, _ in entries[:samples]:
            base, basis = aff = canonical_affine(cb, r)

            # (a) 自配对必抵消
            res = fuse_closed(aff, aff, n)
            assert res.ok and res.cancel and res.dim == -1, f'自配对未判为抵消: {cb!r}'
            checked['self'] += 1

            # (b) 同函数、不同表示(走线段 vs 合成段)必抵消
            fn_shifted = shift_fn(cb.fn(), r, n)
            twin = CompleteBranch(n, synthetic=plan_from_affine(*affine_from_fn(fn_shifted, n), n))
            assert twin.fn() == fn_shifted, f'构造的同函数段不等价: {cb!r}'
            res_twin = fuse_closed(aff, canonical_affine(twin), n)
            assert res_twin.ok and res_twin.cancel, f'同函数不同表示未判为抵消: {cb!r}'
            ok_fn, dim_fn = is_affine_cube(fn_shifted ^ twin.fn(), n)
            assert ok_fn and dim_fn == -1, 'fn 路径对 d=0 的约定不一致'
            checked['twin'] += 1

            # (c) 负向对照:同 K、异陪集 ⇒ kind1 但**不**抵消,维数 +1(C2)
            v = next((1 << j for j in range(n) if not in_span(1 << j, basis)), None)
            if v is None:      # dim K = n(常数 1 段),不存在异陪集
                continue
            res_c2 = fuse_closed(aff, (base ^ v, basis), n)
            assert res_c2.ok and res_c2.kind == 1 and not res_c2.cancel, (
                f'同 K 异陪集应为 C2 融合而非抵消: {cb!r}')
            assert res_c2.dim == len(basis) + 1, (
                f'C2 维数应为 dim K + 1: 得到 {res_c2.dim}, 期望 {len(basis) + 1}')
            checked['c2'] += 1

    print(f'  self-pairs: {checked["self"]}, twin-pairs: {checked["twin"]}, '
          f'C2 controls: {checked["c2"]}')
    assert checked['self'] and checked['twin'] and checked['c2'], '抵消/C2 分支未被覆盖'
    print('Cancellation branch test passed!!!')


# ------------------------------------------------------------------- 入口 ----

# n -> (rounds, max_pairs);阶段 0.1 要求覆盖 n=4..7
PROFILE = {
    4: (150, 60),
    5: (120, 60),
    6: (60, 60),
    7: (30, 40),
}


def main(verbose=False):
    for n in sorted(PROFILE):
        rounds, max_pairs = PROFILE[n]
        print(f'====== n = {n} ======')
        test_closed_form_agreement(n, rounds, max_pairs, verbose)
        test_closed_form_translation(n, max(rounds // 3, 10), 8, verbose)
        test_intersection_basis(n, max(rounds // 2, 10), max_pairs, verbose)
        test_kind_matches_four_case(n, max(rounds // 2, 10), max_pairs, verbose)
        test_cancellation_closed_form(n, max(rounds // 3, 10), 6, verbose)
    print('====== All closed-form consistency tests passed!!! ======')


if __name__ == '__main__':
    main()
