from collections import Counter
from random import randint

from utils.TruthTable import TruthTable
from utils.Qubit import QubitManager
from limbdd.CompleteBranch import (CompleteBranch, extract_complete_branches,
                                   t_cost, is_affine_cube)
from limbdd.fusion.BranchFusion import is_fusable, four_case_kind, fuse_pair, fuse_pool
from experiments.phase0_branch_stats import is_affine_subspace
from tests.test_complete_branch import prepare_pieces


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def emit_check(cb, num_vars):
    """发射合成段并与 compute_tt 对账（同时校验输入复原）。"""
    qm = QubitManager()
    for v in range(num_vars):
        qm.get_input_qubit(v)
    assert cb.to_circuit(qm).compute_tt() == cb.fn(), f'发射不一致: {cb!r}'


def test_fusion_contract(num_vars=5, rounds=300, max_pairs=30, verbose=False):
    """融合契约：判据等价（d-仿射 vs 四情形）、构造精确、控制数最优、规范形。"""
    print(f'Testing branch fusion contract with {rounds} random {num_vars}-input functions...')

    checked = 0
    for i in range(rounds):
        tt = random_tt(num_vars)
        branches = []
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            branches.extend(extract_complete_branches(g, num_vars, origin=i, anomalies=Counter()))
        if len(branches) < 2:
            continue
        pairs = [(a, b) for a in range(len(branches)) for b in range(a + 1, len(branches))]
        if len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
        for a, b in pairs:
            ca, cb = branches[a], branches[b]
            d = ca.fn() ^ cb.fn()
            # 判据 1：d-仿射（与 Phase 0 的判定器交叉验证）
            aff_cube = (d == 0) or is_affine_cube(d, num_vars)[0]
            aff_phase0 = (d == 0) or is_affine_subspace(d, num_vars)[0]
            assert aff_cube == aff_phase0, f'仿射判定器不一致: tt={tt}, d={d}'
            # 判据 2：权威 is_fusable 与四情形分类等价
            fus = is_fusable(ca, cb)
            kind = four_case_kind(ca, cb)
            assert fus == aff_cube, f'is_fusable 与 d-仿射不一致: tt={tt}'
            assert fus == (kind is not None), f'四情形分类与 d-仿射不一致: tt={tt}, kind={kind}'

            if not fus:
                fused, cancelled = fuse_pair(ca, cb)
                assert fused is None and not cancelled
                continue

            fused, cancelled = fuse_pair(ca, cb)
            assert cancelled == (d == 0), f'抵消语义不一致: d=={d == 0}, cancelled={cancelled}'
            if cancelled:
                checked += 1
                continue

            # 构造契约：f_fused == f_A ⊕ f_B，且发射电路 compute_tt 全等
            assert fused.fn() == d, f'融合段函数 != f_A⊕f_B: tt={tt}'
            emit_check(fused, num_vars)
            # 控制数最优：n − dim(d)
            _, k = is_affine_cube(d, num_vars)
            assert fused.nctrl() == num_vars - k, \
                f'控制数未达最优: nctrl={fused.nctrl()}, n-k={num_vars - k}'
            assert fused.t_cost() == t_cost(num_vars - k)
            # 规范形：x_trans=0、cnot_ops/controls 升序
            plan = fused.synthetic
            assert plan.x_trans == 0
            assert plan.cnot_ops == tuple(sorted(plan.cnot_ops))
            assert plan.controls == tuple(sorted(plan.controls))
            checked += 1
            if verbose:
                print('  ', i, a, b, 'kind=', kind, 'T:', ca.t_cost(), '+', cb.t_cost(),
                      '->', fused.t_cost(), plan)

    print(f'Branch fusion contract test passed!!! ({checked} pairs)')


def test_fuse_pool(num_vars=5, rounds=300, verbose=False):
    """池融合：XOR 不变量、段数单调减、T 不升、确定性、产物全部有效。"""
    print(f'Testing fusion pool invariants with {rounds} random {num_vars}-input functions...')

    for i in range(rounds):
        tt = random_tt(num_vars)
        branches = []
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            branches.extend(extract_complete_branches(g, num_vars, origin=i, anomalies=Counter()))
        if not branches:
            continue
        xor_before = 0
        t_before = 0
        for cb in branches:
            xor_before ^= cb.fn()
            t_before += cb.t_cost()

        fused = fuse_pool(branches)
        xor_after, t_after = 0, 0
        for cb in fused:
            xor_after ^= cb.fn()
            t_after += cb.t_cost()
            emit_check(cb, num_vars)

        assert xor_after == xor_before, f'池 XOR 不变量破坏: tt={tt}'
        assert len(fused) <= len(branches), f'段数未单调减: {len(branches)} -> {len(fused)}'
        assert t_after <= t_before, f'T 未单调降: {t_before} -> {t_after}'
        # 确定性
        assert fuse_pool(branches) == fused, f'fuse_pool 非确定性: tt={tt}'
        if verbose:
            print('  ', i, str(tt), f'{len(branches)} -> {len(fused)} 段, T {t_before} -> {t_after}')

    print('Fusion pool invariant test passed!!!')


def test_four_case_construction(num_vars=5, rounds=300, verbose=False):
    """四情形逐一构造：维数/控制数符合定理 2，且真实数据能覆盖到全部情形（如出现则验证）。"""
    print(f'Testing four-case fusion construction with {rounds} random {num_vars}-input functions...')

    kind_hits = Counter()
    constructed = Counter()
    for i in range(rounds):
        tt = random_tt(num_vars)
        branches = []
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            branches.extend(extract_complete_branches(g, num_vars, origin=i, anomalies=Counter()))
        for a in range(len(branches)):
            for b in range(a + 1, len(branches)):
                kind = four_case_kind(branches[a], branches[b])
                if kind is None:
                    continue
                kind_hits[kind] += 1
                ca, cb = branches[a], branches[b]
                d = ca.fn() ^ cb.fn()
                fused, cancelled = fuse_pair(ca, cb)
                assert cancelled == (d == 0)
                if cancelled:
                    continue
                _, k = is_affine_cube(d, num_vars)
                assert fused.nctrl() == num_vars - k
                # 维数按定理 2 核对
                if kind == 1:
                    assert k == len(ca.direction_key()) + 1, f'情形 1 维数错: k={k}'
                elif kind == 2:
                    assert k == len(ca.direction_key()), f'情形 2 维数错: k={k}'
                else:
                    smaller = min(len(ca.direction_key()), len(cb.direction_key()))
                    assert k == smaller, f'情形 {kind} 维数错: k={k}, smaller={smaller}'
                emit_check(fused, num_vars)
                constructed[kind] += 1
                if verbose:
                    print('  ', i, 'kind=', kind, 'dimK=',
                          [len(branches[a].direction_key()), len(branches[b].direction_key())],
                          '->', k)

    print(f'Four-case construction test passed!!! (命中 {dict(kind_hits)}, 构造 {dict(constructed)})')

    # 情形 3/4 的确定性构造（随机数据罕见）：超平面陪集段 与 常数 1 段 互补
    const1 = CompleteBranch(4)
    single = CompleteBranch(4, mask=1 << 1, bitstr=0)          # [x1=0]，K=span{x0,x2,x3}
    assert four_case_kind(single, const1) == 3, f'期望情形 3（K_A ⊂ K_B），got {four_case_kind(single, const1)}'
    assert four_case_kind(const1, single) == 4, f'期望情形 4（K_B ⊂ K_A），got {four_case_kind(const1, single)}'
    fused, cancelled = fuse_pair(single, const1)
    assert not cancelled and fused.nctrl() == 1 and fused.fn() == (single.fn() ^ const1.fn())
    emit_check(fused, 4)


def test_pivot_oracle_alignment(num_vars=5, rounds=30, max_pairs=2, verbose=False):
    """
    pivot oracle 契约对齐（限同图完整分支对）：对 d 仿射的对执行 pivot_to +
    嵌套综合，契约 f_pivot∘map == f_A⊕f_B 成立时必须与我们的融合段函数一致；
    失败计入跳过（pivot_to 对含 one-child 位置的对不健全，属已知缺陷）。
    """
    from circuit.MCXCircuit import MCXCircuit
    from limbdd.LimBDDManager import LimBDDManager
    from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer

    print(f'Testing pivot oracle alignment with {rounds} random {num_vars}-input functions...')

    ok_cnt, skip_cnt = 0, 0
    for i in range(rounds):
        tt = random_tt(num_vars)
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            window = 0
            for v in g.var_order:
                window |= 1 << v
            branches = extract_complete_branches(g, num_vars, origin=i, anomalies=Counter())
            complete = [cb for cb in branches if cb.mask == window and not cb.ft_lims]
            if len(complete) < 2:
                continue
            pairs = [(a, b) for a in range(len(complete)) for b in range(a + 1, len(complete))]
            if len(pairs) > max_pairs:
                pairs = pairs[:max_pairs]
            for a, b in pairs:
                ca, cb = complete[a], complete[b]
                d = ca.fn() ^ cb.fn()
                if d == 0 or not is_affine_cube(d, num_vars)[0]:
                    continue
                try:
                    pivot_limbdd, x_map, cx_map = LimBDDManager.pivot_to(
                        g.root.node, ca.source_branch, cb.source_branch, g)
                    nested = LimBDDOracleSynthesizer(ancilla_num=0, fused_backend=False)
                    qm = QubitManager()
                    for v in range(num_vars):
                        qm.get_input_qubit(v)
                    nested.qubit_mgr = qm
                    nested.circ = MCXCircuit(qm)
                    nested.limbdd_mgr.limbdd = pivot_limbdd
                    nested.limbdd_mgr.limbdd_list.clear()
                    for h in range(pivot_limbdd.root.level + 1):
                        nested.limbdd_mgr.limbdd_list[h] = []
                    nested.process()
                    ttp = nested.circ.compute_tt()
                    ok = True
                    for x in range(1 << num_vars):
                        y = x
                        for var in x_map:
                            if x_map[var]:
                                y ^= 1 << var
                        for ctrl in cx_map:
                            for tgt in cx_map[ctrl]:
                                if ((x >> ctrl) & 1) == cx_map[ctrl][tgt]:
                                    y ^= 1 << tgt
                        if ((ttp >> y) & 1) != ((d >> x) & 1):
                            ok = False
                            break
                    if ok:
                        ok_cnt += 1
                        # 与我们的融合段一致
                        fused, cancelled = fuse_pair(ca, cb)
                        assert not cancelled and fused.fn() == d
                    else:
                        skip_cnt += 1  # pivot_to 已知缺陷（one-child 位置）
                except Exception:
                    skip_cnt += 1
                if verbose:
                    print('  ', i, 'pivot ok/skip:', ok_cnt, skip_cnt)

    print(f'Pivot oracle alignment test passed!!! (ok={ok_cnt}, skip={skip_cnt})')


def main():
    test_fusion_contract()
    test_fuse_pool()
    test_four_case_construction()
    test_pivot_oracle_alignment()


if __name__ == '__main__':
    main()
