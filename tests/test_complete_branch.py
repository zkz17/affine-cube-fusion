from collections import Counter
from random import randint

from utils.TruthTable import TruthTable
from utils.Qubit import QubitManager
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.CompleteBranch import (CompleteBranch, SyntheticPlan, extract_complete_branches, t_cost)
from experiments.phase0_branch_stats import extract_branches, sigs_xor_tt, is_affine_subspace


def prepare_pieces(tt, reorder):
    """复刻流水线图阶段（与 Phase 0 探针一致，但停在 limbdd2circ 之前）。"""
    syn = LimBDDOracleSynthesizer(reorder=reorder, ancilla_num=0)
    syn.construct(tt)
    mgr = syn.limbdd_mgr
    mgr.try_neg(mgr.limbdd)
    mgr.split_neg(mgr.limbdd)
    if reorder:
        syn.variable_reorder()
    pieces = [mgr.limbdd] + [g for lst in mgr.limbdd_list.values() for g in lst]
    return [g for g in pieces if g.root.weight]


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_extraction_equivalence(num_vars=6, rounds=300, verbose=False):
    """压缩抽取 vs Phase 0 位置走线抽取：字段级逐项相等 + 函数一致 + 无异常。"""
    print(f'Testing CompleteBranch compact extraction with {rounds} random {num_vars}-input functions...')

    for i in range(rounds):
        tt = random_tt(num_vars)
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            anom = Counter()
            compact = extract_complete_branches(g, num_vars, origin=i, anomalies=anom)
            oracle = extract_branches(g, anom)
            if verbose:
                print('  ', i, str(tt), 'piece h=', g.root.level, 'br=', len(compact))

            assert len(compact) == len(oracle), f'分支数不一致: {len(compact)} vs {len(oracle)}'
            assert anom['neg_edge'] == 0 and anom['zero_node'] == 0 and anom['zero_level'] == 0 \
                and anom['ft_target_above'] == 0 and anom['lim_overflow'] == 0, \
                f'抽取异常: {dict(anom)}'

            for cb, (sig, bits, bitnum) in zip(compact, oracle):
                # 从位置走线重建期望的压缩字段（oracle 的 lim 已是绝对变量位空间）
                exp_mask, exp_bitstr, exp_xg, exp_ft = 0, 0, 0, []
                for (var, ft, bit, w0, w1, l0, l1) in sig:
                    if ft:
                        common = l0 & l1
                        exp_xg ^= common
                        m0, m1 = l0 ^ common, l1 ^ common
                        if m0 or m1:
                            exp_ft.append((var, m0, m1))
                    else:
                        exp_mask |= 1 << var
                        if bit:
                            exp_bitstr |= 1 << var
                        exp_xg ^= (l1 if bit else l0)

                assert cb.mask == exp_mask, f'mask 不一致: {cb.mask:b} vs {exp_mask:b}'
                assert cb.bitstr == exp_bitstr, f'bitstr 不一致'
                assert cb.x_global == exp_xg, f'x_global 不一致: {cb.x_global:b} vs {exp_xg:b}'
                assert cb.ft_lims == tuple(exp_ft), f'ft_lims 不一致: {cb.ft_lims} vs {exp_ft}'
                assert cb.source_branch.bitstr == bits and cb.source_branch.bitnum == bitnum, \
                    'source_branch 与 SimpleBranch 不一致'
                # 函数一致（压缩语义 == 位置走线语义）
                assert cb.fn() == sigs_xor_tt([sig], 0, num_vars), '压缩段函数与位置走线不一致'
                # 规范化不变量
                for ctrl, m0, m1 in cb.ft_lims:
                    assert m0 & m1 == 0
                    assert not ((cb.mask >> ctrl) & 1)

    print('Compact extraction test passed!!!')


def test_affine_and_emission(num_vars=6, rounds=200, verbose=False):
    """符号化仿射推导自洽 + 规范发射电路与 compute_tt 全等 + 支撑集大小不变量。"""
    print(f'Testing CompleteBranch affine derivation and emission with {rounds} random {num_vars}-input functions...')

    for i in range(rounds):
        tt = random_tt(num_vars)
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            for cb in extract_complete_branches(g, num_vars, origin=i, anomalies=Counter()):
                base, basis = cb.affine()
                k = cb.nctrl()
                # 支撑集 = base + span(basis)，大小 2^(n-k)
                assert cb.fn().bit_count() == (1 << (num_vars - k)), \
                    f'支撑集大小不符: n={num_vars}, k={k}, popcount={cb.fn().bit_count()}'
                # 发射电路与 fn 全等（compute_tt 同时校验输入复原/辅助清零）
                qm = QubitManager()
                for v in range(num_vars):
                    qm.get_input_qubit(v)
                circ_tt = cb.to_circuit(qm).compute_tt()
                assert circ_tt == cb.fn(), f'发射电路与 fn 不一致 (tt={tt}, cb={cb})'
                if verbose:
                    print('  ', i, str(tt), 'k=', k, 'dimK=', len(basis), 'base=', bin(base))

    print('Affine derivation & emission test passed!!!')


def _fusable(ca, cb):
    """§2.4 修订后的完整可融合判定（四种情形；与 BranchFusion.four_case_kind 同一实现）。"""
    from limbdd.fusion.BranchFusion import four_case_kind
    return four_case_kind(ca, cb) is not None


def test_direction_key_consistency(num_vars=5, rounds=300, max_pairs=40, verbose=False):
    """
    验证 §2.4 修订后的融合充要条件：d = f_A⊕f_B 为仿射立方（d==0 或支撑为
    仿射子空间）当且仅当四种情形之一成立（K 相等 / 同维相交余维 1 / 嵌套余维 1
    同大陪集及其对称）。维数关系：K 相等且 d≠0 → dim(d) = dim(K)+1；
    其余可融合情形 → dim(d) = dim(较小者)。
    """
    print(f'Testing fusion criterion consistency with {rounds} random {num_vars}-input functions...')

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
            aff, dim = is_affine_subspace(d, num_vars)
            aff = (d == 0) or aff
            fus = _fusable(ca, cb)
            assert aff == fus, \
                f'融合判据不一致: tt={tt}, d_affine={aff}, fusable={fus}, ' \
                f'K_A={ca.direction_key()}, K_B={cb.direction_key()}'
            if aff:
                same_k = (ca.direction_key() == cb.direction_key())
                if d == 0:
                    assert same_k, f'd==0 必然 K 相等: K_A={ca.direction_key()}, K_B={cb.direction_key()}'
                elif same_k:
                    assert dim == len(ca.direction_key()) + 1, \
                        f'K 相等融合维数应为 dim(K)+1: got {dim}'
                else:
                    smaller = min(len(ca.direction_key()), len(cb.direction_key()))
                    assert dim == smaller, \
                        f'其余可融合情形维数应为 dim(较小者): got {dim}, smaller={smaller}'
            checked += 1
            if verbose:
                print('  ', i, a, b, 'affine=', aff, 'fusable=', fus)

    print(f'Fusion criterion consistency test passed!!! ({checked} pairs)')


def test_synthetic_plan(num_vars=5, rounds=300, verbose=False):
    """合成段：随机 X/CNOT 网络 + MCX 的发射与仿射推导自洽；等价网络键一致。"""
    print(f'Testing SyntheticPlan with {rounds} random plans on {num_vars} inputs...')

    for i in range(rounds):
        n = num_vars
        x_trans = randint(0, (1 << n) - 1)
        cnot_ops = []
        for _ in range(randint(0, n)):
            c = randint(0, n - 1)
            t = randint(0, n - 1)
            if c != t:
                cnot_ops.append((c, t))
        ctrl_vars = [v for v in range(n) if randint(0, 1)]
        controls = tuple((v, randint(0, 1)) for v in ctrl_vars)

        plan = SyntheticPlan(x_trans=x_trans, cnot_ops=tuple(cnot_ops), controls=controls)
        cb = CompleteBranch(n, synthetic=plan)
        k = len(controls)

        assert cb.fn().bit_count() == (1 << (n - k)), \
            f'合成段支撑集大小不符: k={k}, popcount={cb.fn().bit_count()}'
        qm = QubitManager()
        for v in range(n):
            qm.get_input_qubit(v)
        assert cb.to_circuit(qm).compute_tt() == cb.fn(), f'合成段发射不一致: {plan}'
        if verbose:
            print('  ', i, plan, 'k=', k)

    # 等价网络：CNOT(x0→x1)+MCX(x1=0) 与 CNOT(x1→x0)+MCX(x0=0) 均为 [x0⊕x1==0]
    p1 = CompleteBranch(num_vars, synthetic=SyntheticPlan(0, ((0, 1),), ((1, 0),)))
    p2 = CompleteBranch(num_vars, synthetic=SyntheticPlan(0, ((1, 0),), ((0, 0),)))
    assert p1 != p2, '结构不等的两个计划不应相等'
    assert p1.cancel_key() == p2.cancel_key(), '函数相等 ⇒ cancel_key 相等'
    assert p1.direction_key() == p2.direction_key(), '方向子空间相等 ⇒ direction_key 相等'

    print('SyntheticPlan test passed!!!')


def main():
    test_extraction_equivalence()
    test_affine_and_emission()
    test_direction_key_consistency()
    test_synthetic_plan()


if __name__ == '__main__':
    main()
