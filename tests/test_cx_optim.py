# -*- coding: utf-8 -*-
"""
B-4(CX 优化)测试:段重排 + 发射基选择的正确性与不变量。

不变量:
  1. T 不变:发射基选择与段顺序都不改变段集合与控制数 ⇒ 发射 ΣT == 池 ΣT;
  2. CX 不劣:plan_min_cx 的 CNOT 数 ≤ 规范 plan;重排输出 CX ≤ 原始顺序;
  3. 语义不变:新发射电路 compute_tt == 真值表;
  4. 计数一致性:词项级 term_cx_cost == qiskit 级 count_cx_cost。
"""
from random import randint

from utils.Qubit import QubitManager
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_cx_cost, count_t_cost
from limbdd.CompleteBranch import CompleteBranch, plan_from_affine, plan_min_cx
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost
from circuit.MCXCircuit import MCXCircuit, order_segments_greedy, term_cx_cost_list


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def _qm(n):
    qm = QubitManager()
    for v in range(n):
        qm.get_input_qubit(v)
    return qm


def test_plan_min_cx(num_vars=5, rounds=200, verbose=False):
    """plan_min_cx:控制数不变、CNOT 不增、发射语义不变。"""
    print(f'Testing plan_min_cx with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        qm = _qm(n)
        for cb, _, _ in anf_pool(tt.bitstr, n)[:8]:
            base, basis = cb.affine()
            canon = plan_from_affine(base, basis, n)
            best = plan_min_cx(base, basis, n)
            assert len(best.controls) == len(canon.controls), '控制数必须不变(T 不变)'
            assert len(best.cnot_ops) <= len(canon.cnot_ops), 'plan_min_cx 不应增加 CNOT'

            twin = CompleteBranch(n, synthetic=canon)
            assert twin.to_circuit(qm).compute_tt() == twin.fn(), 'CX 最优基发射语义破坏'
    print('plan_min_cx test passed!!!')


def test_emission_t_invariant(num_vars=5, rounds=100, verbose=False):
    """
    端到端不变量(单池口径):重排 + CX 最优基发射后
    T == 池 ΣT、compute_tt == 真值表、词项级与 qiskit 级 CX 计数一致。
    """
    print(f'Testing emission T-invariance with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        entries = anf_pool(tt.bitstr, n)
        kept, standalone = fuse_pool_global(entries, n)

        qm = _qm(n)
        circ = MCXCircuit(qm)
        segs = [cb.to_circuit(qm) for cb, _, _ in kept] + [cb.to_circuit(qm) for cb in standalone]
        for seg_circ in order_segments_greedy(segs):
            circ.merge(seg_circ)
        circ.simplify_terms()

        assert count_t_cost(to_qiskit(circ)) == pool_t_cost(kept, standalone), '发射 T ≠ 池 ΣT'
        assert circ.compute_tt() == tt.bitstr, '发射语义破坏'
        assert count_cx_cost(to_qiskit(circ)) == circ.term_cx_cost(), 'CX 计数口径不一致'
    print('Emission T-invariance test passed!!!')


def test_order_segments_unit(num_vars=5, rounds=100, verbose=False):
    """order_segments_greedy 单元:经真实 merge 管线的输出 CX ≤ 原始顺序。"""
    print(f'Testing order_segments_greedy with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        qm = _qm(n)
        segs = [cb.to_circuit(qm) for cb, _, _ in anf_pool(tt.bitstr, n)[:12]]
        ordered = order_segments_greedy(segs)

        def total_cx(circs):
            scratch = MCXCircuit(qm)
            for c in circs:
                scratch.merge(c)
            scratch.simplify_terms()
            return scratch.term_cx_cost()

        assert total_cx(ordered) <= total_cx(segs), '重排不应劣于原始顺序'
    print('order_segments_greedy unit test passed!!!')


def main(verbose=False):
    test_plan_min_cx(5, 200, verbose)
    test_order_segments_unit(5, 100, verbose)
    test_emission_t_invariant(5, 100, verbose)
    print('====== All CX optimization tests passed!!! ======')


if __name__ == '__main__':
    main()
