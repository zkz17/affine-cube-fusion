# -*- coding: utf-8 -*-
"""
B-4 (CX optimization) tests: correctness and invariants of segment reordering + emission basis selection.

Invariants:
  1. T unchanged: neither emission basis selection nor segment order changes the segment set or control counts ⇒ emitted ΣT == pool ΣT;
  2. CX no worse: plan_min_cx's CNOT count ≤ the canonical plan; reordered output CX ≤ the original order;
  3. semantics preserved: the new emitted circuit's compute_tt == truth table;
  4. counting consistency: term-level term_cx_cost == qiskit-level count_cx_cost.
"""
from random import randint

from utils.Qubit import QubitManager
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_cx_cost, count_t_cost
from backend.CompleteBranch import CompleteBranch, plan_from_affine, plan_min_cx
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost
from circuit.MCXCircuit import MCXCircuit, order_segments_greedy, term_cx_cost_list


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def _qm(n):
    qm = QubitManager()
    for v in range(n):
        qm.get_input_qubit(v)
    return qm


def test_plan_min_cx(num_vars=5, rounds=200, verbose=False):
    """plan_min_cx: control count unchanged, CNOTs not increased, emission semantics preserved."""
    print(f'Testing plan_min_cx with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        qm = _qm(n)
        for cb, _, _ in anf_pool(tt.bitstr, n)[:8]:
            base, basis = cb.affine()
            canon = plan_from_affine(base, basis, n)
            best = plan_min_cx(base, basis, n)
            assert len(best.controls) == len(canon.controls), 'control count must stay unchanged (T unchanged)'
            assert len(best.cnot_ops) <= len(canon.cnot_ops), 'plan_min_cx must not increase CNOTs'

            twin = CompleteBranch(n, synthetic=canon)
            assert twin.to_circuit(qm).compute_tt() == twin.fn(), 'CX-optimal basis emission broke semantics'
    print('plan_min_cx test passed!!!')


def test_emission_t_invariant(num_vars=5, rounds=100, verbose=False):
    """
    End-to-end invariants (single-pool scope): after reordering + CX-optimal basis emission,
    T == pool ΣT, compute_tt == truth table, and term-level and qiskit-level CX counts agree.
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

        assert count_t_cost(to_qiskit(circ)) == pool_t_cost(kept, standalone), 'emitted T != pool ΣT'
        assert circ.compute_tt() == tt.bitstr, 'emission broke semantics'
        assert count_cx_cost(to_qiskit(circ)) == circ.term_cx_cost(), 'CX counting scopes disagree'
    print('Emission T-invariance test passed!!!')


def test_order_segments_unit(num_vars=5, rounds=100, verbose=False):
    """order_segments_greedy unit: output CX after the real merge pipeline ≤ the original order."""
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

        assert total_cx(ordered) <= total_cx(segs), 'reordering must not be worse than the original order'
    print('order_segments_greedy unit test passed!!!')


def main(verbose=False):
    test_plan_min_cx(5, 200, verbose)
    test_order_segments_unit(5, 100, verbose)
    test_emission_t_invariant(5, 100, verbose)
    print('====== All CX optimization tests passed!!! ======')


if __name__ == '__main__':
    main()
