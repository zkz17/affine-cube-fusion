# -*- coding: utf-8 -*-
"""
G-2.2 tests for the optional GL search path (only n ≤ 7, per the testing discipline).

Invariants:
  1. zero regression: emitted T with gl_search=True ≤ emitted T with the default (False)
     (the G candidates contain the FPRM-optimal cF itself ⇒ structurally guaranteed; verified case by case here);
  2. determinism: synthesizing the same truth table twice yields identical emitted T/CX (no randomness);
  3. semantics: the emitted circuit passes compute_tt case by case;
  4. counting consistency: term-level CX == qiskit-level CX (including the wrapper CNOT when G wins).
"""
from random import randint

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_costs
from backend.OracleSynthesizer import OracleSynthesizer


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_gl_zero_regression(num_vars=5, rounds=40, verbose=False):
    """With gl enabled, T is never worse than the default path, case by case.

    Note (2025-08-22): m2_post is enabled by default; this assertion keeps the **structural scope** —
    the comparison side uses m2_post=False explicitly (G candidates contain the FPRM-optimal cF ⇒ structural guarantee).
    G_on ≤ F_on with m2_post enabled on both sides is empirical (P9-M sample 46/46), not a constraint.
    """
    print(f'Testing GL zero-regression with {rounds} random {num_vars}-input functions...')

    n = num_vars
    off = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=False,
                                  m2_post=False)
    on = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=True)
    better = same = 0
    for _ in range(rounds):
        tt = random_tt(n)
        off.synthesize(tt)
        t_off, cx_off = count_costs(to_qiskit(off.circ))
        assert off.circ.verify(tt)

        on.synthesize(tt)
        t_on, cx_on = count_costs(to_qiskit(on.circ))
        assert on.circ.verify(tt), f'GL path verify failed: tt={tt.bitstr}'
        assert t_on <= t_off, f'GL worse than default: {t_on} > {t_off} tt={tt.bitstr}'
        better += t_on < t_off
        same += t_on == t_off
    print(f'  better={better} same={same} (T zero regression holds)')
    print('GL zero-regression test passed!!!')


def test_gl_determinism(num_vars=5, rounds=20, verbose=False):
    """Determinism: two synthesis runs produce bit-identical outputs."""
    print(f'Testing GL determinism with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        s1 = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=True)
        s2 = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=True)
        s1.synthesize(tt)
        s2.synthesize(tt)
        assert s1.circ.tt == s2.circ.tt, 'truth tables differ'
        c1 = count_costs(to_qiskit(s1.circ))
        c2 = count_costs(to_qiskit(s2.circ))
        assert c1 == c2, f'emission costs differ: {c1} vs {c2}'
    print('GL determinism test passed!!!')


def test_gl_counting_consistency(num_vars=5, rounds=20, verbose=False):
    """Counting consistency including the wrapper CNOT when G wins (term-level == qiskit-level)."""
    print(f'Testing GL counting consistency with {rounds} random {num_vars}-input functions...')

    n = num_vars
    s = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=True)
    for _ in range(rounds):
        tt = random_tt(n)
        s.synthesize(tt)
        qc = to_qiskit(s.circ)
        assert count_costs(qc)[1] == s.circ.term_cx_cost(), 'CX counting scopes disagree'
    print('GL counting consistency test passed!!!')


def main(verbose=False):
    test_gl_zero_regression(5, 40, verbose)
    test_gl_determinism(5, 20, verbose)
    test_gl_counting_consistency(5, 20, verbose)
    test_gl_zero_regression(6, 10, verbose)
    print('====== All GL search tests passed!!! ======')


if __name__ == '__main__':
    main()
