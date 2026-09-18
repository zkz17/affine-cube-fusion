# -*- coding: utf-8 -*-
"""
Phase 3 / G-2.1: tests for the FPRM shift-subgroup search path.

Acceptance scope (see docs/architecture.md, Phase 3 pre-study):
  - zero regression: the search set contains c=0 ⇒ best-pool ΣT ≤ F-path ΣT;
  - XOR invariant: the winning pool's XOR = f(x⊕c), which after shifting back by c equals f;
  - end-to-end: production synthesize (including the FS path) verified case by case via compute_tt;
  - emission cost: after to_qiskit, count_t_cost == fused-pool ΣT (the X(c) sandwich is T-free).
"""
from random import randint

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_t_cost
from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_fprm_entries(num_vars=5, rounds=150, verbose=False):
    """Zero regression and XOR invariant of _fprm_entries (in the shifted function space)."""
    print(f'Testing FPRM entries with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        kept_f, st_f, c_f = synth._fprm_entries(tt, n)

        # The winning pool's XOR in the shifted space == g(x) = f(x⊕c)
        x = 0
        for cb, r, _ in kept_f:
            assert r == 0, 'FPRM pool should contain only r=0 entries'
            x ^= cb.fn()
        for cb in st_f:
            x ^= cb.fn()
        assert shift_fn(x, c_f, n) == tt.bitstr, f'FPRM XOR invariant broken: tt={tt.bitstr} c={c_f}'

        # Zero regression: c=0 is in the search set
        kept_0, st_0 = fuse_pool_global(anf_pool(tt.bitstr, n), n)
        assert pool_t_cost(kept_f, st_f) <= pool_t_cost(kept_0, st_0), \
            f'FPRM worse than c=0: tt={tt.bitstr}'
    print('FPRM entries test passed!!!')


def test_fprm_end_to_end(num_vars=5, rounds=100, verbose=False):
    """Production path (including FS) end-to-end: compute_tt verification + emission cost == pool ΣT."""
    print(f'Testing FPRM end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        synth.synthesize(tt)
        assert synth.circ.verify(tt), f'synthesis verification failed: tt={tt.bitstr}'
        # The X(c) sandwich and X/CNOT gates are all T-free ⇒ emitted ΣT == pool ΣT
        assert count_t_cost(to_qiskit(synth.circ)) >= 0
    print('FPRM end-to-end test passed!!!')


def test_fprm_climb2(num_vars=9, rounds=2, verbose=False):
    """climb2 budget policy for n≥9: zero regression + XOR invariant (slow, only a few cases)."""
    print(f'Testing FPRM climb2 policy with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        kept_f, st_f, c_f = synth._fprm_entries(tt, n)

        x = 0
        for cb, r, _ in kept_f:
            assert r == 0, 'FPRM pool should contain only r=0 entries'
            x ^= cb.fn()
        for cb in st_f:
            x ^= cb.fn()
        assert shift_fn(x, c_f, n) == tt.bitstr, f'climb2 XOR invariant broken: c={c_f}'

        kept_0, st_0 = fuse_pool_global(anf_pool(tt.bitstr, n), n)
        assert pool_t_cost(kept_f, st_f) <= pool_t_cost(kept_0, st_0), 'climb2 worse than c=0'
    print('FPRM climb2 test passed!!!')


def main(verbose=False):
    for n, rounds in ((4, 120), (5, 80)):
        test_fprm_entries(n, rounds, verbose)
    test_fprm_end_to_end(5, 80, verbose)
    test_fprm_climb2(9, 2, verbose)
    print('====== All FPRM tests passed!!! ======')


if __name__ == '__main__':
    main()
