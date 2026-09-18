# -*- coding: utf-8 -*-
"""
Macro M2 interleaved post-processing tests (only n ≤ 7, per the testing discipline).

Invariants:
  1. module-level: XOR invariant, T monotonicity (m2_interleaved output ≤ input pool), determinism,
     and **differential agreement** between dim-bucket prescreening and full scan (same optimal total gain);
  2. end-to-end (five configs A/F/FS/FP/FG): emitted T with m2_post=True ≤ the same config with
     m2_post=False case by case (the macro only makes strictly improving moves ⇒ zero regression), and emitted compute_tt passes;
  3. determinism: with m2_post=True, two synthesis runs have bit-identical T/CX.
"""
from random import randint

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_costs
from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.MacroM2 import m2_best_move, m2_interleaved


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def pool_xor(kept, standalone, n):
    x = 0
    for cb, r, _ in kept:
        x ^= shift_fn(cb.fn(), r, n)
    for cb in standalone:
        x ^= cb.fn()
    return x


def test_m2_module_invariants(num_vars=5, rounds=60, verbose=False):
    """Module-level: XOR invariant, T monotonicity, determinism, prescreening differential agreement."""
    print(f'Testing m2 module invariants with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        entries = anf_pool(tt.bitstr, n)
        kept, standalone = fuse_pool_global(entries, n)
        assert pool_xor(kept, standalone, n) == tt.bitstr, 'fused-pool XOR invariant broken'

        mv_full = m2_best_move([(cb, r, k) for cb, r, k in kept] +
                               [(cb, 0, None) for cb in standalone], n, prescreen=False)
        mv_buck = m2_best_move([(cb, r, k) for cb, r, k in kept] +
                               [(cb, 0, None) for cb in standalone], n, prescreen=True)
        assert (mv_full is None) == (mv_buck is None), 'prescreening and full scan disagree on existence'
        if mv_full is not None:
            assert mv_full[0] == mv_buck[0], \
                f'prescreening differential failed: full={mv_full[0]} buck={mv_buck[0]} tt={tt.bitstr}'

        base = pool_t_cost(kept, standalone)
        k1, s1 = m2_interleaved(kept, standalone, n)
        k2, s2 = m2_interleaved(kept, standalone, n)
        assert pool_xor(k1, s1, n) == tt.bitstr, 'post-M2 pool XOR invariant broken'
        assert pool_t_cost(k1, s1) <= base, 'M2 output T worse than the input pool'
        assert pool_t_cost(k2, s2) == pool_t_cost(k1, s1), 'M2 output is nondeterministic'
    print('m2 module invariants test passed!!!')


def _configs():
    return [('FS', False, False), ('A', False, False), ('F', False, False),
            ('FP', False, True), ('FG', True, False)]


def test_m2_post_end_to_end(num_vars=5, rounds=30, verbose=False):
    """End-to-end: zero regression of T with m2_post=True across five configs, case by case, plus semantic verification."""
    print(f'Testing m2_post end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for cfg, gl, proxy in _configs():
        off = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                      gl_search=gl, fprm_proxy=proxy,
                                      paths=('A',) if cfg == 'A' else
                                            ('F',) if cfg == 'F' else ('FS',),
                                      m2_post=False)
        on = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                     gl_search=gl, fprm_proxy=proxy,
                                     paths=('A',) if cfg == 'A' else
                                           ('F',) if cfg == 'F' else ('FS',),
                                     m2_post=True)
        better = same = 0
        for _ in range(rounds):
            tt = random_tt(n)
            off.synthesize(tt)
            t_off, cx_off = count_costs(to_qiskit(off.circ))
            assert off.circ.verify(tt)

            on.synthesize(tt)
            t_on, cx_on = count_costs(to_qiskit(on.circ))
            assert on.circ.verify(tt), f'[{cfg}] M2 emission verification failed: tt={tt.bitstr}'
            assert t_on <= t_off, f'[{cfg}] M2 worse than the same-config baseline: {t_on} > {t_off} tt={tt.bitstr}'
            better += t_on < t_off
            same += t_on == t_off
        print(f'  [{cfg}] T better={better} same={same}/{rounds} (zero regression holds)')
    print('m2_post end-to-end test passed!!!')


def test_m2_post_determinism(num_vars=5, rounds=20, verbose=False):
    """Determinism: with m2_post=True, two synthesis runs have bit-identical T/CX."""
    print(f'Testing m2_post determinism with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for cfg, gl, proxy in _configs():
        s1 = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                     gl_search=gl, fprm_proxy=proxy,
                                     paths=('A',) if cfg == 'A' else
                                           ('F',) if cfg == 'F' else ('FS',),
                                     m2_post=True)
        s2 = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                     gl_search=gl, fprm_proxy=proxy,
                                     paths=('A',) if cfg == 'A' else
                                           ('F',) if cfg == 'F' else ('FS',),
                                     m2_post=True)
        for _ in range(rounds):
            tt = random_tt(n)
            s1.synthesize(tt)
            s2.synthesize(tt)
            assert s1.circ.tt == s2.circ.tt, f'[{cfg}] truth tables differ'
            c1 = count_costs(to_qiskit(s1.circ))
            c2 = count_costs(to_qiskit(s2.circ))
            assert c1 == c2, f'[{cfg}] emission costs differ: {c1} vs {c2}'
    print('m2_post determinism test passed!!!')


def main(verbose=False):
    for n, rounds in ((4, 60), (5, 40)):
        test_m2_module_invariants(n, rounds, verbose)
    for n, rounds in ((4, 30), (5, 20), (6, 10)):
        test_m2_post_end_to_end(n, rounds, verbose)
    test_m2_post_determinism(5, 20, verbose)
    print('====== All macro M2 tests passed!!! ======')
