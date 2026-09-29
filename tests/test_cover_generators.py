# -*- coding: utf-8 -*-
"""
Cover generator tests for the starting-cover route comparison (n <= 7 only).

Invariants:
  1. XOR invariant: for both generators, the XOR of the flip functions of the pool
     equals f (minterm pool by definition; PK pool by the expansion identity);
  2. determinism: two calls produce identical pools;
  3. structure: the minterm pool has popcount(f) entries, each an n-control point;
     on f = x̄1 x̄2 x̄3 (single point at x = 000) the PK pool contains exactly the
     negative-polarity cube x̄1 x̄2 x̄3, i.e. one block of cost t(3);
  4. engine sanity: fusing a two-point minterm pool yields a single line block.
"""
from contextlib import ExitStack
from random import Random, randint
from unittest.mock import patch

from utils.TruthTable import TruthTable
from backend.fusion.BranchFusion import fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.CoverGenerators import (minterm_entries, pk_entries,
                                            pk_proxy_best, pk_shift_best)
from backend.OracleSynthesizer import OracleSynthesizer
from config import config
from evaluate import count_costs
from utils.to_qiskit import to_qiskit


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def pool_xor(entries, n):
    x = 0
    for cb, r, _ in entries:
        x ^= shift_fn(cb.fn(), r, n)
    return x


def test_cover_generators_invariants(num_vars=5, rounds=80, verbose=False):
    """XOR invariant and determinism for both generators."""
    print(f'Testing cover generator invariants with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        for name, gen in (('minterm', minterm_entries), ('pk', pk_entries)):
            p1 = gen(tt.bitstr, n)
            p2 = gen(tt.bitstr, n)
            assert p1 == p2, f'{name} pool not deterministic'
            assert pool_xor(p1, n) == tt.bitstr, f'{name} pool XOR invariant violated'
            if name == 'minterm':
                assert len(p1) == tt.bitstr.bit_count(), 'minterm pool size mismatch'
                assert all(cb.nctrl() == n for cb, _, _ in p1), 'minterm block not a point'
    print('Cover generator invariants test passed!!!')


def test_pk_negative_literals(verbose=False):
    """f = x̄1 x̄2 x̄3: the PK pool must contain the single cube x̄1 x̄2 x̄3 directly."""
    print('Testing PK negative-polarity support...')

    n, fn = 3, 1        # supp = {000}
    pool = pk_entries(fn, n)
    assert len(pool) == 1, f'expected one cube, got {len(pool)}'
    cb, r, _ = pool[0]
    assert cb.nctrl() == 3 and cb.t_cost() == 15, f'unexpected block: ctrl={cb.nctrl()}'
    assert pool_xor(pool, n) == fn
    print('PK negative-literals test passed!!!')


def test_minterm_engine_sanity(verbose=False):
    """Two-point minterm pool fuses into one line block."""
    print('Testing minterm pool fusion sanity...')

    n, fn = 3, (1 << 0) | (1 << 7)        # supp = {000, 111}
    pool = minterm_entries(fn, n)
    assert len(pool) == 2
    kept, st = fuse_pool_global(pool, n)
    assert pool_t_cost(kept, st) == 7, f'expected a line (cost 7), got {pool_t_cost(kept, st)}'
    print('Minterm engine sanity test passed!!!')


def test_pk_shift_routes(num_vars=4, rounds=40, verbose=False):
    """PKS/PKP: XOR invariant after shift restore, no-regression vs PK (c=0 included), determinism."""
    print(f'Testing PKS/PKP routes with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        base = pool_t_cost(*fuse_pool_global(pk_entries(tt.bitstr, n), n))

        k1, s1, c1 = pk_shift_best(tt.bitstr, n)
        k2, s2, c2 = pk_shift_best(tt.bitstr, n)
        assert c1 == c2 and pool_t_cost(k1, s1) == pool_t_cost(k2, s2), 'PKS not deterministic'
        xor1 = pool_xor([(cb, 0, None) for cb, _, _ in k1] + [(cb, 0, None) for cb in s1], n)
        assert shift_fn(xor1, c1, n) == tt.bitstr, 'PKS pool XOR invariant violated'
        assert pool_t_cost(k1, s1) <= base, 'PKS worse than PK at c=0'

        kp, sp, cp = pk_proxy_best(tt.bitstr, n)
        xor2 = pool_xor([(cb, 0, None) for cb, _, _ in kp] + [(cb, 0, None) for cb in sp], n)
        assert shift_fn(xor2, cp, n) == tt.bitstr, 'PKP pool XOR invariant violated'
        assert pool_t_cost(kp, sp) <= base, 'PKP worse than PK at c=0'
    print('PKS/PKP route test passed!!!')


def test_raw_anf_emission(verbose=False):
    """F0 implements f exactly and charges each ANF monomial by its literal count.

    Compute coefficients independently by the subset-parity definition, rather than
    reusing the fast transform or the pool whose costs we are checking.
    """
    print('Testing raw ANF emission (all 3-input functions and boundary cases)...')
    monomials = [sum(1 << x for x in range(32) if x & mask == mask)
                 for mask in (1, 3, 7, 31)]
    cases = [(0, (0, 1)), (3, range(256)), (5, [0, (1 << 32) - 1, 1] + monomials)]
    for n, functions in cases:
        for fn in functions:
            expected = [0, 0]
            for mask in range(1 << n):
                coefficient, submask = 0, mask
                while True:
                    coefficient ^= (fn >> submask) & 1
                    if submask == 0:
                        break
                    submask = (submask - 1) & mask
                if coefficient:
                    expected[0] += config['t_cost'](mask.bit_count())
                    expected[1] += config['cx_cost'](mask.bit_count())
            for m2 in (False, True):
                s = OracleSynthesizer(ancilla_num=0, m2_post=m2)
                s.allocate_input_qubit(n)
                tt = TruthTable(fn, n)
                s.process_cover_route(tt, 'F0')
                assert s.circ.verify(tt), f'F0 semantics: n={n}, tt={fn}'
                assert count_costs(to_qiskit(s.circ)) == tuple(expected), (n, fn, expected)
                assert not s.qubit_mgr.anc, 'F0 allocated an ancilla'
    print('Raw ANF emission test passed!!!')


def test_raw_dd_emission(verbose=False):
    """A0 preserves the preprocessed DD cover, including root masks and split pieces.

    Exhaust all 3-input functions and include larger deterministic samples. Guards
    make accidentally enabling any post-DD optimization fail, even if semantics
    would still be correct. Reuse each synthesizer to exercise batch state clearing.
    """
    print('Testing raw DD emission, DD reorder, and disabled post-DD optimizations...')
    rng = Random(20260923)
    cases = [(3, range(256)),
             (5, [0, (1 << 32) - 1, 1] + [rng.getrandbits(32) for _ in range(16)]),
             (7, [rng.getrandbits(128) for _ in range(8)])]
    seen_root_mask = seen_split_pieces = False
    costs_without_m2 = {}
    with ExitStack() as guards:
        for name in ('fuse_pool_global', 'm2_interleaved', 'order_segments_greedy'):
            guards.enter_context(patch(
                f'backend.OracleSynthesizer.{name}',
                side_effect=AssertionError(f'A0 called disabled optimization {name}')))
        for reorder in (False, True):
            for m2 in (False, True):
                s = OracleSynthesizer(ancilla_num=0, reorder=reorder, m2_post=m2)
                extract = s._bdd_entries

                def checked_entries(tt, path):
                    nonlocal seen_root_mask, seen_split_pieces
                    entries = extract(tt, path)
                    assert pool_xor(entries, tt.num_var) == tt.bitstr, 'DD cover XOR invariant'
                    seen_root_mask |= any(r for _, r, _ in entries)
                    seen_split_pieces |= len({idx for _, _, idx in entries}) > 1
                    return entries

                with patch.object(s, '_bdd_entries', side_effect=checked_entries), \
                        patch.object(s, 'variable_reorder', wraps=s.variable_reorder) as dd_reorder:
                    for n, functions in cases:
                        for fn in functions:
                            s.clear()
                            s.allocate_input_qubit(n)
                            tt = TruthTable(fn, n)
                            dd_reorder.reset_mock()
                            s.process_cover_route(tt, 'A0')
                            assert dd_reorder.call_count == int(reorder), 'DD reorder switch ignored'
                            assert s.circ.verify(tt), ('A0 semantics', reorder, m2, n, fn)
                            assert not s.qubit_mgr.anc, 'A0 allocated an ancilla'
                            costs = count_costs(to_qiskit(s.circ))
                            key = (reorder, n, fn)
                            if not m2:
                                costs_without_m2[key] = costs
                            else:
                                assert costs == costs_without_m2[key], 'A0 depends on M2 setting'
    assert seen_root_mask, 'samples did not exercise DD root-mask preservation'
    assert seen_split_pieces, 'samples did not exercise multiple DD pieces'
    print('Raw DD emission test passed!!!')


def main(verbose=False):
    for n, rounds in ((4, 60), (5, 40), (6, 20)):
        test_cover_generators_invariants(n, rounds, verbose)
    test_pk_negative_literals(verbose)
    test_minterm_engine_sanity(verbose)
    test_pk_shift_routes(4, 40, verbose)
    test_raw_anf_emission(verbose)
    test_raw_dd_emission(verbose)
    print('====== All cover generator tests passed!!! ======')
