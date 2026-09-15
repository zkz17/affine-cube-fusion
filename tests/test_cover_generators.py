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
from random import randint

from utils.TruthTable import TruthTable
from backend.fusion.BranchFusion import fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.CoverGenerators import (minterm_entries, pk_entries,
                                            pk_proxy_best, pk_shift_best)


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


def main(verbose=False):
    for n, rounds in ((4, 60), (5, 40), (6, 20)):
        test_cover_generators_invariants(n, rounds, verbose)
    test_pk_negative_literals(verbose)
    test_minterm_engine_sanity(verbose)
    test_pk_shift_routes(4, 40, verbose)
    print('====== All cover generator tests passed!!! ======')
