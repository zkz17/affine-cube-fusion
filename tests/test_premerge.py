# -*- coding: utf-8 -*-
"""
Phase 1.2 (T-3): tests for the ANF premerge path for monomials differing in one variable.

Identity: x^S ⊕ x^{S∪{j}} = x^S·x̄_j (exact).
Acceptance scope (see docs/architecture.md, section 1.2):
  - pool XOR invariant (before/after premerge and after fusion);
  - every premerged product is a valid affine cube;
  - end-to-end synthesize verified case by case + fused ΣT statistics compared with the raw ANF pool
    (per-case non-worsening is structurally guaranteed by min(A, C, C′); here we only observe statistics).
"""
from collections import Counter
from random import randint

from utils.TruthTable import TruthTable
from backend.fusion.BranchFusion import (anf_pool, anf_pool_premerged, fuse_pool_global,
                                        pool_t_cost)
from backend.OracleSynthesizer import OracleSynthesizer


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def _pool_xor(entries, n):
    xor = 0
    for cb, r, _ in entries:
        assert r == 0
        xor ^= cb.fn()
    return xor


def _result_xor(kept, standalone):
    xor = 0
    for cb, _, _ in kept:
        xor ^= cb.fn()
    for cb in standalone:
        xor ^= cb.fn()
    return xor


def test_premerge_invariant(num_vars=6, rounds=200, verbose=False):
    """XOR invariant of the premerged pool and post-fusion invariant; pool size statistics."""
    print(f'Testing ANF premerge invariant with {rounds} random {num_vars}-input functions...')

    n = num_vars
    stat = Counter()
    for _ in range(rounds):
        tt = random_tt(n)
        raw = anf_pool(tt.bitstr, n)
        pre = anf_pool_premerged(tt.bitstr, n)

        assert _pool_xor(raw, n) == tt.bitstr, 'raw ANF pool XOR invariant broken'
        assert _pool_xor(pre, n) == tt.bitstr, f'premerged pool XOR invariant broken: tt={tt.bitstr}'
        assert len(pre) <= len(raw), 'premerge must not increase the pool size'
        # Premerged products are all wire-walk segments without ft_lims: symbolic self-check dim K = n − nctrl
        for cb, _, _ in pre:
            aff = cb.affine()
            assert len(aff.basis) == n - cb.nctrl(), f'affine representation dimension anomaly: {cb!r}'

        kept_r, st_r = fuse_pool_global(raw, n)
        kept_p, st_p = fuse_pool_global(pre, n)
        assert _result_xor(kept_r, st_r) == tt.bitstr, 'raw pool XOR invariant broken after fusion'
        assert _result_xor(kept_p, st_p) == tt.bitstr, 'premerged pool XOR invariant broken after fusion'

        tr, tp = pool_t_cost(kept_r, st_r), pool_t_cost(kept_p, st_p)
        stat['raw_size'] += len(raw)
        stat['pre_size'] += len(pre)
        stat['raw_cost'] += tr
        stat['pre_cost'] += tp
        if tp < tr: stat['pre_better'] += 1
        elif tp > tr: stat['pre_worse'] += 1
        else: stat['pre_same'] += 1

    print(f'  pool size: raw mean {stat["raw_size"] / rounds:.1f} → pre mean {stat["pre_size"] / rounds:.1f}')
    print(f'  fused ΣT: raw={stat["raw_cost"]} pre={stat["pre_cost"]}  '
          f'(pre better {stat["pre_better"]}, same {stat["pre_same"]}, worse {stat["pre_worse"]})')
    print('Premerge invariant test passed!!!')
    return stat


def test_premerge_end_to_end(num_vars=5, rounds=120, verbose=False):
    """End-to-end: synthesis verified case by case via compute_tt (through the min(A, C, C′) backend)."""
    print(f'Testing premerge end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        synth.synthesize(tt)
        assert synth.circ.verify(tt), f'synthesis verification failed: tt={tt.bitstr}'
    print('Premerge end-to-end test passed!!!')


def main(verbose=False):
    for n, rounds in ((4, 200), (5, 150), (6, 80)):
        test_premerge_invariant(n, rounds, verbose)
    test_premerge_end_to_end(5, 120, verbose)
    print('====== All premerge tests passed!!! ======')


if __name__ == '__main__':
    main()
