# -*- coding: utf-8 -*-
"""
G-3 metric test: the degree lower bound t(deg f).

Corollary D1 (docs/architecture.md, Lemma D): any branch-segment decomposition has ΣT ≥ t(deg f).
This test verifies:
  1. correctness of anf_degree (equal to the maximum popcount of the ANF monomial masks);
  2. invariance of deg under affine shifts (a necessary condition for the FPRM path);
  3. end-to-end: the emitted ΣT of the production backend really is ≥ t(deg f).
"""
from random import randint

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_t_cost
from backend.CompleteBranch import t_cost
from backend.fusion.BranchFusion import anf_degree, anf_pool, shift_fn
from backend.OracleSynthesizer import OracleSynthesizer


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_anf_degree(num_vars=5, rounds=200, verbose=False):
    """anf_degree == max popcount(mask); invariant under shifts."""
    print(f'Testing anf_degree with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        deg = anf_degree(tt.bitstr, n)
        ref = max((cb.mask.bit_count() for cb, _, _ in anf_pool(tt.bitstr, n)), default=0)
        assert deg == ref, f'anf_degree differs from the constructive formula: {deg} vs {ref}'

        # deg is invariant under the shift x ↦ x⊕c (FPRM path is valid)
        c = randint(0, (1 << n) - 1)
        assert anf_degree(shift_fn(tt.bitstr, c, n), n) == deg, f'deg shift invariance broken: c={c}'
    print('anf_degree test passed!!!')


def test_bound_holds(num_vars=5, rounds=60, verbose=False):
    """End-to-end: emitted circuit ΣT ≥ t(deg f)."""
    print(f'Testing degree bound end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        synth.synthesize(tt)
        t_emit = count_t_cost(to_qiskit(synth.circ))
        bound = t_cost(anf_degree(tt.bitstr, n))
        assert t_emit >= bound, f'lower bound violated: emit={t_emit} < t(deg f)={bound}'
        assert synth.circ.verify(tt)
    print('degree bound test passed!!!')


def main(verbose=False):
    test_anf_degree(5, 200, verbose)
    test_bound_holds(5, 60, verbose)
    print('====== All degree bound tests passed!!! ======')


if __name__ == '__main__':
    main()
