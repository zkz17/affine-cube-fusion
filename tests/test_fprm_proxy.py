# -*- coding: utf-8 -*-
"""
Plan A (fprm_proxy) tests: invariants of the proxy-prescreening path (n ≤ 7).

Invariants:
  1. zero regression vs F: the candidate set always contains c=0 ⇒ proxy result ΣT ≤ F-path ΣT;
  2. XOR invariant: the winning pool's XOR = f(x⊕c), which after shifting back by c equals f;
  3. determinism: the same input gives identical results across two runs;
  4. end-to-end: synthesis with fprm_proxy=True verified case by case via compute_tt.
(Exploration-phase discipline: fprm_proxy is a **choice relationship**, not a min with default FS, so no comparison assertion against FS is made;
  for the quality gap see the P1 probe: top-8 retention 71.7%~84.6%.)
"""
from random import randint

from utils.TruthTable import TruthTable
from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.FprmProxy import fprm_proxy_best


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_proxy_entries(num_vars=5, rounds=100, verbose=False):
    """Zero regression (vs F) and XOR invariant of the proxy candidates."""
    print(f'Testing fprm_proxy entries with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        kept_f, st_f, c_f = fprm_proxy_best(tt.bitstr, n)

        x = 0
        for cb, r, _ in kept_f:
            assert r == 0, 'proxy-path pool should contain only r=0 entries'
            x ^= cb.fn()
        for cb in st_f:
            x ^= cb.fn()
        assert shift_fn(x, c_f, n) == tt.bitstr, f'XOR invariant broken: tt={tt.bitstr} c={c_f}'

        kept_0, st_0 = fuse_pool_global(anf_pool(tt.bitstr, n), n)
        assert pool_t_cost(kept_f, st_f) <= pool_t_cost(kept_0, st_0), 'proxy worse than F path (c=0)'
    print('fprm_proxy entries test passed!!!')


def test_proxy_determinism(num_vars=5, rounds=30, verbose=False):
    """Determinism: two calls return identical results."""
    print(f'Testing fprm_proxy determinism with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        k1, s1, c1 = fprm_proxy_best(tt.bitstr, n)
        k2, s2, c2 = fprm_proxy_best(tt.bitstr, n)
        assert c1 == c2, 'candidate shifts differ'
        assert [cb for cb, _, _ in k1] == [cb for cb, _, _ in k2], 'kept entries differ'
        assert s1 == s2, 'standalone entries differ'
    print('fprm_proxy determinism test passed!!!')


def test_proxy_end_to_end(num_vars=5, rounds=60, verbose=False):
    """End-to-end verify of the production path (fprm_proxy=True)."""
    print(f'Testing fprm_proxy end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    s = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, fprm_proxy=True)
    for _ in range(rounds):
        tt = random_tt(n)
        s.synthesize(tt)
        assert s.circ.verify(tt), f'synthesis verification failed: tt={tt.bitstr}'
    print('fprm_proxy end-to-end test passed!!!')


def main(verbose=False):
    for n, rounds in ((4, 100), (5, 80)):
        test_proxy_entries(n, rounds, verbose)
    test_proxy_determinism(5, 30, verbose)
    test_proxy_end_to_end(5, 60, verbose)
    print('====== All fprm_proxy tests passed!!! ======')


if __name__ == '__main__':
    main()
