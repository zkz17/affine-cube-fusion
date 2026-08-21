# -*- coding: utf-8 -*-
"""
G-3 度量测试:度数下界 t(deg f)。

推论 D1(docs/Optimization_roadmap.md 引理 D):任意分支段分解的 ΣT ≥ t(deg f)。
本测试验证:
  1. anf_degree 的正确性(等于 ANF 单项式掩码的最大 popcount);
  2. deg 在仿射平移下的不变性(FPRM 路径合法性的必要条件);
  3. 端到端:生产后端的发射 ΣT 确实 ≥ t(deg f)。
"""
from random import randint

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_t_cost
from limbdd.CompleteBranch import t_cost
from limbdd.fusion.BranchFusion import anf_degree, anf_pool, shift_fn
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_anf_degree(num_vars=5, rounds=200, verbose=False):
    """anf_degree == max popcount(mask);且在平移下不变。"""
    print(f'Testing anf_degree with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        deg = anf_degree(tt.bitstr, n)
        ref = max((cb.mask.bit_count() for cb, _, _ in anf_pool(tt.bitstr, n)), default=0)
        assert deg == ref, f'anf_degree 与构造式不一致: {deg} vs {ref}'

        # deg 在平移 x ↦ x⊕c 下不变(FPRM 路径合法)
        c = randint(0, (1 << n) - 1)
        assert anf_degree(shift_fn(tt.bitstr, c, n), n) == deg, f'deg 平移不变性破坏: c={c}'
    print('anf_degree test passed!!!')


def test_bound_holds(num_vars=5, rounds=60, verbose=False):
    """端到端:发射电路 ΣT ≥ t(deg f)。"""
    print(f'Testing degree bound end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        synth.synthesize(tt)
        t_emit = count_t_cost(to_qiskit(synth.circ))
        bound = t_cost(anf_degree(tt.bitstr, n))
        assert t_emit >= bound, f'下界被违反: emit={t_emit} < t(deg f)={bound}'
        assert synth.circ.verify(tt)
    print('degree bound test passed!!!')


def main(verbose=False):
    test_anf_degree(5, 200, verbose)
    test_bound_holds(5, 60, verbose)
    print('====== All degree bound tests passed!!! ======')


if __name__ == '__main__':
    main()
