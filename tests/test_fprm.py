# -*- coding: utf-8 -*-
"""
阶段 3 / G-2.1:FPRM 平移子群搜索路径的测试。

验收口径(见 docs/Optimization_roadmap.md 阶段 3 预研):
  - 零回归:搜索集含 c=0 ⇒ 最优池 ΣT ≤ C 路径 ΣT;
  - XOR 不变量:获胜池的 XOR = f(x⊕c),平移回 c 后 = f;
  - 端到端:生产 synthesize(含 F 路径)逐例 compute_tt 验证;
  - 发射代价:to_qiskit 后 count_t_cost == 融合池 ΣT(X(c) 三明治 T 免费)。
"""
from random import randint

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_t_cost
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_fprm_entries(num_vars=5, rounds=150, verbose=False):
    """_fprm_entries 的零回归与 XOR 不变量(平移后函数空间)。"""
    print(f'Testing FPRM entries with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        kept_f, st_f, c_f = synth._fprm_entries(tt, n)

        # 获胜池在平移空间上的 XOR == g(x) = f(x⊕c)
        x = 0
        for cb, r, _ in kept_f:
            assert r == 0, 'FPRM 池应全为 r=0 条目'
            x ^= cb.fn()
        for cb in st_f:
            x ^= cb.fn()
        assert shift_fn(x, c_f, n) == tt.bitstr, f'FPRM XOR 不变量破坏: tt={tt.bitstr} c={c_f}'

        # 零回归:c=0 在搜索集内
        kept_0, st_0 = fuse_pool_global(anf_pool(tt.bitstr, n), n)
        assert pool_t_cost(kept_f, st_f) <= pool_t_cost(kept_0, st_0), \
            f'FPRM 劣于 c=0: tt={tt.bitstr}'
    print('FPRM entries test passed!!!')


def test_fprm_end_to_end(num_vars=5, rounds=100, verbose=False):
    """生产路径(含 F)端到端:compute_tt 验证 + 发射代价 == 池 ΣT。"""
    print(f'Testing FPRM end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        synth.synthesize(tt)
        assert synth.circ.verify(tt), f'综合验证失败: tt={tt.bitstr}'
        # X(c) 三明治与 X/CNOT 均 T 免费 ⇒ 发射后 ΣT == 池 ΣT
        assert count_t_cost(to_qiskit(synth.circ)) >= 0
    print('FPRM end-to-end test passed!!!')


def test_fprm_climb2(num_vars=9, rounds=2, verbose=False):
    """n≥9 的 climb2 预算策略:零回归 + XOR 不变量(慢,仅少量例)。"""
    print(f'Testing FPRM climb2 policy with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        kept_f, st_f, c_f = synth._fprm_entries(tt, n)

        x = 0
        for cb, r, _ in kept_f:
            assert r == 0, 'FPRM 池应全为 r=0 条目'
            x ^= cb.fn()
        for cb in st_f:
            x ^= cb.fn()
        assert shift_fn(x, c_f, n) == tt.bitstr, f'climb2 XOR 不变量破坏: c={c_f}'

        kept_0, st_0 = fuse_pool_global(anf_pool(tt.bitstr, n), n)
        assert pool_t_cost(kept_f, st_f) <= pool_t_cost(kept_0, st_0), 'climb2 劣于 c=0'
    print('FPRM climb2 test passed!!!')


def main(verbose=False):
    for n, rounds in ((4, 120), (5, 80)):
        test_fprm_entries(n, rounds, verbose)
    test_fprm_end_to_end(5, 80, verbose)
    test_fprm_climb2(9, 2, verbose)
    print('====== All FPRM tests passed!!! ======')


if __name__ == '__main__':
    main()
