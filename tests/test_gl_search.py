# -*- coding: utf-8 -*-
"""
G-2.2 可选 GL 搜索路径的测试(仅 n ≤ 7,遵循测试纪律)。

不变量:
  1. 零回归:gl_search=True 的发射 T ≤ 默认(False)的发射 T
     (G 候选含 FPRM 最优 cF 本身 ⇒ 结构性保证,这里逐例实证);
  2. 确定性:同一真值表两次综合,发射 T/CX 完全相同(无随机);
  3. 语义:发射电路 compute_tt 逐例通过;
  4. 计数一致性:词项级 CX == qiskit 级 CX(G 获胜时含 wrapper CNOT)。
"""
from random import randint

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_costs
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_gl_zero_regression(num_vars=5, rounds=40, verbose=False):
    """gl 开启的 T 逐例不劣于默认路径。

    注(2025-08-22):m2_post 已默认开启;本断言保持**结构性口径**——
    对照侧显式 m2_post=False(G 候选含 FPRM 最优 cF ⇒ 结构性保证)。
    m2_post 两侧同开时的 G_on ≤ F_on 为经验性(P9-M 样本 46/46),非约束。
    """
    print(f'Testing GL zero-regression with {rounds} random {num_vars}-input functions...')

    n = num_vars
    off = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=False,
                                  m2_post=False)
    on = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=True)
    better = same = 0
    for _ in range(rounds):
        tt = random_tt(n)
        off.synthesize(tt)
        t_off, cx_off = count_costs(to_qiskit(off.circ))
        assert off.circ.verify(tt)

        on.synthesize(tt)
        t_on, cx_on = count_costs(to_qiskit(on.circ))
        assert on.circ.verify(tt), f'GL 路径 verify 失败: tt={tt.bitstr}'
        assert t_on <= t_off, f'GL 劣于默认: {t_on} > {t_off} tt={tt.bitstr}'
        better += t_on < t_off
        same += t_on == t_off
    print(f'  better={better} same={same} (T 零回归成立)')
    print('GL zero-regression test passed!!!')


def test_gl_determinism(num_vars=5, rounds=20, verbose=False):
    """确定性:两次综合输出逐位相同。"""
    print(f'Testing GL determinism with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        s1 = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=True)
        s2 = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=True)
        s1.synthesize(tt)
        s2.synthesize(tt)
        assert s1.circ.tt == s2.circ.tt, '真值表不一致'
        c1 = count_costs(to_qiskit(s1.circ))
        c2 = count_costs(to_qiskit(s2.circ))
        assert c1 == c2, f'发射代价不一致: {c1} vs {c2}'
    print('GL determinism test passed!!!')


def test_gl_counting_consistency(num_vars=5, rounds=20, verbose=False):
    """G 获胜时含 wrapper CNOT 的计数一致性(词项级 == qiskit 级)。"""
    print(f'Testing GL counting consistency with {rounds} random {num_vars}-input functions...')

    n = num_vars
    s = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, gl_search=True)
    for _ in range(rounds):
        tt = random_tt(n)
        s.synthesize(tt)
        qc = to_qiskit(s.circ)
        assert count_costs(qc)[1] == s.circ.term_cx_cost(), 'CX 计数口径不一致'
    print('GL counting consistency test passed!!!')


def main(verbose=False):
    test_gl_zero_regression(5, 40, verbose)
    test_gl_determinism(5, 20, verbose)
    test_gl_counting_consistency(5, 20, verbose)
    test_gl_zero_regression(6, 10, verbose)
    print('====== All GL search tests passed!!! ======')


if __name__ == '__main__':
    main()
