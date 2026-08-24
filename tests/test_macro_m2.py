# -*- coding: utf-8 -*-
"""
宏 M2 相间后处理测试(仅 n ≤ 7,遵循测试纪律)。

不变量:
  1. 模块级:XOR 不变量、T 单调(m2_interleaved 输出 ≤ 输入池)、确定性、
     以及 dim 分桶预筛 vs 全扫描的**差分一致**(最优总收益相同);
  2. 端到端(五配置 A/C/F/FP/G):m2_post=True 的发射 T 逐例 ≤ 同配置
     m2_post=False(宏只做严格下降移动 ⇒ 零回归),发射 compute_tt 通过;
  3. 确定性:m2_post=True 两次综合 T/CX 逐位相同。
"""
from random import randint

from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_costs
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from limbdd.fusion.MacroM2 import m2_best_move, m2_interleaved


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
    """模块级:XOR 不变量、T 单调、确定性、预筛差分一致。"""
    print(f'Testing m2 module invariants with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        entries = anf_pool(tt.bitstr, n)
        kept, standalone = fuse_pool_global(entries, n)
        assert pool_xor(kept, standalone, n) == tt.bitstr, '融合池 XOR 不变量破坏'

        mv_full = m2_best_move([(cb, r, k) for cb, r, k in kept] +
                               [(cb, 0, None) for cb in standalone], n, prescreen=False)
        mv_buck = m2_best_move([(cb, r, k) for cb, r, k in kept] +
                               [(cb, 0, None) for cb in standalone], n, prescreen=True)
        assert (mv_full is None) == (mv_buck is None), '预筛与全扫描的存在性不一致'
        if mv_full is not None:
            assert mv_full[0] == mv_buck[0], \
                f'预筛差分失败: full={mv_full[0]} buck={mv_buck[0]} tt={tt.bitstr}'

        base = pool_t_cost(kept, standalone)
        k1, s1 = m2_interleaved(kept, standalone, n)
        k2, s2 = m2_interleaved(kept, standalone, n)
        assert pool_xor(k1, s1, n) == tt.bitstr, 'M2 后池 XOR 不变量破坏'
        assert pool_t_cost(k1, s1) <= base, 'M2 输出 T 劣于输入池'
        assert pool_t_cost(k2, s2) == pool_t_cost(k1, s1), 'M2 输出不确定'
    print('m2 module invariants test passed!!!')


def _configs():
    return [('F', False, False), ('A', False, False), ('C', False, False),
            ('FP', False, True), ('G', True, False)]


def test_m2_post_end_to_end(num_vars=5, rounds=30, verbose=False):
    """端到端:五配置下 m2_post=True 的 T 逐例零回归 + 语义验证。"""
    print(f'Testing m2_post end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for cfg, gl, proxy in _configs():
        off = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                      gl_search=gl, fprm_proxy=proxy,
                                      paths=('A',) if cfg == 'A' else
                                            ('C',) if cfg == 'C' else ('F',),
                                      m2_post=False)
        on = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                     gl_search=gl, fprm_proxy=proxy,
                                     paths=('A',) if cfg == 'A' else
                                           ('C',) if cfg == 'C' else ('F',),
                                     m2_post=True)
        better = same = 0
        for _ in range(rounds):
            tt = random_tt(n)
            off.synthesize(tt)
            t_off, cx_off = count_costs(to_qiskit(off.circ))
            assert off.circ.verify(tt)

            on.synthesize(tt)
            t_on, cx_on = count_costs(to_qiskit(on.circ))
            assert on.circ.verify(tt), f'[{cfg}] M2 发射验证失败: tt={tt.bitstr}'
            assert t_on <= t_off, f'[{cfg}] M2 劣于同配置基线: {t_on} > {t_off} tt={tt.bitstr}'
            better += t_on < t_off
            same += t_on == t_off
        print(f'  [{cfg}] T better={better} same={same}/{rounds} (零回归成立)')
    print('m2_post end-to-end test passed!!!')


def test_m2_post_determinism(num_vars=5, rounds=20, verbose=False):
    """确定性:m2_post=True 两次综合 T/CX 逐位相同。"""
    print(f'Testing m2_post determinism with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for cfg, gl, proxy in _configs():
        s1 = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                     gl_search=gl, fprm_proxy=proxy,
                                     paths=('A',) if cfg == 'A' else
                                           ('C',) if cfg == 'C' else ('F',),
                                     m2_post=True)
        s2 = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True,
                                     gl_search=gl, fprm_proxy=proxy,
                                     paths=('A',) if cfg == 'A' else
                                           ('C',) if cfg == 'C' else ('F',),
                                     m2_post=True)
        for _ in range(rounds):
            tt = random_tt(n)
            s1.synthesize(tt)
            s2.synthesize(tt)
            assert s1.circ.tt == s2.circ.tt, f'[{cfg}] 真值表不一致'
            c1 = count_costs(to_qiskit(s1.circ))
            c2 = count_costs(to_qiskit(s2.circ))
            assert c1 == c2, f'[{cfg}] 发射代价不一致: {c1} vs {c2}'
    print('m2_post determinism test passed!!!')


def main(verbose=False):
    for n, rounds in ((4, 60), (5, 40)):
        test_m2_module_invariants(n, rounds, verbose)
    for n, rounds in ((4, 30), (5, 20), (6, 10)):
        test_m2_post_end_to_end(n, rounds, verbose)
    test_m2_post_determinism(5, 20, verbose)
    print('====== All macro M2 tests passed!!! ======')
