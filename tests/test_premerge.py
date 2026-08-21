# -*- coding: utf-8 -*-
"""
阶段 1.2(T-3):ANF 差一变量单项式预合并路径的测试。

恒等式:x^S ⊕ x^{S∪{j}} = x^S·x̄_j(精确)。
验收口径(见 docs/Optimization_roadmap.md 1.2 节):
  - 池 XOR 不变量(预合并前/后、融合后);
  - 每条预合并产物是合法仿射立方;
  - 端到端 synthesize 逐例验证 + 相比原始 ANF 池的融合 ΣT 统计
    (逐例不劣由 min(A, C, C′) 结构性保证,这里只做统计观察)。
"""
from collections import Counter
from random import randint

from utils.TruthTable import TruthTable
from limbdd.fusion.BranchFusion import (anf_pool, anf_pool_premerged, fuse_pool_global,
                                        pool_t_cost)
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer


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
    """预合并池的 XOR 不变量与融合后不变量;池规模统计。"""
    print(f'Testing ANF premerge invariant with {rounds} random {num_vars}-input functions...')

    n = num_vars
    stat = Counter()
    for _ in range(rounds):
        tt = random_tt(n)
        raw = anf_pool(tt.bitstr, n)
        pre = anf_pool_premerged(tt.bitstr, n)

        assert _pool_xor(raw, n) == tt.bitstr, '原始 ANF 池 XOR 不变量被破坏'
        assert _pool_xor(pre, n) == tt.bitstr, f'预合并池 XOR 不变量被破坏: tt={tt.bitstr}'
        assert len(pre) <= len(raw), '预合并不应增加池规模'
        # 预合并产物均为无 ft_lims 的走线段:符号表示自检 dim K = n − nctrl
        for cb, _, _ in pre:
            aff = cb.affine()
            assert len(aff.basis) == n - cb.nctrl(), f'仿射表示维数异常: {cb!r}'

        kept_r, st_r = fuse_pool_global(raw, n)
        kept_p, st_p = fuse_pool_global(pre, n)
        assert _result_xor(kept_r, st_r) == tt.bitstr, '原始池融合后 XOR 不变量被破坏'
        assert _result_xor(kept_p, st_p) == tt.bitstr, '预合并池融合后 XOR 不变量被破坏'

        tr, tp = pool_t_cost(kept_r, st_r), pool_t_cost(kept_p, st_p)
        stat['raw_size'] += len(raw)
        stat['pre_size'] += len(pre)
        stat['raw_cost'] += tr
        stat['pre_cost'] += tp
        if tp < tr: stat['pre_better'] += 1
        elif tp > tr: stat['pre_worse'] += 1
        else: stat['pre_same'] += 1

    print(f'  池规模: raw 均值 {stat["raw_size"] / rounds:.1f} → pre 均值 {stat["pre_size"] / rounds:.1f}')
    print(f'  融合 ΣT: raw={stat["raw_cost"]} pre={stat["pre_cost"]}  '
          f'(pre 更优 {stat["pre_better"]}, 相同 {stat["pre_same"]}, 更差 {stat["pre_worse"]})')
    print('Premerge invariant test passed!!!')
    return stat


def test_premerge_end_to_end(num_vars=5, rounds=120, verbose=False):
    """端到端:合成逐例 compute_tt 验证(经 min(A, C, C′) 后端)。"""
    print(f'Testing premerge end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    for _ in range(rounds):
        tt = random_tt(n)
        synth.synthesize(tt)
        assert synth.circ.verify(tt), f'综合验证失败: tt={tt.bitstr}'
    print('Premerge end-to-end test passed!!!')


def main(verbose=False):
    for n, rounds in ((4, 200), (5, 150), (6, 80)):
        test_premerge_invariant(n, rounds, verbose)
    test_premerge_end_to_end(5, 120, verbose)
    print('====== All premerge tests passed!!! ======')


if __name__ == '__main__':
    main()
