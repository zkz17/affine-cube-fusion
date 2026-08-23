# -*- coding: utf-8 -*-
"""
方案 A(fprm_proxy)测试:代理预筛路径的不变量(n ≤ 7)。

不变量:
  1. 零回归 vs C:候选恒含 c=0 ⇒ 代理结果 ΣT ≤ C 路径 ΣT;
  2. XOR 不变量:获胜池的 XOR = f(x⊕c),平移回 c 后 = f;
  3. 确定性:同一输入两次结果一致;
  4. 端到端:fprm_proxy=True 的综合逐例 compute_tt 验证。
(探索期纪律:fprm_proxy 是**选择关系**,不与默认 F 取 min,故不做与 F 的比较断言;
  质量差距见 P1 探针:top-8 保留率 71.7%~84.6%。)
"""
from random import randint

from utils.TruthTable import TruthTable
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from limbdd.fusion.FprmProxy import fprm_proxy_best


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_proxy_entries(num_vars=5, rounds=100, verbose=False):
    """代理候选的零回归(vs C)与 XOR 不变量。"""
    print(f'Testing fprm_proxy entries with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        kept_f, st_f, c_f = fprm_proxy_best(tt.bitstr, n)

        x = 0
        for cb, r, _ in kept_f:
            assert r == 0, '代理路径池应全为 r=0 条目'
            x ^= cb.fn()
        for cb in st_f:
            x ^= cb.fn()
        assert shift_fn(x, c_f, n) == tt.bitstr, f'XOR 不变量破坏: tt={tt.bitstr} c={c_f}'

        kept_0, st_0 = fuse_pool_global(anf_pool(tt.bitstr, n), n)
        assert pool_t_cost(kept_f, st_f) <= pool_t_cost(kept_0, st_0), '代理劣于 C 路径(c=0)'
    print('fprm_proxy entries test passed!!!')


def test_proxy_determinism(num_vars=5, rounds=30, verbose=False):
    """确定性:两次调用结果一致。"""
    print(f'Testing fprm_proxy determinism with {rounds} random {num_vars}-input functions...')

    n = num_vars
    for _ in range(rounds):
        tt = random_tt(n)
        k1, s1, c1 = fprm_proxy_best(tt.bitstr, n)
        k2, s2, c2 = fprm_proxy_best(tt.bitstr, n)
        assert c1 == c2, '候选平移不一致'
        assert [cb for cb, _, _ in k1] == [cb for cb, _, _ in k2], 'kept 不一致'
        assert s1 == s2, 'standalone 不一致'
    print('fprm_proxy determinism test passed!!!')


def test_proxy_end_to_end(num_vars=5, rounds=60, verbose=False):
    """生产路径(fprm_proxy=True)端到端 verify。"""
    print(f'Testing fprm_proxy end-to-end with {rounds} random {num_vars}-input functions...')

    n = num_vars
    s = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True, fprm_proxy=True)
    for _ in range(rounds):
        tt = random_tt(n)
        s.synthesize(tt)
        assert s.circ.verify(tt), f'综合验证失败: tt={tt.bitstr}'
    print('fprm_proxy end-to-end test passed!!!')


def main(verbose=False):
    for n, rounds in ((4, 100), (5, 80)):
        test_proxy_entries(n, rounds, verbose)
    test_proxy_determinism(5, 30, verbose)
    test_proxy_end_to_end(5, 60, verbose)
    print('====== All fprm_proxy tests passed!!! ======')


if __name__ == '__main__':
    main()
