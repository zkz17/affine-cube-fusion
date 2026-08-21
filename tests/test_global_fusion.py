from collections import Counter
from random import randint

from utils.TruthTable import TruthTable
from utils.Qubit import QubitManager
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.CompleteBranch import (CompleteBranch, abs_lim, extract_complete_branches,
                                   affine_from_fn, plan_from_affine, is_affine_cube)
from limbdd.fusion.BranchFusion import shift_fn, fuse_pool_global
from utils.to_qiskit import to_qiskit
from experiments.phase0_branch_stats import count_t_cost
from tests.test_complete_branch import prepare_pieces


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def collect_entries(tt, reorder):
    """图阶段后收集 (CompleteBranch, root_mask, piece_key) 全局条目。"""
    entries = []
    for key, g in enumerate(prepare_pieces(tt, reorder)):
        r = abs_lim(g.root.lim, g.var_order)
        for cb in extract_complete_branches(g, tt.num_var, anomalies=Counter()):
            entries.append((cb, r, key))
    return entries


def emit_check(cb, num_vars):
    qm = QubitManager()
    for v in range(num_vars):
        qm.get_input_qubit(v)
    assert cb.to_circuit(qm).compute_tt() == cb.fn(), f'发射不一致: {cb!r}'


def test_shift_fusion_contract(num_vars=5, rounds=300, max_pairs=30, verbose=False):
    """平移融合契约：d' = f_A(x⊕r_A) ⊕ f_B(x⊕r_B) 的判定与构造精确。"""
    print(f'Testing shifted fusion contract with {rounds} random {num_vars}-input functions...')

    checked = 0
    for i in range(rounds):
        tt = random_tt(num_vars)
        entries = collect_entries(tt, reorder=(i % 2 == 1))
        if len(entries) < 2:
            continue
        pairs = [(a, b) for a in range(len(entries)) for b in range(a + 1, len(entries))]
        if len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
        for a, b in pairs:
            ca, ra, _ = entries[a]
            cb, rb, _ = entries[b]
            fa = shift_fn(ca.fn(), ra, num_vars)
            fb = shift_fn(cb.fn(), rb, num_vars)
            d = fa ^ fb
            aff = (d == 0) or is_affine_cube(d, num_vars)[0]
            if not aff:
                continue
            if d == 0:
                continue  # 抵消语义由池测试覆盖
            base, basis = affine_from_fn(d, num_vars)
            plan = plan_from_affine(base, basis, num_vars)
            fused = CompleteBranch(num_vars, synthetic=plan)
            assert fused.fn() == d, f'平移融合契约失败: ra={ra}, rb={rb}'
            emit_check(fused, num_vars)
            checked += 1
            if verbose:
                print('  ', i, a, b, f'ra={ra:b} rb={rb:b} T {ca.t_cost()}+{cb.t_cost()} -> {fused.t_cost()}')

    print(f'Shifted fusion contract test passed!!! ({checked} pairs)')


def test_global_pool_invariant(num_vars=5, rounds=300, verbose=False):
    """全局池：有效函数 XOR 不变量、段数单调减、产物可发射。"""
    print(f'Testing global pool invariants with {rounds} random {num_vars}-input functions...')

    for i in range(rounds):
        tt = random_tt(num_vars)
        entries = collect_entries(tt, reorder=(i % 2 == 1))
        if not entries:
            continue
        xor_before = 0
        for cb, r, _ in entries:
            xor_before ^= shift_fn(cb.fn(), r, num_vars)

        kept, standalone = fuse_pool_global(entries, num_vars)
        xor_after = 0
        for cb, r, _ in kept:
            xor_after ^= shift_fn(cb.fn(), r, num_vars)
        for cb in standalone:
            xor_after ^= cb.fn()

        assert xor_after == xor_before, f'全局池 XOR 不变量破坏: tt={tt}'
        assert len(kept) + len(standalone) <= len(entries), \
            f'段数未单调减: {len(entries)} -> {len(kept) + len(standalone)}'
        for cb in standalone:
            emit_check(cb, num_vars)
        if verbose:
            print('  ', i, str(tt), f'{len(entries)} -> {len(kept)}+{len(standalone)}')

    print('Global pool invariant test passed!!!')


def test_global_backend_equivalence(num_vars=5, rounds=300, verbose=False):
    """新旧后端差分（fused_backend=True = 全局融合路径）。"""
    print(f'Testing global fused backend with {rounds} random {num_vars}-input functions...')

    t_old_sum, t_new_sum = 0, 0
    for i in range(rounds):
        tt = random_tt(num_vars)
        old = LimBDDOracleSynthesizer(reorder=(i % 2 == 1), ancilla_num=0, fused_backend=False)
        new = LimBDDOracleSynthesizer(reorder=(i % 2 == 1), ancilla_num=0, fused_backend=True)
        old.synthesize(tt)
        new.synthesize(tt)
        assert old.circ.verify(tt), f'旧后端验证失败: {tt}'
        assert new.circ.verify(tt), f'全局后端验证失败: {tt}'
        t_old = count_t_cost(to_qiskit(old.circ))
        t_new = count_t_cost(to_qiskit(new.circ))
        assert t_new <= t_old, f'全局后端 T 上升: {t_old} -> {t_new} (tt={tt})'
        t_old_sum += t_old
        t_new_sum += t_new
        if verbose:
            print('  ', i, str(tt), f'T {t_old} -> {t_new}')

    print(f'Global backend equivalence test passed!!! (ΣT {t_old_sum} -> {t_new_sum}, '
          f'-{1 - t_new_sum / max(1, t_old_sum):.1%})')


def test_global_backend_stats(n4=200, n6=100, n7=30, verbose=False):
    """n=4/6/7 差分统计样本。"""
    for num_vars, rounds in ((4, n4), (6, n6), (7, n7)):
        t_old_sum, t_new_sum = 0, 0
        rng_tts = [random_tt(num_vars) for _ in range(rounds)]
        for i, tt in enumerate(rng_tts):
            old = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=False)
            new = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
            old.synthesize(tt)
            new.synthesize(tt)
            assert old.circ.verify(tt) and new.circ.verify(tt), f'验证失败: {tt}'
            t_old = count_t_cost(to_qiskit(old.circ))
            t_new = count_t_cost(to_qiskit(new.circ))
            assert t_new <= t_old, f'全局后端 T 上升: {t_old} -> {t_new} (tt={tt})'
            t_old_sum += t_old
            t_new_sum += t_new
            if verbose:
                print('  ', i, str(tt), f'T {t_old} -> {t_new}')
        print(f'Global backend stats (n={num_vars}, {rounds} cases): ΣT {t_old_sum} -> {t_new_sum}, '
              f'-{1 - t_new_sum / max(1, t_old_sum):.1%}')


def test_anf_pool(num_vars=6, rounds=200, verbose=False):
    """ANF 分解池：XOR 不变量、融合后池有效且不劣于单项式池。"""
    from limbdd.fusion.BranchFusion import anf_pool, pool_t_cost

    print(f'Testing ANF pool with {rounds} random {num_vars}-input functions...')

    for i in range(rounds):
        tt = random_tt(num_vars)
        entries = anf_pool(tt.bitstr, num_vars)
        assert entries, 'ANF 池不应为空'
        xor_before = 0
        for cb, r, _ in entries:
            xor_before ^= cb.fn()
        assert xor_before == tt.bitstr, f'ANF 池 XOR 不变量破坏: {tt}'

        kept, st = fuse_pool_global(entries, num_vars)
        assert pool_t_cost(kept, st) <= sum(cb.t_cost() for cb, _, _ in entries)
        x = 0
        for cb, r, _ in kept:
            x ^= shift_fn(cb.fn(), r, num_vars)
        for cb in st:
            x ^= cb.fn()
        assert x == tt.bitstr, f'ANF 池融合破坏 XOR 不变量: {tt}'
        if verbose:
            print('  ', i, str(tt), f'{len(entries)} -> {len(kept)}+{len(st)}')

    print('ANF pool test passed!!!')


def main():
    test_shift_fusion_contract()
    test_global_pool_invariant()
    test_global_backend_equivalence()
    test_global_backend_stats()
    test_anf_pool()


if __name__ == '__main__':
    main()
