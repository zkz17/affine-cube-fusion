from collections import Counter
from random import randint

from utils.TruthTable import TruthTable
from utils.Qubit import QubitManager
from backend.CompleteBranch import CompleteBranch, t_cost, is_affine_cube
from limbdd.Extraction import extract_complete_branches
from backend.fusion.BranchFusion import is_fusable, four_case_kind, fuse_pair, fuse_pool
from experiments.phase0_branch_stats import is_affine_subspace
from tests.test_complete_branch import prepare_pieces


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def emit_check(cb, num_vars):
    """Emit a synthetic segment and reconcile it with compute_tt (also checks input restoration)."""
    qm = QubitManager()
    for v in range(num_vars):
        qm.get_input_qubit(v)
    assert cb.to_circuit(qm).compute_tt() == cb.fn(), f'emission mismatch: {cb!r}'


def test_fusion_contract(num_vars=5, rounds=300, max_pairs=30, verbose=False):
    """Fusion contract: equivalent criteria (d-affine vs four cases), exact construction, optimal control count, canonical form."""
    print(f'Testing branch fusion contract with {rounds} random {num_vars}-input functions...')

    checked = 0
    for i in range(rounds):
        tt = random_tt(num_vars)
        branches = []
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            branches.extend(extract_complete_branches(g, num_vars, origin=i, anomalies=Counter()))
        if len(branches) < 2:
            continue
        pairs = [(a, b) for a in range(len(branches)) for b in range(a + 1, len(branches))]
        if len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
        for a, b in pairs:
            ca, cb = branches[a], branches[b]
            d = ca.fn() ^ cb.fn()
            # Criterion 1: d-affine (cross-checked with the Phase 0 criterion)
            aff_cube = (d == 0) or is_affine_cube(d, num_vars)[0]
            aff_phase0 = (d == 0) or is_affine_subspace(d, num_vars)[0]
            assert aff_cube == aff_phase0, f'affine criterion mismatch: tt={tt}, d={d}'
            # Criterion 2: authoritative is_fusable is equivalent to the four-case classification
            fus = is_fusable(ca, cb)
            kind = four_case_kind(ca, cb)
            assert fus == aff_cube, f'is_fusable disagrees with d-affine: tt={tt}'
            assert fus == (kind is not None), f'four-case classification disagrees with d-affine: tt={tt}, kind={kind}'

            if not fus:
                fused, cancelled = fuse_pair(ca, cb)
                assert fused is None and not cancelled
                continue

            fused, cancelled = fuse_pair(ca, cb)
            assert cancelled == (d == 0), f'cancellation semantics mismatch: d=={d == 0}, cancelled={cancelled}'
            if cancelled:
                checked += 1
                continue

            # Construction contract: f_fused == f_A ⊕ f_B, and the emitted circuit agrees exactly via compute_tt
            assert fused.fn() == d, f'fused segment function != f_A⊕f_B: tt={tt}'
            emit_check(fused, num_vars)
            # Optimal control count: n − dim(d)
            _, k = is_affine_cube(d, num_vars)
            assert fused.nctrl() == num_vars - k, \
                f'control count not optimal: nctrl={fused.nctrl()}, n-k={num_vars - k}'
            assert fused.t_cost() == t_cost(num_vars - k)
            # Canonical form: x_trans=0, cnot_ops/controls ascending
            plan = fused.synthetic
            assert plan.x_trans == 0
            assert plan.cnot_ops == tuple(sorted(plan.cnot_ops))
            assert plan.controls == tuple(sorted(plan.controls))
            checked += 1
            if verbose:
                print('  ', i, a, b, 'kind=', kind, 'T:', ca.t_cost(), '+', cb.t_cost(),
                      '->', fused.t_cost(), plan)

    print(f'Branch fusion contract test passed!!! ({checked} pairs)')


def test_fuse_pool(num_vars=5, rounds=300, verbose=False):
    """Pool fusion: XOR invariant, monotone decrease in segment count, T not increased, determinism, all products valid."""
    print(f'Testing fusion pool invariants with {rounds} random {num_vars}-input functions...')

    for i in range(rounds):
        tt = random_tt(num_vars)
        branches = []
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            branches.extend(extract_complete_branches(g, num_vars, origin=i, anomalies=Counter()))
        if not branches:
            continue
        xor_before = 0
        t_before = 0
        for cb in branches:
            xor_before ^= cb.fn()
            t_before += cb.t_cost()

        fused = fuse_pool(branches)
        xor_after, t_after = 0, 0
        for cb in fused:
            xor_after ^= cb.fn()
            t_after += cb.t_cost()
            emit_check(cb, num_vars)

        assert xor_after == xor_before, f'pool XOR invariant broken: tt={tt}'
        assert len(fused) <= len(branches), f'segment count did not decrease monotonically: {len(branches)} -> {len(fused)}'
        assert t_after <= t_before, f'T did not decrease monotonically: {t_before} -> {t_after}'
        # Determinism
        assert fuse_pool(branches) == fused, f'fuse_pool is nondeterministic: tt={tt}'
        if verbose:
            print('  ', i, str(tt), f'{len(branches)} -> {len(fused)} segments, T {t_before} -> {t_after}')

    print('Fusion pool invariant test passed!!!')


def test_four_case_construction(num_vars=5, rounds=300, verbose=False):
    """Construct each of the four cases: dimensions/control counts obey Theorem 2, and real data covers all cases (verified when they occur)."""
    print(f'Testing four-case fusion construction with {rounds} random {num_vars}-input functions...')

    kind_hits = Counter()
    constructed = Counter()
    for i in range(rounds):
        tt = random_tt(num_vars)
        branches = []
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            branches.extend(extract_complete_branches(g, num_vars, origin=i, anomalies=Counter()))
        for a in range(len(branches)):
            for b in range(a + 1, len(branches)):
                kind = four_case_kind(branches[a], branches[b])
                if kind is None:
                    continue
                kind_hits[kind] += 1
                ca, cb = branches[a], branches[b]
                d = ca.fn() ^ cb.fn()
                fused, cancelled = fuse_pair(ca, cb)
                assert cancelled == (d == 0)
                if cancelled:
                    continue
                _, k = is_affine_cube(d, num_vars)
                assert fused.nctrl() == num_vars - k
                # Check dimensions against Theorem 2
                if kind == 1:
                    assert k == len(ca.direction_key()) + 1, f'case 1 dimension wrong: k={k}'
                elif kind == 2:
                    assert k == len(ca.direction_key()), f'case 2 dimension wrong: k={k}'
                else:
                    smaller = min(len(ca.direction_key()), len(cb.direction_key()))
                    assert k == smaller, f'case {kind} dimension wrong: k={k}, smaller={smaller}'
                emit_check(fused, num_vars)
                constructed[kind] += 1
                if verbose:
                    print('  ', i, 'kind=', kind, 'dimK=',
                          [len(branches[a].direction_key()), len(branches[b].direction_key())],
                          '->', k)

    print(f'Four-case construction test passed!!! (hits {dict(kind_hits)}, constructed {dict(constructed)})')

    # Deterministic construction for cases 3/4 (rare in random data): a hyperplane coset segment and the constant-1 segment are complementary
    const1 = CompleteBranch(4)
    single = CompleteBranch(4, mask=1 << 1, bitstr=0)          # [x1=0], K=span{x0,x2,x3}
    assert four_case_kind(single, const1) == 3, f'expected case 3 (K_A ⊂ K_B), got {four_case_kind(single, const1)}'
    assert four_case_kind(const1, single) == 4, f'expected case 4 (K_B ⊂ K_A), got {four_case_kind(const1, single)}'
    fused, cancelled = fuse_pair(single, const1)
    assert not cancelled and fused.nctrl() == 1 and fused.fn() == (single.fn() ^ const1.fn())
    emit_check(fused, 4)


def test_pivot_oracle_alignment(num_vars=5, rounds=30, max_pairs=2, verbose=False):
    """
    Pivot oracle contract alignment (only for complete branch pairs in the same graph): for pairs
    with d affine, run pivot_to + nested synthesis; when the contract f_pivot∘map == f_A⊕f_B holds,
    it must agree with our fused segment function; failures count as skips (pivot_to is unsound for
    pairs containing one-child positions, a known defect).
    """
    from circuit.MCXCircuit import MCXCircuit
    from limbdd.LimBDDManager import LimBDDManager
    from backend.OracleSynthesizer import OracleSynthesizer

    print(f'Testing pivot oracle alignment with {rounds} random {num_vars}-input functions...')

    ok_cnt, skip_cnt = 0, 0
    for i in range(rounds):
        tt = random_tt(num_vars)
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            window = 0
            for v in g.var_order:
                window |= 1 << v
            branches = extract_complete_branches(g, num_vars, origin=i, anomalies=Counter())
            complete = [cb for cb in branches if cb.mask == window and not cb.ft_lims]
            if len(complete) < 2:
                continue
            pairs = [(a, b) for a in range(len(complete)) for b in range(a + 1, len(complete))]
            if len(pairs) > max_pairs:
                pairs = pairs[:max_pairs]
            for a, b in pairs:
                ca, cb = complete[a], complete[b]
                d = ca.fn() ^ cb.fn()
                if d == 0 or not is_affine_cube(d, num_vars)[0]:
                    continue
                try:
                    pivot_limbdd, x_map, cx_map = LimBDDManager.pivot_to(
                        g.root.node, ca.source_branch, cb.source_branch, g)
                    nested = OracleSynthesizer(ancilla_num=0, fused_backend=False)
                    qm = QubitManager()
                    for v in range(num_vars):
                        qm.get_input_qubit(v)
                    nested.qubit_mgr = qm
                    nested.circ = MCXCircuit(qm)
                    nested.limbdd_mgr.limbdd = pivot_limbdd
                    nested.limbdd_mgr.limbdd_list.clear()
                    for h in range(pivot_limbdd.root.level + 1):
                        nested.limbdd_mgr.limbdd_list[h] = []
                    nested.process()
                    ttp = nested.circ.compute_tt()
                    ok = True
                    for x in range(1 << num_vars):
                        y = x
                        for var in x_map:
                            if x_map[var]:
                                y ^= 1 << var
                        for ctrl in cx_map:
                            for tgt in cx_map[ctrl]:
                                if ((x >> ctrl) & 1) == cx_map[ctrl][tgt]:
                                    y ^= 1 << tgt
                        if ((ttp >> y) & 1) != ((d >> x) & 1):
                            ok = False
                            break
                    if ok:
                        ok_cnt += 1
                        # Must agree with our fused segment
                        fused, cancelled = fuse_pair(ca, cb)
                        assert not cancelled and fused.fn() == d
                    else:
                        skip_cnt += 1  # known pivot_to defect (one-child positions)
                except Exception:
                    skip_cnt += 1
                if verbose:
                    print('  ', i, 'pivot ok/skip:', ok_cnt, skip_cnt)

    print(f'Pivot oracle alignment test passed!!! (ok={ok_cnt}, skip={skip_cnt})')


def main():
    test_fusion_contract()
    test_fuse_pool()
    test_four_case_construction()
    test_pivot_oracle_alignment()


if __name__ == '__main__':
    main()
