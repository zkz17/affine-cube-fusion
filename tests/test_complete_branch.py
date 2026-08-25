from collections import Counter
from random import randint

from utils.TruthTable import TruthTable
from utils.Qubit import QubitManager
from backend.OracleSynthesizer import OracleSynthesizer
from backend.CompleteBranch import CompleteBranch, SyntheticPlan, t_cost
from limbdd.Extraction import extract_complete_branches
from experiments.phase0_branch_stats import extract_branches, sigs_xor_tt, is_affine_subspace


def prepare_pieces(tt, reorder):
    """Replicate the pipeline graph phase (same as the Phase 0 probe, but stopping before limbdd2circ)."""
    syn = OracleSynthesizer(reorder=reorder, ancilla_num=0)
    syn.construct(tt)
    mgr = syn.limbdd_mgr
    mgr.try_neg(mgr.limbdd)
    mgr.split_neg(mgr.limbdd)
    if reorder:
        syn.variable_reorder()
    pieces = [mgr.limbdd] + [g for lst in mgr.limbdd_list.values() for g in lst]
    return [g for g in pieces if g.root.weight]


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def test_extraction_equivalence(num_vars=6, rounds=300, verbose=False):
    """Compact extraction vs Phase 0 position-walk extraction: field-level equality + function agreement + no anomalies."""
    print(f'Testing CompleteBranch compact extraction with {rounds} random {num_vars}-input functions...')

    for i in range(rounds):
        tt = random_tt(num_vars)
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            anom = Counter()
            compact = extract_complete_branches(g, num_vars, origin=i, anomalies=anom)
            oracle = extract_branches(g, anom)
            if verbose:
                print('  ', i, str(tt), 'piece h=', g.root.level, 'br=', len(compact))

            assert len(compact) == len(oracle), f'branch count mismatch: {len(compact)} vs {len(oracle)}'
            assert anom['neg_edge'] == 0 and anom['zero_node'] == 0 and anom['zero_level'] == 0 \
                and anom['ft_target_above'] == 0 and anom['lim_overflow'] == 0, \
                f'extraction anomaly: {dict(anom)}'

            for cb, (sig, bits, bitnum) in zip(compact, oracle):
                # Rebuild the expected compact fields from the position walk (the oracle's lim is already in absolute variable bit space)
                exp_mask, exp_bitstr, exp_xg, exp_ft = 0, 0, 0, []
                for (var, ft, bit, w0, w1, l0, l1) in sig:
                    if ft:
                        common = l0 & l1
                        exp_xg ^= common
                        m0, m1 = l0 ^ common, l1 ^ common
                        if m0 or m1:
                            exp_ft.append((var, m0, m1))
                    else:
                        exp_mask |= 1 << var
                        if bit:
                            exp_bitstr |= 1 << var
                        exp_xg ^= (l1 if bit else l0)

                assert cb.mask == exp_mask, f'mask mismatch: {cb.mask:b} vs {exp_mask:b}'
                assert cb.bitstr == exp_bitstr, f'bitstr mismatch'
                assert cb.x_global == exp_xg, f'x_global mismatch: {cb.x_global:b} vs {exp_xg:b}'
                assert cb.ft_lims == tuple(exp_ft), f'ft_lims mismatch: {cb.ft_lims} vs {exp_ft}'
                assert cb.source_branch.bitstr == bits and cb.source_branch.bitnum == bitnum, \
                    'source_branch differs from SimpleBranch'
                # Function agreement (compact semantics == position-walk semantics)
                assert cb.fn() == sigs_xor_tt([sig], 0, num_vars), 'compact segment function differs from the position walk'
                # Canonicalization invariants
                for ctrl, m0, m1 in cb.ft_lims:
                    assert m0 & m1 == 0
                    assert not ((cb.mask >> ctrl) & 1)

    print('Compact extraction test passed!!!')


def test_affine_and_emission(num_vars=6, rounds=200, verbose=False):
    """Symbolic affine derivation is self-consistent + canonical emitted circuit agrees with compute_tt + support size invariant."""
    print(f'Testing CompleteBranch affine derivation and emission with {rounds} random {num_vars}-input functions...')

    for i in range(rounds):
        tt = random_tt(num_vars)
        for g in prepare_pieces(tt, reorder=(i % 2 == 1)):
            for cb in extract_complete_branches(g, num_vars, origin=i, anomalies=Counter()):
                base, basis = cb.affine()
                k = cb.nctrl()
                # Support set = base + span(basis), with size 2^(n-k)
                assert cb.fn().bit_count() == (1 << (num_vars - k)), \
                    f'support size mismatch: n={num_vars}, k={k}, popcount={cb.fn().bit_count()}'
                # Emitted circuit agrees with fn (compute_tt also checks input restoration / ancilla cleanup)
                qm = QubitManager()
                for v in range(num_vars):
                    qm.get_input_qubit(v)
                circ_tt = cb.to_circuit(qm).compute_tt()
                assert circ_tt == cb.fn(), f'emitted circuit differs from fn (tt={tt}, cb={cb})'
                if verbose:
                    print('  ', i, str(tt), 'k=', k, 'dimK=', len(basis), 'base=', bin(base))

    print('Affine derivation & emission test passed!!!')


def _fusable(ca, cb):
    """Complete fusable criterion after the §2.4 revision (four cases; same implementation as BranchFusion.four_case_kind)."""
    from backend.fusion.BranchFusion import four_case_kind
    return four_case_kind(ca, cb) is not None


def test_direction_key_consistency(num_vars=5, rounds=300, max_pairs=40, verbose=False):
    """
    Verify the necessary and sufficient fusion condition after the §2.4 revision: d = f_A⊕f_B is
    an affine cube (d==0 or its support is an affine subspace) iff one of the four cases holds
    (K equal / same dimension intersecting with codimension 1 / nested with codimension 1 and the
    same large coset, plus its symmetric case). Dimension relations: K equal and d≠0 → dim(d) = dim(K)+1;
    other fusable cases → dim(d) = dim(smaller one).
    """
    print(f'Testing fusion criterion consistency with {rounds} random {num_vars}-input functions...')

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
            aff, dim = is_affine_subspace(d, num_vars)
            aff = (d == 0) or aff
            fus = _fusable(ca, cb)
            assert aff == fus, \
                f'fusion criterion mismatch: tt={tt}, d_affine={aff}, fusable={fus}, ' \
                f'K_A={ca.direction_key()}, K_B={cb.direction_key()}'
            if aff:
                same_k = (ca.direction_key() == cb.direction_key())
                if d == 0:
                    assert same_k, f'd==0 necessarily implies K equal: K_A={ca.direction_key()}, K_B={cb.direction_key()}'
                elif same_k:
                    assert dim == len(ca.direction_key()) + 1, \
                        f'K-equal fusion dimension should be dim(K)+1: got {dim}'
                else:
                    smaller = min(len(ca.direction_key()), len(cb.direction_key()))
                    assert dim == smaller, \
                        f'other fusable cases should have dim = dim(smaller one): got {dim}, smaller={smaller}'
            checked += 1
            if verbose:
                print('  ', i, a, b, 'affine=', aff, 'fusable=', fus)

    print(f'Fusion criterion consistency test passed!!! ({checked} pairs)')


def test_synthetic_plan(num_vars=5, rounds=300, verbose=False):
    """Synthetic segments: emission and affine derivation are self-consistent for random X/CNOT networks + MCX; equivalent networks have identical keys."""
    print(f'Testing SyntheticPlan with {rounds} random plans on {num_vars} inputs...')

    for i in range(rounds):
        n = num_vars
        x_trans = randint(0, (1 << n) - 1)
        cnot_ops = []
        for _ in range(randint(0, n)):
            c = randint(0, n - 1)
            t = randint(0, n - 1)
            if c != t:
                cnot_ops.append((c, t))
        ctrl_vars = [v for v in range(n) if randint(0, 1)]
        controls = tuple((v, randint(0, 1)) for v in ctrl_vars)

        plan = SyntheticPlan(x_trans=x_trans, cnot_ops=tuple(cnot_ops), controls=controls)
        cb = CompleteBranch(n, synthetic=plan)
        k = len(controls)

        assert cb.fn().bit_count() == (1 << (n - k)), \
            f'synthetic segment support size mismatch: k={k}, popcount={cb.fn().bit_count()}'
        qm = QubitManager()
        for v in range(n):
            qm.get_input_qubit(v)
        assert cb.to_circuit(qm).compute_tt() == cb.fn(), f'synthetic segment emission mismatch: {plan}'
        if verbose:
            print('  ', i, plan, 'k=', k)

    # Equivalent networks: CNOT(x0→x1)+MCX(x1=0) and CNOT(x1→x0)+MCX(x0=0) both compute [x0⊕x1==0]
    p1 = CompleteBranch(num_vars, synthetic=SyntheticPlan(0, ((0, 1),), ((1, 0),)))
    p2 = CompleteBranch(num_vars, synthetic=SyntheticPlan(0, ((1, 0),), ((0, 0),)))
    assert p1 != p2, 'two structurally different plans should not be equal'
    assert p1.cancel_key() == p2.cancel_key(), 'equal functions ⇒ equal cancel_key'
    assert p1.direction_key() == p2.direction_key(), 'equal direction subspaces ⇒ equal direction_key'

    print('SyntheticPlan test passed!!!')


def main():
    test_extraction_equivalence()
    test_affine_and_emission()
    test_direction_key_consistency()
    test_synthetic_plan()


if __name__ == '__main__':
    main()
