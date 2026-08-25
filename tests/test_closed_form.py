# -*- coding: utf-8 -*-
"""
Phase 0.1: full consistency cross-validation of the closed-form criterion/construction vs the fn criterion/construction.

Object under test: `backend/fusion/ClosedForm.py` (closed form, O(n^2))
Reference baseline: the fn path (O(2^n)) of `backend/fusion/BranchFusion.py` + `limbdd/CompleteBranch.py`

This is the **acceptance precondition** for Phase 1.1 (T-1 switch) in docs/architecture.md:
zero divergence must first be confirmed on n=4..7 before the production path may be switched to the closed form.

Core equivalence criterion
--------------------------
`plan_from_affine(base, basis, n)` depends only on the affine **set**, not on the choice of representative/basis:
  - `rref` is unique for a given subspace ⇒ the pivot/free column partition is unique;
  - if e' = e ⊕ v (v ∈ K), then the free-column constants c_f are unchanged
    (because v satisfies the homogeneous constraints v_f ⊕ Σ_p (r_p)_f v_p = 0).
Therefore "both paths give the same SyntheticPlan" ⟺ "both paths give the same affine set",
so we can use `==` as the strongest assertion without comparing the (base, basis) representations one by one.
"""
from collections import Counter
from random import Random, randint

from utils.TruthTable import TruthTable
from backend.CompleteBranch import (CompleteBranch, affine_from_fn, in_span, is_affine_cube,
                                   plan_from_affine, reduce_basis, t_cost)
from limbdd.Extraction import abs_lim, extract_complete_branches
from backend.fusion.BranchFusion import anf_pool, four_case_kind, shift_fn
from backend.fusion.ClosedForm import (_build_joint, _subspace_leq, canonical_affine,
                                      closed_gain, fuse_closed)
from tests.test_complete_branch import prepare_pieces


def random_tt(num_vars):
    return TruthTable(randint(0, (1 << (1 << num_vars)) - 1), num_vars)


def bdd_entries(tt, reorder):
    """Global entries after the BDD graph phase (cb, root_mask, key); nonzero root_mask exercises the shifted closed form."""
    entries = []
    for key, g in enumerate(prepare_pieces(tt, reorder)):
        r = abs_lim(g.root.lim, g.var_order)
        for cb in extract_complete_branches(g, tt.num_var, anomalies=Counter()):
            entries.append((cb, r, key))
    return entries


def _pairs(count, max_pairs):
    """All pairs; when over the limit, **sample uniformly** (plain truncation would only cover (0, j), a severe bias)."""
    out = [(i, j) for i in range(count) for j in range(i + 1, count)]
    if len(out) <= max_pairs:
        return out
    step = len(out) / max_pairs
    return [out[int(k * step)] for k in range(max_pairs)]


def _pools(tt, i):
    """Alternate between a BDD pool (with root masks) and an ANF pool (root mask always 0); both kinds must be covered."""
    yield bdd_entries(tt, reorder=(i % 2 == 1))
    yield anf_pool(tt.bitstr, tt.num_var)


# kind3 and kind4 are strict mirrors (C3 and its symmetric case); kind1/kind2 are self-mirrors.
#
# **Why both orders must each be checked once**: C3/C4 require the two segments' direction subspaces to be nested
# with codimension 1, and the ordering of pool entries is **systematically correlated** with their dimensions,
# so a single orientation would make one side never appear — increasing the sample size does not help.
#   ANF pool: the direction subspace of x^S is span{e_j : j∉S}, so nesting requires T ⊊ S with |S|=|T|+1;
#            on bitmasks T ⊂ S implies T < S, and entries are ordered by ascending mask, so for fixed a<b we always
#            have mask_a ⊂ mask_b, i.e. dim K_A = dim K_B + 1 ⇒ **always kind4, kind3 never appears**.
#   BDD pool: the same bias is observed (ascending order always kind4, reversed order always kind3, kind1/kind2/None counts unchanged).
MIRROR_KIND = {None: None, 1: 1, 2: 2, 3: 4, 4: 3}


# ------------------------------------------------------------------ main test ----

def test_closed_form_agreement(num_vars=5, rounds=1200, max_pairs=600, verbose=False):
    """Closed form and fn path agree pair by pair on four items: fusable / result dimension / fused segment / gain."""
    print(f'Testing closed-form vs fn agreement with {rounds} random {num_vars}-input functions...')

    n = num_vars
    stat = Counter()
    for i in range(rounds):
        tt = random_tt(n)
        for entries in _pools(tt, i):
            if len(entries) < 2:
                continue
            for a, b in _pairs(len(entries), max_pairs):
                ca, ra, _ = entries[a]
                cb, rb, _ = entries[b]

                # ---- baseline: fn path ----
                d = shift_fn(ca.fn(), ra, n) ^ shift_fn(cb.fn(), rb, n)
                ok_fn, dim_fn = is_affine_cube(d, n)

                # ---- under test: closed form (both orders, doubling as a symmetry check) ----
                aff_a = canonical_affine(ca, ra)
                aff_b = canonical_affine(cb, rb)
                res = fuse_closed(aff_a, aff_b, n)
                res_rev = fuse_closed(aff_b, aff_a, n)

                stat['pairs'] += 1
                assert res.ok == ok_fn, (
                    f'fusable divergence: closed={res.ok} fn={ok_fn}\n  A={ca!r} r={ra}\n  B={cb!r} r={rb}')

                # ---- symmetry: d = f_A⊕f_B is order-independent ⇒ everything agrees except kind 3↔4 swap ----
                assert res_rev.ok == res.ok, f'symmetry broken (ok): {res.ok} vs {res_rev.ok}'
                assert res_rev.cancel == res.cancel, f'symmetry broken (cancel)'
                assert res_rev.kind == MIRROR_KIND[res.kind], (
                    f'symmetry broken (kind): forward={res.kind} reverse={res_rev.kind}\n'
                    f'  A={ca!r} r={ra}\n  B={cb!r} r={rb}')

                if not ok_fn:
                    stat['not_fusable'] += 1
                    continue

                if d == 0:
                    stat['cancel'] += 1
                    assert res.cancel, f'cancellation not recognized by the closed form: A={ca!r} B={cb!r}'
                    assert res.dim == dim_fn == -1, f'cancellation dimension convention mismatch: {res.dim} vs {dim_fn}'
                    continue

                assert not res.cancel, f'closed form wrongly judged as cancellation (d != 0): A={ca!r} B={cb!r}'
                assert res.dim == dim_fn, (
                    f'result dimension divergence: closed={res.dim} fn={dim_fn}\n  A={ca!r} r={ra}\n  B={cb!r} r={rb}')
                assert res_rev.dim == res.dim, f'symmetry broken (dim): {res.dim} vs {res_rev.dim}'

                # ---- the fused segment must be field-identical (same affine set), in both orders ----
                plan_cf = plan_from_affine(res.base, res.basis, n)
                plan_rev = plan_from_affine(res_rev.base, res_rev.basis, n)
                plan_fn = plan_from_affine(*affine_from_fn(d, n), n)
                assert plan_cf == plan_fn, (
                    f'fused segment divergence:\n  closed={plan_cf!r}\n  fn    ={plan_fn!r}\n'
                    f'  A={ca!r} r={ra}\n  B={cb!r} r={rb}')
                assert plan_rev == plan_fn, (
                    f'reverse-order fused segment divergence:\n  reverse={plan_rev!r}\n  fn      ={plan_fn!r}\n'
                    f'  A={ca!r} r={ra}\n  B={cb!r} r={rb}')

                # ---- fusion contract: the segment function equals d exactly ----
                assert CompleteBranch(n, synthetic=plan_cf).fn() == d, (
                    f'closed-form fusion contract failed: A={ca!r} ⊕ B={cb!r}')

                # ---- gain scope agreement ----
                gain_fn = ca.t_cost() + cb.t_cost() - t_cost(n - dim_fn)
                assert closed_gain(ca, cb, res, t_cost, n) == gain_fn, 'gain scopes diverge'

                # Count each of the two orders once: kind3/kind4 are mirrors so their counts must be equal; kind1/kind2 are self-mirrors so they count twice
                stat[f'kind{res.kind}'] += 1
                stat[f'kind{res_rev.kind}'] += 1
                if gain_fn > 0:
                    stat['positive_gain'] += 1

    print(f'  checked pairs: {stat["pairs"]}, fusable: {stat["pairs"] - stat["not_fusable"]}, '
          f'cancel: {stat["cancel"]}, positive gain: {stat["positive_gain"]}')
    print(f'  kind distribution (each of the two orders counted once): ' + ', '.join(
        f'{k}={stat[k]}' for k in ('kind1', 'kind2', 'kind3', 'kind4')))
    # Coverage warning: if a case never appears ⇒ that branch is unverified, insufficient to support the Phase 1.1 switch
    unseen = [k for k in ('kind1', 'kind2', 'kind3', 'kind4') if not stat[k]]
    if unseen:
        print(f'  [WARN] the following cases were not covered in this round, so the corresponding closed-form branches are unverified: {", ".join(unseen)}')
    if verbose:
        print(f'  raw stats: {dict(stat)}')
    assert stat['pairs'] > 0, 'no branch pair was checked; the test is ineffective'
    print('Closed-form agreement test passed!!!')
    return stat


def test_closed_form_translation(num_vars=5, rounds=120, samples=8, verbose=False):
    """Shifted closed form: canonical_affine(cb, r) and affine_from_fn(shift_fn(fn, r)) describe the same affine set."""
    print(f'Testing closed-form translation with {rounds} random {num_vars}-input functions...')

    n = num_vars
    rng = Random(0)
    checked = 0
    for i in range(rounds):
        tt = random_tt(n)
        entries = bdd_entries(tt, reorder=(i % 2 == 1)) + anf_pool(tt.bitstr, n)
        for cb, _, _ in entries[:samples]:
            for r in (0, rng.randrange(1 << n), (1 << n) - 1):
                base_cf, basis_cf = canonical_affine(cb, r)
                aff = affine_from_fn(shift_fn(cb.fn(), r, n), n)
                # Subspace equality uses "same dimension + mutual inclusion", not the uniqueness of row echelon form
                assert (len(basis_cf) == len(aff.basis)
                        and _subspace_leq(basis_cf, aff.basis)
                        and _subspace_leq(aff.basis, basis_cf)), (
                    f'shifted direction subspace divergence: closed={basis_cf} fn={aff.basis} (r={r}, {cb!r})')
                assert in_span(base_cf ^ aff.base, basis_cf), (
                    f'shifted base not in the same coset: closed={base_cf} fn={aff.base} (r={r}, {cb!r})')
                # Strongest criterion: same affine set ⟺ same canonical SyntheticPlan
                assert (plan_from_affine(base_cf, basis_cf, n)
                        == plan_from_affine(aff.base, aff.basis, n)), (
                    f'shifted affine set divergence (r={r}, {cb!r})')
                checked += 1

    print(f'  checked (branch, shift) pairs: {checked}')
    assert checked > 0, 'no branch was checked; the test is ineffective'
    print('Closed-form translation test passed!!!')


def test_intersection_basis(num_vars=5, rounds=80, max_pairs=40, verbose=False):
    """_build_joint's intersection-space basis: the dimension satisfies dim(A∩B) = dimA + dimB − dim(A+B), and members are valid."""
    print(f'Testing intersection basis with {rounds} random {num_vars}-input functions...')

    n = num_vars
    checked = 0
    for i in range(rounds):
        tt = random_tt(n)
        entries = bdd_entries(tt, reorder=(i % 2 == 1)) + anf_pool(tt.bitstr, n)
        if len(entries) < 2:
            continue
        for a, b in _pairs(len(entries), max_pairs):
            _, ka = canonical_affine(entries[a][0], entries[a][1])
            _, kb = canonical_affine(entries[b][0], entries[b][1])

            _, wb = _build_joint(ka, kb)
            expected = len(ka) + len(kb) - len(reduce_basis(ka + kb))
            assert len(wb) == expected, (
                f'intersection-space dimension wrong: got {len(wb)}, expected {expected} (ka={ka}, kb={kb})')
            for v in wb:
                assert in_span(v, ka) and in_span(v, kb), f'intersection-space vector {v} is not in both subspaces'
            checked += 1

    print(f'  checked subspace pairs: {checked}')
    assert checked > 0, 'no subspace pair was checked; the test is ineffective'
    print('Intersection basis test passed!!!')


def test_kind_matches_four_case(num_vars=5, rounds=120, max_pairs=60, verbose=False):
    """The closed form's four-case numbering matches the existing BranchFusion.four_case_kind (compared in both orders)."""
    print(f'Testing kind agreement with four_case_kind on {rounds} random {num_vars}-input functions...')

    n = num_vars
    checked = 0
    seen = Counter()
    for _ in range(rounds):
        tt = random_tt(n)
        entries = anf_pool(tt.bitstr, n)   # ANF pool root_mask is always 0, so it can be compared directly with four_case_kind
        if len(entries) < 2:
            continue
        for a, b in _pairs(len(entries), max_pairs):
            ca, cb = entries[a][0], entries[b][0]
            for x, y in ((ca, cb), (cb, ca)):   # only the reversed order covers kind3
                res = fuse_closed(canonical_affine(x), canonical_affine(y), n)
                expected = four_case_kind(x, y)
                assert res.kind == expected, (
                    f'case numbering divergence: closed={res.kind} four_case_kind={expected}\n  A={x!r}\n  B={y!r}')
                seen[res.kind] += 1
                checked += 1
            # Also check the production implementation's own symmetry
            assert four_case_kind(cb, ca) == MIRROR_KIND[four_case_kind(ca, cb)], (
                f'four_case_kind symmetry broken:\n  A={ca!r}\n  B={cb!r}')

    print(f'  checked (pair, order) combos: {checked}, kinds seen: '
          + ', '.join(f'{k}={seen[k]}' for k in (1, 2, 3, 4)))
    assert checked > 0, 'no branch pair was checked; the test is ineffective'
    print('Kind agreement test passed!!!')


def test_cancellation_closed_form(num_vars=5, rounds=60, samples=6, verbose=False):
    """
    The C1 (cancellation) branch and the C1/C2 distinction inside kind1.

    The initial pool naturally contains no duplicate segments (in practice cancel=0 for all n; duplicates
    only appear after fusion products re-enter the pool), so the `fuse_closed` cancel branch can only be
    covered by **explicitly constructing segments with the same function**.
    """
    print(f'Testing cancellation branch with {rounds} random {num_vars}-input functions...')

    n = num_vars
    checked = Counter()
    for i in range(rounds):
        tt = random_tt(n)
        entries = bdd_entries(tt, reorder=(i % 2 == 1)) + anf_pool(tt.bitstr, n)
        for cb, r, _ in entries[:samples]:
            base, basis = aff = canonical_affine(cb, r)

            # (a) self-pairing must cancel
            res = fuse_closed(aff, aff, n)
            assert res.ok and res.cancel and res.dim == -1, f'self-pair not judged as cancellation: {cb!r}'
            checked['self'] += 1

            # (b) same function, different representation (wire-walk segment vs synthetic segment) must cancel
            fn_shifted = shift_fn(cb.fn(), r, n)
            twin = CompleteBranch(n, synthetic=plan_from_affine(*affine_from_fn(fn_shifted, n), n))
            assert twin.fn() == fn_shifted, f'constructed same-function segment is not equivalent: {cb!r}'
            res_twin = fuse_closed(aff, canonical_affine(twin), n)
            assert res_twin.ok and res_twin.cancel, f'same function with different representation not judged as cancellation: {cb!r}'
            ok_fn, dim_fn = is_affine_cube(fn_shifted ^ twin.fn(), n)
            assert ok_fn and dim_fn == -1, 'fn path convention for d=0 is inconsistent'
            checked['twin'] += 1

            # (c) negative control: same K, different coset ⇒ kind1 but **not** cancellation, dimension +1 (C2)
            v = next((1 << j for j in range(n) if not in_span(1 << j, basis)), None)
            if v is None:      # dim K = n (constant-1 segment), no different coset exists
                continue
            res_c2 = fuse_closed(aff, (base ^ v, basis), n)
            assert res_c2.ok and res_c2.kind == 1 and not res_c2.cancel, (
                f'same K with different coset should be a C2 fusion, not cancellation: {cb!r}')
            assert res_c2.dim == len(basis) + 1, (
                f'C2 dimension should be dim K + 1: got {res_c2.dim}, expected {len(basis) + 1}')
            checked['c2'] += 1

    print(f'  self-pairs: {checked["self"]}, twin-pairs: {checked["twin"]}, '
          f'C2 controls: {checked["c2"]}')
    assert checked['self'] and checked['twin'] and checked['c2'], 'cancellation/C2 branches were not covered'
    print('Cancellation branch test passed!!!')


# ------------------------------------------------------------------- entry ----

# n -> (rounds, max_pairs); Phase 0.1 requires covering n=4..7
PROFILE = {
    4: (150, 60),
    5: (120, 60),
    6: (60, 60),
    7: (30, 40),
}


def main(verbose=False):
    for n in sorted(PROFILE):
        rounds, max_pairs = PROFILE[n]
        print(f'====== n = {n} ======')
        test_closed_form_agreement(n, rounds, max_pairs, verbose)
        test_closed_form_translation(n, max(rounds // 3, 10), 8, verbose)
        test_intersection_basis(n, max(rounds // 2, 10), max_pairs, verbose)
        test_kind_matches_four_case(n, max(rounds // 2, 10), max_pairs, verbose)
        test_cancellation_closed_form(n, max(rounds // 3, 10), 6, verbose)
    print('====== All closed-form consistency tests passed!!! ======')


if __name__ == '__main__':
    main()
