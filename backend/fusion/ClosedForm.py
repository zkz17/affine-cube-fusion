# -*- coding: utf-8 -*-
"""
Closed-form fusion decision and construction (the T-1 implementation).

This is the production decision path: BranchFusion's greedy engines call `fuse_closed` /
`canonical_affine` directly. Its equivalence with the old fn path has been fully
cross-validated by `tests/test_closed_form.py` (n=4..7), and after the zero-divergence
acceptance (stage 0.1) `fuse_pool` / `fuse_pool_global` were switched to this module
(stage 1.1).

Math basis
----------
- `CompleteBranch_algebra.md` §2.2 Theorem 2: two segments are fusible ⟺ supp(f_A⊕f_B) is an
  affine subspace, and it splits **necessarily and sufficiently** into the four cases C1-C4.
  Hence the structural test is the authoritative test.
- `CompleteBranch_algebra.md` §3.7: the algebraic closed form (e, B) of the fusion result in each case.
- `CompleteBranch_algebra.md` §1.4: a cross-slice shift only shifts base, the direction subspace K is unchanged.

Complexity comparison (n = variable count; the production path's 2^n terms all disappear)
---------------------------------------------------------------------------------------
| step             | fn path                          | this module          |
|------------------|----------------------------------|----------------------|
| fusibility test  | is_affine_cube  O(2^n)+O(|S|^2)  | O(n^2) bit ops       |
| result dimension | same as above                    | O(1)                 |
| shift            | shift_fn        O(2^n)           | base ^= r  O(1)      |
| result build     | affine_from_fn  O(2^n)           | O(n^2)               |

Canonical-form convention (important)
-------------------------------------
Every `basis` must be the output of `reduce_basis` (distinct lowest set bits, pivots ascending),
otherwise `in_span` does not hold. `CompleteBranch.affine().basis` does **not** guarantee that
form -- it is built from the free columns of an RREF and several basis vectors may share the same
lowest bit (`BranchFusion.four_case_kind` explicitly re-reduces for that reason). Therefore every
entry point must go through `canonical_affine()`.

Zero-cube convention: case C1 (cancellation) returns dim = -1, the "zero cube" of
empty support, consistent with is_affine_cube(fn == 0) == (True, -1) and the zero block of
docs/CompleteBranch_algebra.md §0; the unified gain formula tA + tB - t(n - dim F) then evaluates
to tA + tB under the convention t(-1) = 0, so no special-casing is needed.
"""
from collections import namedtuple

from backend.CompleteBranch import in_span, reduce_basis

# ok     : whether fusible (including cancellation)
# kind   : 1..4, same numbering as BranchFusion.four_case_kind (1=C1+C2, 2=C4, 3/4=C3 and its mirror)
# cancel : d == 0, the two segments cancel each other
# dim    : dimension of the fusion result F; -1 on cancellation (consistent with is_affine_cube's empty-support convention)
# base/basis : F = base + span(basis); meaningless when not fusible or cancelled
ClosedFusion = namedtuple('ClosedFusion', 'ok kind cancel dim base basis')

NOT_FUSABLE = ClosedFusion(False, None, False, 0, 0, ())


# ---------------------------------------------------------------- canonicalization ----

def canonical_affine(cb, root_mask=0):
    """
    Take the branch's canonical affine representation (base, basis) and shift it by the
    slice-root mask root_mask.

    Shift basis (algebra doc §1.4): g(x) = f(x⊕r) ⇒ supp(g) = supp(f) ⊕ r,
    i.e. base ^= r with K unchanged. This replaces the fn path's O(2^n) `shift_fn`.
    """
    aff = cb.affine()
    return aff.base ^ root_mask, reduce_basis(aff.basis)


# ---------------------------------------------------------------- subspaces ----

def _subspace_leq(small, big):
    """span(small) ⊆ span(big). Both must be canonical bases."""
    return all(in_span(v, big) for v in small)


def _subspace_eq(ba, bb):
    """
    span(ba) == span(bb).

    Tuple equality must **not** be used instead: `reduce_basis` produces a row-echelon form
    (distinct lowest bits per vector) rather than an RREF, and row-echelon forms are generally
    not unique (e.g. both (0b01,0b10) and (0b11,0b10) are legal for span{e0,e1}). We decide via
    "same dimension + one-way inclusion", relying on no canonicity assumption.
    """
    return len(ba) == len(bb) and _subspace_leq(ba, bb)


def _pick_outside(basis, sub_basis):
    """Pick a vector of basis outside span(sub_basis) (the caller guarantees existence)."""
    for v in basis:
        if not in_span(v, sub_basis):
            return v
    raise AssertionError('span(basis) ⊆ span(sub_basis): caller precondition violated')


def _insert(piv, vec, acc, w_out):
    """
    Gaussian-elimination insert with source tags.

    Invariant: every pivot entry (vec, acc) satisfies acc ∈ span(A) and vec⊕acc ∈ span(B).
    When vec is eliminated to 0, acc belongs to both, hence acc ∈ A∩B, collected into w_out.
    """
    while vec:
        low = vec & (-vec)
        hit = piv.get(low)
        if hit is None:
            piv[low] = (vec, acc)
            return
        vec ^= hit[0]
        acc ^= hit[1]
    if acc:
        w_out.append(acc)


def _build_joint(basis_a, basis_b):
    """
    Build the pivot table of span(A)+span(B) and the canonical basis of A∩B.

    A's vectors are inserted with acc=v (vec⊕acc = 0 ∈ span(B)); B's vectors with acc=0.
    Returns (piv, w_basis).
    """
    piv, w = {}, []
    for v in basis_a:
        _insert(piv, v, v, w)
    for v in basis_b:
        _insert(piv, v, 0, w)
    return piv, reduce_basis(w)


def _solve_split(d, piv):
    """
    Find x ∈ span(A) with d⊕x ∈ span(B); returns None when d ∉ span(A)+span(B).

    Used by C4 to take the intersection: t = a_A ⊕ x lies in both S_A and S_B.
    """
    x, r = 0, d
    while r:
        low = r & (-r)
        hit = piv.get(low)
        if hit is None:
            return None
        r ^= hit[0]
        x ^= hit[1]
    return x


# ---------------------------------------------------------------- decision ----

def fuse_closed(aff_a, aff_b, n):
    """
    Closed-form fusion decision and construction. aff_* are canonical_affine() outputs (base, basis).

    Returns a ClosedFusion. The four-case numbering matches BranchFusion.four_case_kind.
    """
    base_a, ka = aff_a
    base_b, kb = aff_b
    d = base_a ^ base_b

    # ---- kind 1: K_A = K_B → C1 cancellation / C2 fusion ----
    if _subspace_eq(ka, kb):
        if in_span(d, ka):
            return ClosedFusion(True, 1, True, -1, 0, ())
        basis = reduce_basis(ka + (d,))
        assert len(basis) == len(ka) + 1, 'C2: dimension must increase by 1 when d ∉ K'
        return ClosedFusion(True, 1, False, len(ka) + 1, base_a, basis)

    # ---- kind 3: K_A ⊊ K_B, codim 1, same large coset → F = S_B ∖ S_A, direction K_A ----
    if len(kb) == len(ka) + 1 and _subspace_leq(ka, kb) and in_span(d, kb):
        w = _pick_outside(kb, ka)
        return ClosedFusion(True, 3, False, len(ka), base_a ^ w, ka)

    # ---- kind 4: K_B ⊊ K_A, codim 1, same large coset → F = S_A ∖ S_B, direction K_B ----
    if len(ka) == len(kb) + 1 and _subspace_leq(kb, ka) and in_span(d, ka):
        w = _pick_outside(ka, kb)
        return ClosedFusion(True, 4, False, len(kb), base_b ^ w, kb)

    # ---- kind 2: same dimension, intersection codim 1, cosets meet → C4 ----
    if len(ka) == len(kb):
        piv, wb = _build_joint(ka, kb)
        if len(wb) == len(ka) - 1:
            x = _solve_split(d, piv)
            if x is not None:
                t = base_a ^ x
                u1 = _pick_outside(ka, wb)
                u2 = _pick_outside(kb, wb)
                basis = reduce_basis(wb + (u1 ^ u2,))
                assert len(basis) == len(ka), 'C4: result dimension must equal dim K_A'
                return ClosedFusion(True, 2, False, len(ka), t ^ u1, basis)

    return NOT_FUSABLE


def closed_gain(cb_a, cb_b, res, t_cost_fn, n):
    """
    Closed-form gain (same convention as BranchFusion's gain).

    Cancellation: gain = sum of both segment costs (always accepted; even 0 reduces segment and
    gate counts). Fusion: gain = t(k_A) + t(k_B) − t(n − dim F).
    """
    if not res.ok:
        return None
    base = cb_a.t_cost() + cb_b.t_cost()
    if res.cancel:
        return base
    return base - t_cost_fn(n - res.dim)
