# -*- coding: utf-8 -*-
"""
Phase 3: branch fusion primitives.

Since stage 1.1 (docs/architecture.md T-1), decision and construction both use the **closed-form path**;
since stage 1.3 (T-2), greedy candidate maintenance uses **bucketing + incremental** (see `_greedy_fusion`).

- is_fusable(a, b)   : authoritative fusion test (structural test of Theorem 2, O(n²); d==0 fusion = cancellation);
- four_case_kind(a, b): Theorem 2 four-case structural classification (used by tests and bucket prefiltering);
- fuse_pair(a, b)    : construct the fused segment (SyntheticPlan); cancellation returns (None, True);
- fuse_pool(branches): greedy fusion pool (XOR invariant, segment count monotonically decreases);
- fuse_pool_global   : cross-slice global fusion pool (root mask shift = base ^= r).

Math basis: docs/CompleteBranch_algebra.md (Theorem 2 four cases, §3.6/§3.7 RREF canonical construction);
engineering spec: docs/architecture.md §2.4 / §7.9.
Closed-form implementation: backend/fusion/ClosedForm.py; its equivalence with the old fn path has been
fully cross-validated by tests/test_closed_form.py (n=4..7), and the bitwise output identity is
accepted by the diff against experiments/phase0_results/baseline_tcost_{n}.txt.

Bucketing and incremental maintenance (1.3; why the old trajectory is reproduced bit-for-bit):
  - Theorem 2 guarantees fusable pairs have dim(K) differing by ≤ 1, so only same-bucket and adjacent-bucket pairs are enumerated (bucketing);
  - the (i, j) key of a candidate tuple (gain, i, j, cancel) is globally unique, and during pool rebuild the
    old→new index map of kept items is strictly monotonic ⇒ the surviving remapped candidates **remain sorted in
    descending order**, so they only need to be merged with new candidates, not re-evaluated (incremental);
    new candidates only involve fusion products, O(|kept|·|fused| + |fused|²).

The old fn path (fn() / is_affine_cube / affine_from_fn / shift_fn) is still kept:
  - for research scripts such as counterexample_search.py and residual_measurement.py;
  - when LIMBDD_FUSION_DEBUG=1 is set, the production path cross-checks against the fn path after every fusion.
"""
import os

from backend.CompleteBranch import (CompleteBranch, t_cost, in_span, reduce_basis,
                                   plan_from_affine)
from backend.fusion.ClosedForm import ClosedFusion, NOT_FUSABLE, canonical_affine, fuse_closed

# Debug switch: re-validate the closed-form product against the fn path after each fusion (slow, debug only)
FUSION_DEBUG = os.environ.get('LIMBDD_FUSION_DEBUG') == '1'


def is_fusable(a, b):
    """Authoritative fusion test: supp(f_A⊕f_B) is an affine subspace (including the d==0 cancellation). Closed-form, O(n²)."""
    assert isinstance(a, CompleteBranch) and isinstance(b, CompleteBranch) and a.n == b.n
    return fuse_closed(canonical_affine(a), canonical_affine(b), a.n).ok


def four_case_kind(a, b):
    """Theorem 2 four-case structural classification (docs/CompleteBranch_algebra.md §2.2). Returns 1..4 or None."""
    # First reduce to a canonical basis with pivots in ascending order (the direction_key form does not guarantee that order), then test membership
    ka, kb = reduce_basis(a.direction_key()), reduce_basis(b.direction_key())
    ba, bb = a.affine().base, b.affine().base
    if ka == kb:
        return 1
    if len(kb) == len(ka) + 1 and all(in_span(v, kb) for v in ka) and in_span(ba ^ bb, kb):
        return 3  # K_A ⊂ K_B, codimension 1, same large coset
    if len(ka) == len(kb) + 1 and all(in_span(v, ka) for v in kb) and in_span(ba ^ bb, ka):
        return 4  # K_B ⊂ K_A, codimension 1, same large coset
    if len(ka) == len(kb):
        red = reduce_basis(ka + kb)
        inter_dim = 2 * len(ka) - len(red)
        if inter_dim == len(ka) - 1 and in_span(ba ^ bb, red):
            return 2  # same dimension, intersection of codimension 1, cosets intersect
    return None


def _fuse_from_result(res, n, origin):
    """Build the fused segment from a closed-form decision result (canonical SyntheticPlan, no 2^n enumeration)."""
    plan = plan_from_affine(res.base, res.basis, n)
    return CompleteBranch(n, synthetic=plan, origin=origin)


def fuse_pair(a, b):
    """
    Build the fused segment. Returns (fused, cancelled):
      cancelled=True  -> f_A XOR f_B == 0, both segments removed from the pool;
      fused not None  -> the synthetic segment replacing the pair (canonical SyntheticPlan);
      (None, False)   -> not fusible.
    """
    if not isinstance(a, CompleteBranch) or not isinstance(b, CompleteBranch) or a.n != b.n:
        return None, False
    res = fuse_closed(canonical_affine(a), canonical_affine(b), a.n)
    if not res.ok:
        return None, False
    if res.cancel:
        return None, True
    fused = _fuse_from_result(res, a.n, a.origin)
    if FUSION_DEBUG:
        d = a.fn() ^ b.fn()
        assert fused.fn() == d, f'fusion contract violated: {a!r} XOR {b!r}'
    return fused, False


# ------------------------------------------------------------------ greedy engine ----

class _Entry:
    """Greedy-engine pool entry: branch + precomputed canonical affine representation
    (root-mask shift already baked in for the global pool)."""

    __slots__ = ('cb', 'aff', 'cost', 'extra', 'plain')

    def __init__(self, cb, aff, extra=None):
        self.cb = cb
        self.aff = aff            # (base, basis), output of canonical_affine
        self.cost = cb.t_cost()
        self.extra = extra        # fuse_pool: None;global: (root_mask, key)
        # Plain-cube fast path: when the direction subspace is a coordinate subspace
        # (all basis vectors are unit vectors), record the constraint-set mask
        # C = { constrained variables } = non-free columns. ANF monomials and most
        # BDD walk segments satisfy this.
        basis = aff[1]
        if all(v & (v - 1) == 0 for v in basis):
            cols = 0
            for v in basis:
                cols |= v
            self.plain = ((1 << cb.n) - 1) ^ cols
        else:
            self.plain = None


def _fuse_plain(a, b, ca, cb, n):
    """
    O(popcount) closed-form check for two "plain cubes" (direction subspace is a
    coordinate subspace).

    Support = {x : x_C = (projection of base onto C)}, C being the constraint-set mask.
    Identical to the Theorem 2 classification of fuse_closed; the construction yields
    the same affine set (the canonical SyntheticPlan is uniquely determined by
    plan_from_affine, so the output is bit-for-bit unchanged).
    """
    d = a ^ b
    if ca == cb:
        if d & ca == 0:
            return ClosedFusion(True, 1, True, -1, 0, ())      # C1
        units = tuple(1 << v for v in range(n) if not ((ca >> v) & 1))
        return ClosedFusion(True, 1, False, n - ca.bit_count() + 1, a, units + (d & ca,))  # C2
    if ca.bit_count() == cb.bit_count() + 1 and cb & ~ca == 0:          # K_A ⊂ K_B (codim 1)
        if d & cb == 0:                                                 # same large coset
            w = ca & ~cb
            units = tuple(1 << v for v in range(n) if not ((ca >> v) & 1))
            return ClosedFusion(True, 3, False, n - ca.bit_count(), a ^ w, units)
    if cb.bit_count() == ca.bit_count() + 1 and ca & ~cb == 0:          # K_B ⊂ K_A (codim 1)
        if d & ca == 0:
            w = cb & ~ca
            units = tuple(1 << v for v in range(n) if not ((cb >> v) & 1))
            return ClosedFusion(True, 4, False, n - cb.bit_count(), b ^ w, units)
    if ca.bit_count() == cb.bit_count():
        inter = ca & cb
        if inter.bit_count() == ca.bit_count() - 1 and d & inter == 0:  # C4: intersection codim 1 and cosets meet
            j = (cb & ~ca).bit_length() - 1
            jp = (ca & ~cb).bit_length() - 1
            t = a ^ (d & ~ca)
            wb = tuple(1 << v for v in range(n) if not ((ca | cb) >> v) & 1)
            return ClosedFusion(True, 2, False, n - ca.bit_count(),
                                t ^ (1 << j), wb + ((1 << j) ^ (1 << jp),))
    return NOT_FUSABLE


def _fuse_two(e_i, e_j, n):
    """Closed-form check for two pool entries: plain-cube fast path first, then generic fuse_closed."""
    if e_i.plain is not None and e_j.plain is not None:
        return _fuse_plain(e_i.aff[0], e_j.aff[0], e_i.plain, e_j.plain, n)
    return fuse_closed(e_i.aff, e_j.aff, n)


def _pair_candidate(e_i, e_j, i, j, n, accept_neutral=False):
    """dim prefilter + closed-form check; returns a candidate tuple (gain, i, j, cancel) or None.

    accept_neutral=True (stage-2 hypothesis test G-1a): non-cancelling fusions with gain == 0 also enter the
    candidates. The segment count still strictly decreases every round, so termination is natural, no tabu needed.
    """
    if abs(len(e_i.aff[1]) - len(e_j.aff[1])) > 1:
        return None      # Theorem 2: fusible implies dim(K) differs by ≤ 1
    res = _fuse_two(e_i, e_j, n)
    if not res.ok:
        return None
    if res.cancel:
        # cancellation is always beneficial (even with 0 T gain it reduces segment and gate count), always candidate
        return (e_i.cost + e_j.cost, i, j, True)
    gain = e_i.cost + e_j.cost - t_cost(n - res.dim)
    if gain > 0 or (accept_neutral and gain == 0):
        return (gain, i, j, False)
    return None


def _bucket_pairs(pool):
    """
    Yield all (i, j) pairs with |Δdim| ≤ 1 grouped by dim(K) buckets (i < j), generator avoids materialization.

    The (i, j) key of a candidate tuple (gain, i, j, cancel) is globally unique, so the yield order does not
    affect the sort result; compared to the old full (i<j) enumeration the only difference is skipping pairs
    that Theorem 2 guarantees are not fusible.
    """
    buckets = {}
    for idx, e in enumerate(pool):
        buckets.setdefault(len(e.aff[1]), []).append(idx)
    for d in sorted(buckets):
        b = buckets[d]
        for a in range(len(b)):
            for c in range(a + 1, len(b)):
                yield b[a], b[c]
        b2 = buckets.get(d + 1)
        if b2:
            for i in b:
                for j in b2:
                    if i < j:
                        yield i, j
                    else:
                        yield j, i


def _merge_desc(a, b):
    """Merge two candidate lists sorted by (gain, i, j, cancel) descending (elements are disjoint, so no tie ambiguity)."""
    out = []
    p = q = 0
    while p < len(a) and q < len(b):
        if a[p] >= b[q]:
            out.append(a[p])
            p += 1
        else:
            out.append(b[q])
            q += 1
    out.extend(a[p:])
    out.extend(b[q:])
    return out


def _greedy_fusion(entries, n, max_rounds, fused_extra, debug_check=None, accept_neutral=False):
    """
    Bucketed + incremental greedy fusion engine, **bit-for-bit identical** to the old
    "full re-enumeration + re-sort every round" (with accept_neutral=False; True is the
    zero-gain plateau-move switch of the stage-2 hypothesis test):

    1. first round enumerates all |Δdim|≤1 pairs via buckets, builds and sorts candidates;
    2. each round takes a maximal matching in descending (gain, i, j) order (each branch used at most once);
    3. pool rebuild = kept items (original order) + fusion products (candidate processing order);
    4. surviving candidates only get their indices remapped (strictly monotonic mapping preserves order),
       new candidates only involve fusion products; merging the two descending lists yields this round's
       candidates -- kept pairs never need re-checking.

    fused_extra: the extra field of fusion-product _Entry (fuse_pool uses None; global uses (0, None)).
    Returns the final pool (list of _Entry).
    """
    pool = list(entries)
    if max_rounds is None:
        max_rounds = len(pool) + 1

    cands = []
    for i, j in _bucket_pairs(pool):
        c = _pair_candidate(pool[i], pool[j], i, j, n, accept_neutral)
        if c is not None:
            cands.append(c)
    cands.sort(reverse=True)

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        if not cands:
            break

        used = set()
        fused_list = []
        for gain, i, j, cancel in cands:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            if cancel:
                continue
            res = _fuse_two(pool[i], pool[j], n)
            assert res.ok and not res.cancel
            fused = _fuse_from_result(res, n, pool[i].cb.origin)
            if FUSION_DEBUG and debug_check is not None:
                debug_check(pool[i], pool[j], fused)
            fused_list.append(fused)

        kept = [e for idx, e in enumerate(pool) if idx not in used]
        fused_entries = [_Entry(f, canonical_affine(f), extra=fused_extra)
                         for f in fused_list]
        pos = {}
        for new_idx, old_idx in enumerate(idx for idx in range(len(pool)) if idx not in used):
            pos[old_idx] = new_idx
        pool = kept + fused_entries

        # ---- incremental candidate maintenance ----
        remapped = []
        for g, i, j, c in cands:
            if i in used or j in used:
                continue
            remapped.append((g, pos[i], pos[j], c))
        # pos strictly monotonic, so the remapped list stays descending

        n_kept = len(kept)
        m_new = len(pool)
        new_cands = []
        for i in range(n_kept):
            for j in range(n_kept, m_new):
                c = _pair_candidate(pool[i], pool[j], i, j, n, accept_neutral)
                if c is not None:
                    new_cands.append(c)
        for i in range(n_kept, m_new):
            for j in range(i + 1, m_new):
                c = _pair_candidate(pool[i], pool[j], i, j, n, accept_neutral)
                if c is not None:
                    new_cands.append(c)
        new_cands.sort(reverse=True)
        cands = _merge_desc(remapped, new_cands)

    return pool


def fuse_pool(branches, max_rounds=None, accept_neutral=False):
    """
    Greedy fusion pool: XOR invariant, segment count monotonically decreases; returns a new fused
    list (input unmodified).

    Strategy: each round collects all positive-gain candidates (cancellation / d affine fusion),
    greedily matches in descending gain order (each branch used at most once), fusion segments
    re-enter the pool; stops when no positive-gain candidate remains.
    Decision and construction are all closed-form; candidate maintenance is bucketed + incremental.
    accept_neutral=True: gain == 0 fusions are also accepted (segment count still strictly decreases).
    """
    if not branches:
        return []
    n = branches[0].n
    assert all(isinstance(b, CompleteBranch) and b.n == n for b in branches)

    entries = [_Entry(b, canonical_affine(b)) for b in branches]
    if FUSION_DEBUG:
        def debug_check(e_i, e_j, fused):
            d = e_i.cb.fn() ^ e_j.cb.fn()
            assert fused.fn() == d, f'fusion contract violated: {e_i.cb!r} XOR {e_j.cb!r}'
    else:
        debug_check = None
    pool = _greedy_fusion(entries, n, max_rounds, None, debug_check, accept_neutral)
    return [e.cb for e in pool]


def shift_fn(fn, r, n):
    """Shifted function g(x) = fn(x XOR r) (the slice-root sandwich X(r) makes the in-slice
    branches act on the shifted input)."""
    out = 0
    for x in range(1 << n):
        if (fn >> (x ^ r)) & 1:
            out |= 1 << x
    return out


def fuse_pool_global(entries, n, max_rounds=None, accept_neutral=False):
    """
    Cross-slice global fusion pool (Phase 5).

    entries: list[(CompleteBranch, root_mask, key)]; root_mask is the absolute mask of the
             slice's root-edge Lim, key is the slice identity (used for emission grouping;
             fusion products get key=None).
    Returns (kept, standalone):
      kept       : list[(CompleteBranch, root_mask, key)], must be emitted inside the root
                   sandwich of the corresponding slice (grouped by key);
      standalone : list[CompleteBranch] (key=None, self-contained, may be emitted anywhere).

    Decision and construction use the shifted effective function d' = f_A(x⊕r_A) ⊕ f_B(x⊕r_B).
    Closed-form-wise this equals base ^= r (K shift-invariant), i.e. canonical_affine(cb, r);
    see the algebra doc §1.4.
    accept_neutral=True: gain == 0 fusions are also accepted (segment count still strictly decreases).
    """
    if not entries:
        return [], []
    pool_entries = [_Entry(cb, canonical_affine(cb, r), extra=(r, key))
                    for cb, r, key in entries]
    if FUSION_DEBUG:
        def debug_check(e_i, e_j, fused):
            d = shift_fn(e_i.cb.fn(), e_i.extra[0], n) ^ shift_fn(e_j.cb.fn(), e_j.extra[0], n)
            assert fused.fn() == d, f'global fusion contract violated: {e_i.cb!r} XOR {e_j.cb!r}'
    else:
        debug_check = None
    pool = _greedy_fusion(pool_entries, n, max_rounds, (0, None), debug_check, accept_neutral)
    kept = [(e.cb, e.extra[0], e.extra[1]) for e in pool if e.extra[1] is not None]
    standalone = [e.cb for e in pool if e.extra[1] is None]
    return kept, standalone


def pool_t_cost(kept, standalone):
    """ΣT of the pool (kept+standalone)."""
    return sum(cb.t_cost() for cb, _, _ in kept) + sum(cb.t_cost() for cb in standalone)


def anf_degree(bitstr, n):
    """
    The ANF degree of f = max popcount of an ANF monomial mask.

    Corollary D1 (docs/architecture.md lemma D): any branch-segment decomposition satisfies
    ΣT ≥ t(deg f), so t(anf_degree) is a **free global quality lower bound** (G-3 metric).
    Note deg is invariant under invertible affine substitutions (including FPRM shifts).
    Cost = one anf_pool (O(n·2^n), same as the ANF path).
    """
    deg = 0
    for cb, _, _ in anf_pool(bitstr, n):
        deg = max(deg, cb.mask.bit_count())
    return deg


def anf_pool(bitstr, n):
    """
    ANF decomposition pool (purely algebraic initial decomposition): the truth-table bitstr
    goes through a fast Möbius transform to obtain ANF coefficients; each monomial x^S
    (= affine cube) becomes one CompleteBranch (mask=bitstr=S, all-positive polarity).
    Returns [(CompleteBranch, root=0, key)], ready to feed fuse_pool_global.
    Pool XOR invariant: XOR of branch functions == original truth table (ANF definition).
    """
    a = bitstr
    size = 1 << n
    for j in range(n):
        step = 1 << j
        block = step << 1
        for base in range(0, size, block):
            for m in range(base, base + step):
                if (a >> m) & 1:
                    a ^= 1 << (m + step)
    entries = []
    for m in range(size):
        if (a >> m) & 1:
            entries.append((CompleteBranch(n, mask=m, bitstr=m), 0, m))
    return entries


def anf_pool_premerged(bitstr, n):
    """
    ANF decomposition pool + pre-merge of one-variable-apart monomial pairs (docs/architecture.md T-3 / 1.2).

    Exact identity (no approximation):
        x^S XOR x^{S∪{j}} = x^S(1 XOR x_j) = x^S · x̄_j        <- still an affine cube (C3 special case)
    Representation: CompleteBranch(n, mask=S∪{j}, bitstr=S) (walk segment, all-positive polarity except j).

    Pairing strategy: greedy maximal matching in ascending mask order -- for each unconsumed monomial m
    take the smallest j such that m|(1<<j) is still an unconsumed monomial, merge the two into x^m x̄_j.
    O(|pool|·n).

    Note: this is **forced** pairing (before the fusion engine); on individual cases it may be worse than
    letting the greedy engine choose on its own. Hence the production path runs it as a third decomposition
    path F' alongside F (raw ANF), taking the smaller ΣT per case, so it is never worse per case. Pre-merge
    shrinks the pool substantially (a random f's ANF pool shrinks 30%~50%), and is also a runtime
    optimization at large n (products are still plain cubes and hit the O(popcount) fast path).
    """
    monomials = set()
    a = bitstr
    size = 1 << n
    for j in range(n):
        step = 1 << j
        block = step << 1
        for base in range(0, size, block):
            for m in range(base, base + step):
                if (a >> m) & 1:
                    a ^= 1 << (m + step)
    for m in range(size):
        if (a >> m) & 1:
            monomials.add(m)

    consumed = set()
    entries = []
    for m in range(size):                    # ascending mask order, deterministic
        if not (a >> m) & 1 or m in consumed:
            continue
        partner = None
        for j in range(n):
            if (m >> j) & 1:
                continue
            p = m | (1 << j)
            if p in monomials and p not in consumed:
                partner = p
                break
        if partner is not None:
            consumed.add(m)
            consumed.add(partner)
            entries.append((CompleteBranch(n, mask=partner, bitstr=m), 0, partner))
        else:
            entries.append((CompleteBranch(n, mask=m, bitstr=m), 0, m))
    return entries
