# -*- coding: utf-8 -*-
"""
CompleteBranch: the complete information of one branch (compressed representation).

Naming note (2025-08): in the paper this object is called **Affine Block** -- the two names
denote the same concept. The algebraic essence is independent of the BDD lineage: an affine
block is the indicator function of an affine subspace S = base + span(basis), together with
its canonical emission plan (X/CNOT network + one MCX + inverse network); see
docs/CompleteBranch_algebra.md (Theorem 1, Theorem 3').

Zero-block convention (2025-08): the empty circuit is the *zero block*, realizing the
identically-zero function; at the function level this is the "zero cube" of dimension -1
(empty support). This matches is_affine_cube(fn==0) == (True, -1) and the cancellation
branch (dim = -1) of ClosedForm.fuse_closed, keeping the cost formulas exception-free.

Design: docs/architecture.md (2025-08 revision).

Primary data (immutable; absolute variable bit space, bit k = variable k):
  mask      : int   walk-edge positions (MCX control variables) mask; 0 bits = fallthrough
  bitstr    : int   walk-edge polarities (valid only on mask bits; always 0 where mask is 0)
  x_global  : int   net mask of all unconditional X's (XOR of walk-edge Lims + the lifted
                    lim0∩lim1 part of each ft position)
  ft_lims   : tuple[(ctrl, m0, m1), ...]  top-down; m0&m1==0; targets of m0/m1 both lie below ctrl
  synthetic : SyntheticPlan|None          fusion product (Phase 3); None = walk-type branch

Derived (lazily cached):
  AffineInfo(base, basis): support S = base + span(basis), basis the canonical basis of K (RREF)
  fn                      : the 2^n-bit truth table

Correctness basis (no approximation):
  1. any level's outgoing Lim only acts on "lower" variables, so the walk-edge Lim can be XORed
     into x_global as a whole;
  2. [c=0→m0]⊕[c=1→m1] = (m0∩m1) ⊕ [c=0→m0∖m1] ⊕ [c=1→m1∖m0], so the common part is lifted into x_global;
  3. an ft term's condition only depends on higher-level variables, so emitting ft_lims top-down
     reproduces the semantics exactly.
"""
from collections import namedtuple

from config import config
from circuit.Control import Control
from circuit.MCXCircuit import MCXCircuit


def t_cost(k):
    """T cost of a k-controlled MCX (single source of truth: config.py's 't_cost', near-linear new model)."""
    return config['t_cost'](k)


AffineInfo = namedtuple('AffineInfo', 'base basis')


def _bits(mask):
    """Ascending list of the set bits in mask."""
    out = []
    while mask:
        low = mask & (-mask)
        out.append(low.bit_length() - 1)
        mask ^= low
    return out

def rref(rows, n):
    """
    Reduce constraint rows (mask, const) to RREF (pivot = lowest set bit, full elimination).
    Returns [(mask, const, pivot_col), ...] (pivot columns ascending; row values are the **final**
    RREF form; math basis in CompleteBranch_algebra.md §1.2/§3.6).
    """
    row_masks = [m for m, _ in rows]
    row_consts = [c for _, c in rows]
    nrows = len(row_masks)
    pivot_cols = []
    r = 0
    for col in range(n):
        sel = None
        for i in range(r, nrows):
            if (row_masks[i] >> col) & 1:
                sel = i
                break
        if sel is None:
            continue
        row_masks[r], row_masks[sel] = row_masks[sel], row_masks[r]
        row_consts[r], row_consts[sel] = row_consts[sel], row_consts[r]
        for i in range(nrows):
            if i != r and (row_masks[i] >> col) & 1:
                row_masks[i] ^= row_masks[r]
                row_consts[i] ^= row_consts[r]
        pivot_cols.append(col)
        r += 1
    # Note: all elimination must finish before reading row values (later pivots keep clearing this column in earlier rows)
    return [(row_masks[i], row_consts[i], pivot_cols[i]) for i in range(r)]


def is_affine_cube(fn, n):
    """
    Whether fn's (2^n-bit truth table) support is an affine subspace.
    Returns (True, dim) or (False, 0); empty support (fn==0) returns (True, -1).
    """
    bits = [i for i in range(1 << n) if (fn >> i) & 1]
    if not bits:
        return True, -1
    pc = len(bits)
    if pc & (pc - 1):
        return False, 0
    base = bits[0]
    tset = {e ^ base for e in bits}
    tlist = sorted(tset)
    for i in range(len(tlist)):
        for j in range(i, len(tlist)):
            if tlist[i] ^ tlist[j] not in tset:
                return False, 0
    return True, len(tlist).bit_length() - 1


def reduce_basis(vectors):
    """Reduce a vector set to a canonical basis: distinct lowest set bits, returned in ascending pivot (lowest bit) order."""
    by_bit = {}
    for v in vectors:
        x = v
        while x:
            low = x & (-x)
            if low in by_bit:
                x ^= by_bit[low]
            else:
                by_bit[low] = x
                break
    return tuple(by_bit[low] for low in sorted(by_bit))


def in_span(x, basis):
    """Whether x is in span(basis) (basis must be the canonical reduce_basis output: pivots ascending)."""
    for b in basis:
        if x & (b & (-b)):
            x ^= b
    return x == 0


def affine_from_fn(fn, n):
    """(base, basis) of fn's support: base = lowest support point, basis = reduce_basis canonical basis."""
    bits = [i for i in range(1 << n) if (fn >> i) & 1]
    if not bits:
        return AffineInfo(0, ())
    base = bits[0]
    return AffineInfo(base, reduce_basis(e ^ base for e in bits))


class SyntheticPlan:
    """Emission plan of a fused segment: X(x_trans) → CNOT network → MCX(controls) → CNOT network (inverse) → X(x_trans)."""

    __slots__ = ('x_trans', 'cnot_ops', 'controls')

    def __init__(self, x_trans=0, cnot_ops=(), controls=()):
        self.x_trans = x_trans
        self.cnot_ops = tuple(cnot_ops)
        self.controls = tuple(sorted(controls))

    def __eq__(self, other):
        return (isinstance(other, SyntheticPlan)
                and self.x_trans == other.x_trans
                and self.cnot_ops == other.cnot_ops
                and self.controls == other.controls)

    def __hash__(self):
        return hash((self.x_trans, self.cnot_ops, self.controls))

    def __repr__(self):
        return f'SyntheticPlan(x_trans={self.x_trans:b}, cnot_ops={self.cnot_ops}, controls={self.controls})'


def plan_from_affine(base, basis, n):
    """
    RREF canonical construction (docs/CompleteBranch_algebra.md §3.6):
    affine cube e + span(basis) → SyntheticPlan(x_trans=0, cnot_ops, controls).

    Constraint rows are indexed by free columns: x_f ⊕ Σ_p (r_p)_f x_p = c_f;
    CNOT network controls = pivot variables (unmodified), targets = free variables (order irrelevant);
    constants are absorbed into the MCX polarity.

    This function keeps the **canonical form** (unique, hashable) for idempotent re-entry into the
    fusion pool; CX minimization at emission time is done by plan_min_cx (T unchanged, only a basis change).
    """
    pivots = rref([(b, 0) for b in basis], n)
    pivot_cols = {col for _, _, col in pivots}
    free_cols = [f for f in range(n) if f not in pivot_cols]
    cnot_ops = []
    controls = []
    for f in free_cols:
        c = (base >> f) & 1
        for mask, _, col in pivots:
            if (mask >> f) & 1:
                c ^= (base >> col) & 1
                cnot_ops.append((col, f))
        controls.append((f, c))
    return SyntheticPlan(0, tuple(sorted(cnot_ops)), tuple(sorted(controls)))


def _plan_for_basis(base, basis, n):
    """Build the emission plan for a given basis, returning (SyntheticPlan, CNOT count). The construction body of plan_from_affine."""
    pivots = rref([(b, 0) for b in basis], n)
    pivot_cols = {col for _, _, col in pivots}
    free_cols = [f for f in range(n) if f not in pivot_cols]
    cnot_ops = []
    controls = []
    for f in free_cols:
        c = (base >> f) & 1
        for mask, _, col in pivots:
            if (mask >> f) & 1:
                c ^= (base >> col) & 1
                cnot_ops.append((col, f))
        controls.append((f, c))
    return SyntheticPlan(0, tuple(sorted(cnot_ops)), tuple(sorted(controls))), len(cnot_ops)


def plan_min_cx(base, basis, n):
    """
    B-4: CX-optimal emission plan.

    The same affine cube has several basis representations; over the deterministic candidate set
    "canonical basis + all pairwise vector sums" pick the construction with the smallest CNOT count
    (ties keep the canonical basis first, hence deterministic). The control count is unchanged, so
    T is unchanged and only the CNOT network shrinks. The canonical plan (plan_from_affine) is still
    used for the fusion pool's idempotence/hashing.
    """
    candidates = [tuple(basis)]
    k = len(basis)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            alt = list(basis)
            alt[i] = alt[i] ^ alt[j]
            candidates.append(tuple(alt))
    best, best_cx = None, None
    for cand in candidates:
        plan, ncn = _plan_for_basis(base, cand, n)
        if best is None or ncn < best_cx:
            best, best_cx = plan, ncn
    return best


class CompleteBranch:
    __slots__ = ('n', 'mask', 'bitstr', 'x_global', 'ft_lims', 'source_branch', 'origin',
                 'synthetic', '_affine', '_fn', '_hash')

    def __init__(self, n, mask=0, bitstr=0, x_global=0, ft_lims=(),
                 source_branch=None, origin=0, synthetic=None):
        assert n >= 0
        assert bitstr & ~mask == 0, 'bitstr may only set bits on mask positions'
        assert synthetic is None or (mask == 0 and bitstr == 0 and x_global == 0 and not ft_lims), \
            'synthetic segment and walk fields are mutually exclusive'
        for ctrl, m0, m1 in ft_lims:
            assert m0 & m1 == 0, 'ft term must be normalized: m0 & m1 == 0'
            assert not ((mask >> ctrl) & 1), 'ft control variable must not also be a walk-edge position'
            assert not ((m0 | m1) >> ctrl) & 1, 'ft targets must not include their own control variable'

        self.n = n
        self.mask = mask
        self.bitstr = bitstr
        self.x_global = x_global
        self.ft_lims = tuple(ft_lims)
        self.source_branch = source_branch
        self.origin = origin
        self.synthetic = synthetic
        self._affine = None
        self._fn = None
        self._hash = None

    # ------------------------------------------------------------ structure ----

    def is_walk(self):
        return self.synthetic is None

    def nctrl(self):
        """MCX control count (walk-type = mask popcount; synthetic = number of controls)."""
        if self.synthetic is not None:
            return len(self.synthetic.controls)
        return self.mask.bit_count()

    def controls(self):
        """MCX control set: sorted tuple[(var, polar)]."""
        if self.synthetic is not None:
            return self.synthetic.controls
        return tuple((v, (self.bitstr >> v) & 1) for v in _bits(self.mask))

    def t_cost(self):
        return t_cost(self.nctrl())

    def is_constant_one(self):
        return self.nctrl() == 0

    # ------------------------------------------------------------ derived ----

    def affine(self):
        """Symbolic derivation of the support S = base + span(basis) (O(n^2), no 2^n enumeration)."""
        if self._affine is None:
            self._affine = self._compute_affine()
        return self._affine

    def fn(self):
        """Branch function (2^n-bit truth table)."""
        if self._fn is None:
            base, basis = self.affine()
            fn = 0
            for s in range(1 << len(basis)):
                k = 0
                for j in range(len(basis)):
                    if (s >> j) & 1:
                        k ^= basis[j]
                fn |= 1 << (base ^ k)
            self._fn = fn
        return self._fn

    def direction_key(self):
        """Canonical-basis tuple of the direction subspace K (fusion-candidate bucket key)."""
        return self.affine().basis

    def cancel_key(self):
        """Function truth table (equality implies cancellability)."""
        return self.fn()

    def _constraint_rows(self):
        """Constraint rows (mask, const) of the branch function (equation mask·x ≡ const)."""
        n = self.n
        masks = [1 << v for v in range(n)]
        consts = [0] * n

        if self.synthetic is not None:
            plan = self.synthetic
            for v in range(n):
                consts[v] = (plan.x_trans >> v) & 1
            for c, t in plan.cnot_ops:
                masks[t] ^= masks[c]
                consts[t] ^= consts[c]
            return [(masks[v], consts[v] ^ polar) for v, polar in plan.controls]

        for v in range(n):
            consts[v] = (self.x_global >> v) & 1
        for ctrl, m0, m1 in self.ft_lims:
            cmask, cconst = masks[ctrl], consts[ctrl]
            for t in _bits(m0):
                masks[t] ^= cmask
                consts[t] ^= cconst ^ 1
            for t in _bits(m1):
                masks[t] ^= cmask
                consts[t] ^= cconst
        return [(masks[v], consts[v] ^ ((self.bitstr >> v) & 1)) for v in _bits(self.mask)]

    def affine_rows(self):
        """RREF canonical form of the constraint rows: tuple[(pivot, free_mask, const)] (pivots ascending)."""
        pivots = rref(self._constraint_rows(), self.n)
        return tuple((col, mask ^ (1 << col), const) for mask, const, col in pivots)

    def _compute_affine(self):
        pivots = rref(self._constraint_rows(), self.n)
        pivot_cols = {col for _, _, col in pivots}

        basis = []
        for f in range(self.n):
            if f in pivot_cols:
                continue
            v = 1 << f
            for mask, _, col in pivots:
                if (mask >> f) & 1:
                    v |= 1 << col
            basis.append(v)

        base = 0
        for _, const, col in pivots:
            if const:
                base |= 1 << col
        return AffineInfo(base, tuple(basis))

    # ------------------------------------------------------------ equality ----

    def _key(self):
        return (self.n, self.mask, self.bitstr, self.x_global, self.ft_lims, self.synthetic)

    def __eq__(self, other):
        return isinstance(other, CompleteBranch) and self._key() == other._key()

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._key())
        return self._hash

    def __repr__(self):
        if self.synthetic is not None:
            return f'CompleteBranch(n={self.n}, synthetic={self.synthetic})'
        ft = ','.join(f'({c},{m0:b},{m1:b})' for c, m0, m1 in self.ft_lims)
        return (f'CompleteBranch(n={self.n}, mask={self.mask:b}, bitstr={self.bitstr:b}, '
                f'x_global={self.x_global:b}, ft_lims=[{ft}])')

    # ------------------------------------------------------------ emission ----

    def to_circuit(self, qubit_mgr, target=None):
        """
        Emit as an MCXCircuit (Phase 4 entry point; used here to reconcile against compute_tt).
        The caller must have already allocated all n input qubits.
        """
        circ = MCXCircuit(qubit_mgr, target)
        if self.synthetic is not None:
            # B-4: emission uses the CX-optimal basis (the stored plan keeps the RREF canonical form; control count unchanged, so T unchanged)
            plan = plan_min_cx(*self.affine(), self.n)
            x_qubits = [qubit_mgr.get_input_qubit(v) for v in _bits(plan.x_trans)]
            circ.add_lim_operator(x_qubits)
            for c, t in plan.cnot_ops:
                circ.add_lim_operator([qubit_mgr.get_input_qubit(t)],
                                      Control(qubit_mgr.get_input_qubit(c), 1))
            circ.add_gate([Control(qubit_mgr.get_input_qubit(v), p) for v, p in plan.controls],
                          circ.target)
            for c, t in reversed(plan.cnot_ops):
                circ.add_lim_operator([qubit_mgr.get_input_qubit(t)],
                                      Control(qubit_mgr.get_input_qubit(c), 1))
            circ.add_lim_operator(x_qubits)
            return circ

        x_qubits = [qubit_mgr.get_input_qubit(v) for v in _bits(self.x_global)]
        circ.add_lim_operator(x_qubits)
        for ctrl, m0, m1 in self.ft_lims:
            c = qubit_mgr.get_input_qubit(ctrl)
            circ.add_lim_operator([qubit_mgr.get_input_qubit(t) for t in _bits(m0)],
                                  Control(c, 0))
            circ.add_lim_operator([qubit_mgr.get_input_qubit(t) for t in _bits(m1)],
                                  Control(c, 1))
        circ.add_gate([Control(qubit_mgr.get_input_qubit(v), p) for v, p in self.controls()],
                      circ.target)
        for ctrl, m0, m1 in reversed(self.ft_lims):
            c = qubit_mgr.get_input_qubit(ctrl)
            circ.add_lim_operator([qubit_mgr.get_input_qubit(t) for t in _bits(m0)],
                                  Control(c, 0))
            circ.add_lim_operator([qubit_mgr.get_input_qubit(t) for t in _bits(m1)],
                                  Control(c, 1))
        circ.add_lim_operator(x_qubits)
        return circ


# ---------------------------------------------------------------- extraction ----
