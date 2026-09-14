# Algebraic Foundations of CompleteBranch Branch Fusion

> **Naming note**: the code object `CompleteBranch` and the paper term **Affine Block** are
> two names for the same concept — the essence of the object is independent of the BDD lineage; it is
> "the indicator function of an affine subspace + a canonical emission plan"
> (Theorem 1 and Theorem 3′). The two terms are used interchangeably in this document.
>
> This file is the algebraic-foundation document of the framework; it states only the mathematics and
> does not cover implementation details. See `docs/architecture.md` for the system-level design.
> All conclusions were verified under the probe framework of `experiments/phase0_branch_stats.py`;
> the verification methods and numbers are listed in the appendices at the end.
>
> Notation conventions are given in §0. It is recommended to go through the §8 checklist item by item.

---

## 0. Notation and conventions

| Notation | Meaning |
|---|---|
| `V = GF(2)^n` | the binary vector space spanned by the n input variables; vectors are written as binary strings or bitmask integers, **bit k = variable k** (absolute variable bit space) |
| `⊕` | GF(2) addition (XOR of bits / vectors / integers) |
| `⟨v₁,…,v_k⟩` / `span{…}` | the linear subspace spanned by vectors |
| `S △ T` | symmetric difference of sets `(S∖T) ∪ (T∖S)` |
| `supp(f)` | support of the function f, `{x : f(x) = 1}` |
| `1_S(x)` | indicator function of the set S |
| `t(c)` | T cost of a c-controlled MCX (looked up in `T_COST`; X/CX/CNOT count as 0) |
| `f_A` | flip function of branch segment A on the target bit (`f_A : V → GF(2)`) |
| affine subspace | a set of the form `S = a + K`, K a linear subspace (the direction), a the shift; `dim S := dim K` |

**Object-family definitions (used throughout the document)**:

- **Affine cube function**: for dimension `k ≥ 0`, the indicator function `1_S` of an affine subspace
  `S = a + K` (`dim K = k`); **for dimension `k = −1`, the identically-zero function `0`** (support `∅`).
  Strictly speaking `∅` is not an affine subspace — this document uses the term **zero cube** only at
  the **function level**; at the set level we always write `supp(f) = ∅` literally.
  **−1 is a "default" sentinel value and does not participate in the two formulas defined only for k ≥ 0**:
  the cardinality formula `|S| = 2^k` and the control-count formula `m = n − k` are not extended to −1
  (otherwise m = n+1 would be meaningless); the only quantity extended to −1 is the cost `t̄(−1) := 0`
  (see below). The reason not to redefine the dimension of ∅ to some other value: if we set `dim ∅ = n`,
  it collides with `1_V` (support V, dim n), breaking the "dim ↔ support" injection and all the cardinality
  arguments of Theorem 2; if we set it to −∞, the operational form of `|S| = 2^k` breaks. Thus −1 is the
  only choice that preserves the integrity of those two formulas.
- **Affine Block (= the code's CompleteBranch)**: an involution circuit on the data
  register
  `X network · CNOT network · MCX(C, p) · CNOT inverse network · X network`, where **the MCX may be omitted**;
  when omitted the circuit is empty and is called the **zero block**, realizing the zero function with
  T cost 0.
  The zero block is the representation-level resting place of C1 cancellation (Theorems 1/2 below).
- **Correspondence between zero cube and zero block**: the affine cube function of `dim = −1` is exactly
  the one realized by the zero block; a zero block is removed from the pool as soon as it is produced
  (C1 two-step merge), so the pool invariant can be stated as "the pool never contains a zero block".

Two facts used throughout:

1. **Circuit additivity**: when all branch segments are concatenated in order, the net flip on the target
   bit = XOR of the individual flip functions (each segment is an involution: the input is restored and
   the target flips according to `f_A(x)`, so sequential execution = term-by-term addition).
2. **Fusion identity**: the entire algebraic content of branch fusion is the set/function identity
   `{A, B} → {A⊕B}` (replacing `f_A + f_B` by `f_A ⊕ f_B` in the pool's XOR sum).

---

## 1. Circuit semantics of a branch segment and its function

### 1.1 Definition of a branch segment

A **branch segment (BranchSegment)** is described by a top-down sequence of positions, one `Position` per level:

```
Position = (var, ft, bit, w0, w1, lim0, lim1)
   var        : absolute index of the decision variable at this level
   ft ∈ {0,1} : whether this level is fallthrough (the low/high edges point to the same child and w0=w1=1)
   bit ∈ {0,1}: the 0/1 choice of the edge taken at a non-ft level (meaningless at an ft level, written -)
   w0, w1     : weights of the low/high edges
   lim0, lim1 : Lim flip masks of the low/high edges (integers in the absolute variable bit space, bit k = variable k)
```

Emission semantics (mirroring `branch2circ` in `backend/OracleSynthesizer.py` level by level):

- **Non-ft level**: first apply the unconditional flip `X(lim_bit)` (the Lim mask of the chosen edge, acting
  on **deeper variables**, not including this level's variable), then place the entire subsequent circuit
  under the control `x_var == bit`; at the end apply `X(lim_bit)` once more symmetrically (uncomputation).
- **ft level**: apply the two conditional flips `[x_var=0 → X(lim0)]` and `[x_var=1 → X(lim1)]`, and the
  subsequent circuit continues **unconditionally** (this level produces no control bit); symmetric
  uncomputation at the end.
- **Termination**: the target bit is flipped unconditionally (X gate, effective jointly with the accumulated
  control conditions).

Weight convention: at a non-ft level `w_bit` must be 1 (a weight-0 edge is not on the branch); nodes with
`w0=w1=0` do not appear in any branch.

### 1.2 State-evolution equations

Let the input be `x`, and let `e_i` be the **effective state** after the first i levels (before entering level
i+1). Level by level:

- non-ft level: `e_i = e_{i-1} ⊕ lim_bit` (constant flip);
- ft level: `e_i = e_{i-1} ⊕ A_var(e_{i-1})`, where
  `A_var(z) = lim0 ⊕ z_var · (lim0 ⊕ lim1)` (conditional flip, `z_var` the current value of that variable);
- the decision condition of each non-ft level is `(e_{i-1})_{var} == bit`;
- finally `f(x) = 1` ⟺ all non-ft decision conditions hold simultaneously.

### 1.3 Theorem 1 (Affine Block ⟺ affine cube function: equivalence theorem)

**Theorem 1**:

1. **Necessity (⟹)**: the flip function of every Affine Block is an affine cube function; moreover, the
   support dimension of the flip function of an m-controlled (MCX control count m) Affine Block is exactly
   `n − m`; the MCX is omitted (zero block) iff the flip function is ≡ 0 (zero cube, dim = −1).
2. **Sufficiency (⟸)**: every affine cube function can be realized by an Affine Block: `dim = −1` corresponds
   to the zero block; `dim = k ≥ 0` corresponds to the RREF canonical construction of §3.6, with MCX control
   count exactly `n − k`.

**Preparation Lemma 1.3.a (all basic gates on the data register are affine involutions)**: X, CNOT, and any
"data-controlled conditional flip" `[x_c = p → X(t)]` (c a data bit, t the flipped bit) are invertible affine
transformations on `V = GF(2)^n`, and each is an involution.

*Proof*: X is the translation `y ↦ y ⊕ x_trans`; CNOT(c→t) is the transvection `y ↦ y ⊕ y_c·e_t` (linear).
For a conditional flip: its indicator function can be written `1_{x_c = p} = x_c ⊕ p ⊕ 1` (verify bit by bit
over GF(2): when `x_c = p` the right side is 1, otherwise 0), hence
`y = x ⊕ e_t · 1_{x_c = p} = x ⊕ e_t·x_c ⊕ (p⊕1)·e_t` — the linear part `I + e_t e_c^T` is invertible
(an involution), the constant term is `(p⊕1)·e_t`. All three are involutions. ∎

**Corollary 1.3.b (the prefix and suffix of the sandwich network are mutually inverse)**: if
`N = g_1 ∘ … ∘ g_k` is a product of affine involutions, then N is invertible and
`N⁻¹ = g_k ∘ … ∘ g_1` (each factor self-inverse + reverse order). Hence every block of the form
"affine network N · MCX · reverse network" automatically satisfies involutivity (input restoration); this is
the algebraic origin of the §6.2 sandwich requirement.

**(1) Complete proof of necessity**:
1. Let N be the data-network prefix of an Affine Block B (a product of X/CNOT/conditional flips). By Lemma 1.3.a
   and Corollary 1.3.b, `N(x) = Lx ⊕ b`, `L ∈ GL(n,2)` (a product of affine maps is affine; each factor
   invertible ⇒ L invertible), and the suffix in the emission semantics is exactly `N⁻¹`. Thus the net effect
   of the whole block on the register = `N · MCX · N⁻¹`.
2. **When the MCX is present**: the MCX flips the target ⟺ `N(x)_C = p_C` ⟺ for every control coordinate
   i ∈ C: `ℓ_i(x) = c_i`, where `ℓ_i(x) = (Lx)_i` is a linear form in x and `c_i = p_i ⊕ b_i`. Since L is
   invertible, its i-th rows are pairwise linearly independent ⇒ these are `m = |C|` **linearly independent**
   affine equations (hyperplanes). The intersection of finitely many hyperplanes is an affine subspace
   (nonempty: acting on a point satisfying the polarities with `N⁻¹` yields a solution) of dimension
   `n − rank{ℓ_i} = n − m`. Hence `supp(f_B) = S = a + K`, `dim K = n − m`, and f_B is an affine cube
   function of dimension n − m.
3. **When the MCX is omitted (zero block)**: no conditionally controlled target flip acts on the register
   (the block is the empty circuit, or only the X/CNOT networks of the prefix and suffix remain — note that
   prefix N and suffix N⁻¹ cancel each other and do not flip the target), so `f_B ≡ 0` = zero cube (dim −1).
4. **Uniqueness in the reverse direction**: if `f_B ≡ 0`, the block must contain no MCX — with an MCX present
   the support is a **nonempty** affine subspace (an affine subspace always contains its base point a) and
   cannot be identically zero. Therefore `f ≡ 0 ⟺ zero block`. ∎

**(2) Complete proof of sufficiency**:
1. `dim = −1` (`f ≡ 0`): the zero block (empty circuit) realizes it. ∎
2. `dim = k ≥ 0` (`f = 1_{a+K}`): take any basis `B = {b_1, …, b_k}` of K and run the RREF canonical
   construction of §3.6. Its correctness reduces to the following **membership-characterization lemma**:

   **Lemma 1.3.c (parametrization of the RREF row space by the free columns)**: let B be reduced to RREF with
   "pivot = lowest set bit, full elimination", giving pivot columns P, free columns F = [n]∖P and rows
   `r_p` (p ∈ P). Then `z ∈ span(B)` ⟺ for every free column `f ∈ F`:
   `z_f ⊕ Σ_{p ∈ P} (r_p)_f · z_p = 0`.

   *Proof*: (⊆) check each RREF row `r_q` (pivot q) term by term: at a free column f `(r_q)_f = (r_q)_f`;
   in `Σ_{p ∈ P} (r_p)_f · (r_q)_p`, full elimination makes `(r_q)_p = 1` iff `p = q`,
   so the sum = `(r_q)_f`; the two sides are equal ⇒ the row satisfies the equation. The equation system is
   linear and is satisfied by the basis ⇒ the whole span(B) satisfies it. (⊇) every free variable `z_f`
   appears in exactly one equation as the "unique free term", so the |F| equations are linearly independent,
   and the solution-space dimension = `n − |F| = |P| = k = dim span(B)`; an inclusion of equal dimension is
   an equality. ∎

3. Substitute `z = x ⊕ a` into Lemma 1.3.c: `x ∈ a + K ⟺` for every `f ∈ F`,
   `x_f ⊕ Σ_{p∈P} (r_p)_f x_p = c_f`, where `c_f = a_f ⊕ Σ_{p∈P} (r_p)_f a_p`.
4. Circuit construction: for each `(p, f)` with `(r_p)_f = 1` apply `CNOT(p → f)`. The control set ⊆ P,
   the target set ⊆ F, and `P ∩ F = ∅` ⇒ no target is later used as a control ⇒ the CNOTs do not interfere
   and any order is equivalent; after applying them the free coordinates become
   `y_f = x_f ⊕ Σ_{p: (r_p)_f=1} x_p = x_f ⊕ Σ_{p∈P} (r_p)_f x_p`.
   Then apply the MCX with control set F and polarities `c_f`: it flips the target ⟺ `y_f = c_f` (∀f) ⟺
   `x ∈ a + K`. Repeating the CNOTs in reverse order (Corollary 1.3.b) restores the input, yielding a legal
   Affine Block with control count `m = |F| = n − |P| = n − k`. ∎

**(i) Minimal control count (secondary conclusion, and stronger than the earlier statement: it is an
equality, not merely a lower bound)**: first define the extended cost
`t̄(dim) := t(n − dim)` (dim ≥ 0), `t̄(−1) := 0`. Then for dim = k ≥ 0: by necessity, any Affine Block
realizing the function satisfies `n − m = k`, i.e. **m = n − k is forced**, and the single-block T cost
`T = t(n − k) = t̄(k)`. For dim = −1 (f ≡ 0): only the zero block can realize it, **the MCX is omitted and m
is undefined** (the control-count formula `m = n − k` does not apply to this case; do not write
`m = n + 1`), cost `T = 0 = t̄(−1)`. In summary `T_min(f) = t̄(dim f)` holds for all
`dim ∈ {−1} ∪ [0, n]`, and **every** correct realization attains it.

**(ii) The canonical construction attains the lower bound and is unique (secondary conclusion)**:
- **Attainment**: step 4 of the sufficiency proof gives `m = n − k`.
- **Uniqueness/canonicity**: classical uniqueness of RREF — two RREFs of the same row space are equal (proof
  sketch: the pivot positions are uniquely determined by "whether column j lies in the space spanned by the
  first j−1 columns"; once the pivot columns are given, full elimination makes each pivot row 1 at its pivot
  and 0 in the other pivot columns, so the row contents are uniquely determined by the linear-combination
  coefficients). Our additional deterministic conventions (pivot scan order = ascending columns taking the
  lowest set bit; cnot_ops in ascending (p,f) order; controls in ascending f order) make the output of §3.6
  a **function** of **(a, K)** — independent of the choice of input basis B. All constants are absorbed into
  the MCX polarity (convention `x_trans = 0`), so `SyntheticPlan(0, cnot_ops, controls)` is hashable and
  equality-comparable, and fusion products are idempotent on re-entry to the pool (the same (a,K) always
  generates the same plan). ∎

**Corollary (characterization of affine cube functions by linear forms)**: an affine cube function of
dimension k is exactly one writable as `f(x) = Π_{j=1}^{n−k} [ℓ_j(x) = c_j]`, where the `ℓ_j` are
`n − k` linearly independent linear forms (a basis of the annihilator of the direction subspace K),
`c_j = ℓ_j(a)`. (Proof: the constraint equations of step 3 of the sufficiency construction are obtained by
taking `ℓ_f(x) = x_f ⊕ Σ_p (r_p)_f x_p`.)

**Relation to the BDD lineage (faithfulness note, not part of the theorem content)**: the
Position / lim / weight / fallthrough walk semantics of §1.1–§1.2 is another implementation-level
description of the same object given by the **A-path extraction** — each walk-type branch segment, expanded
level by level, is exactly the sandwich form of Theorem 1, hence the output of `extract_complete_branches`
(`limbdd/Extraction.py`) lies entirely in the affine-cube-function family. Measured: for 38 sampled real
extracted branch segments, every `supp(f)` passed the affine test (Appendix A). The theorem does not depend
on this historical argument.

**Example** (pure MCX segment, no Lim at all): control set C, polarity vector c, then
`supp = {x : x_C = c}`, `K = span{x_j : j ∉ C}`, `dim K = n − |C|`.

**Example** (the two ends: zero block / constant segment): an all-ft chain with zero controls has support
`= V`, `dim K = n` (one of the degenerate ends of MCX omission); the zero block has support `= ∅`,
`dim = −1` (the other degenerate end).

**Example** (with a conditional flip): the flip introduced on deeper variables by the ft level
`[x_var=0 → X(lim0)] ⊕ [x_var=1 → X(lim1)]` is `lim0 ⊕ x_var·(lim0⊕lim1)` — an affine dependence, so
"support = affine subspace" is preserved.

### 1.3.1 Single-segment T optimality lemma (a direct restatement of
Theorem 1 (i)+(ii))

**Proposition**: if `supp(f) = a + K`, `dim K = k`, then the MCX control count of any **single branch
segment** realizing f is **exactly** `n − k` (the equality is forced, not merely a lower bound), and the RREF
canonical construction of §3.6 gives that realization. Therefore the single-segment T cost is
`T_min(f) = t(n − k)`, and every correct realization attains it.

**Proof**: the direct combination of Theorem 1's necessity (`n − m = k`) and sufficiency (the construction
attains `n − k`); see Theorem 1 (i), (ii). The zero-cube (dim = −1) case has no MCX and an undefined control
count; its cost is charged as `t̄(−1) = 0` (Theorem 1(i)).

### 1.4 Root-edge Lim and inter-piece concatenation

The root-edge Lim `r` of each subgraph (piece) is realized by the circuit as the sandwich
`X(r) · (the piece's branch segments) · X(r)`. Hence the relation between the piece-internal segment function
`f_A` and the whole-piece function is `F_piece(x) = ⊕_A f_A(x ⊕ r)`.

When two pieces (root Lims `r_A, r_B`) are concatenated, the net contribution of one pair of branch segments is

```
X(r_A) F_A X(r_A) X(r_B) F_B X(r_B)
 = X(r_A) · F_A · X(r_A ⊕ r_B) · F_B · X(r_B)     (group property of X gates)
```

The net flip function = `f_A(x ⊕ r_A) ⊕ f_B(x ⊕ r_B)`. Put `y = x ⊕ r_A`, then
`= f_A(y) ⊕ f_B(y ⊕ (r_A ⊕ r_B))`. **Conclusion**: the support of the net function of cross-piece fusion is
`S_A △ (S_B ⊕ (r_A ⊕ r_B))` — i.e. shift B's support by `r_A ⊕ r_B` and take the symmetric difference with
`S_A`; the direction subspace K of each branch is invariant under translation. The leftover X-gate groups at
the two ends (`X(r_A)`, `X(r_B)`) are T-free global factors.

**Translation sensitivity (confirmed by the current implementation)**: K invariance does
**not** mean "the fusion decision is invariant" — the **coset-position conditions of Theorem 2 are
translation-sensitive**: the inclusion condition of C3 (`a_A⊕a_B ∈ K_large`) and the intersection condition
of C4 (`a_A⊕a_B ∈ K_A+K_B`) both change when B is shifted by `w = r_A⊕r_B`; hence fusibility of (A, B) does
not imply fusibility of (A, B⊕w), and vice versa. **Global cross-piece fusion must re-run the direct decision
on the shifted effective function `d' = f_A(x⊕r_A) ⊕ f_B(x⊕r_B)`** (support
`S_A △ (S_B ⊕ (r_A⊕r_B))`, direction subspace the same as without the shift, only the base shifted) — the
implementation `fuse_pool_global` in `backend/fusion/BranchFusion.py` does exactly this. The branch-probe
statistics that ignored root masks are an "opportunity upper bound" statement; the realized figures (n=7
ΣT −21.9% vs upper bound 19.8%) show that, under the actual root-mask distribution, shifts that destroy
fusibility are rare.

---

## 2. Symmetric-difference classification of affine subspaces (the fusion criterion)

### 2.1 Basic operations

Let `S_1 = a_1 + K_1`, `S_2 = a_2 + K_2`, `k_i = dim K_i`, `W = K_1 ∩ K_2`, `m = dim W`.

- Intersection: when nonempty, `S_1 ∩ S_2` is a coset of W, `|S_1 ∩ S_2| = 2^m`; empty ⟺
  `a_1 ⊕ a_2 ∉ K_1 + K_2`.
- Translation: `S ⊕ v = (a ⊕ v) + K`.
- Inclusion: `S_2 ⊆ S_1` ⟺ `K_2 ⊆ K_1` and `a_1 ⊕ a_2 ∈ K_1`.

**Lemma 2.0 (codimension-1 decomposition into two cosets)**: let `W ⊆ K` be linear
subspaces with `dim W = dim K − 1`. Then K is exactly the union of two W-cosets: `K = W ⊔ (u + W)`, where
`u ∈ K ∖ W` is arbitrary.

**Proof**: the quotient space `K/W` has dimension `dim K − dim W = 1`, so `K/W ≅ GF(2)` has exactly 2
elements, i.e. K contains exactly 2 W-cosets. W itself is one coset; take any `u ∈ K ∖ W` (`W ⊊ K`
guarantees it exists), `u + W` is the other coset. They are disjoint: if `x ∈ W ∩ (u+W)`, there exist
`w, w′ ∈ W` with `x = w = u ⊕ w′`, giving `u = w ⊕ w′ ∈ W`, contradicting `u ∉ W`. The two cosets are
mutually exclusive and `2 · |W| = |K|`, hence `K = W ⊔ (u + W)`. ∎

### 2.2 Theorem 2 (symmetric-difference classification — necessary and sufficient condition for fusion)

**Theorem 2**: `S_1 △ S_2` is an affine subspace if and only if one of the following four cases holds:

| Case | Condition | Result F = S₁ △ S₂ | dim F |
|---|---|---|---|
| **C1** | `S_1 = S_2` (i.e. `K_1=K_2` and `a_1⊕a_2 ∈ K_1`) | **zero cube** (identically-zero function, `supp(F) = ∅`) | **−1** |
| **C2** | `K_1 = K_2` and `S_1 ∩ S_2 = ∅` (i.e. `a_1⊕a_2 ∉ K_1`) | `a_1 + (K_1 + ⟨a_1⊕a_2⟩)` | `k_1 + 1` |
| **C3** | proper nesting with codimension 1: `K_2 ⊊ K_1`, `dim K_2 = dim K_1 − 1`, and `S_2 ⊆ S_1` (or symmetric) | `S_1 ∖ S_2` = the other coset of `K_2` inside `S_1` (direction still `K_2`) | `k_2 = k_1 − 1` |
| **C4** | crossing, equal dimension: `k_1 = k_2`, `m = k_1 − 1`, and neither contains the other | `(t ⊕ u_1) + (W + ⟨u_1 ⊕ u_2⟩)`, where `t ∈ S_1∩S_2`, `u_1 ∈ K_1∖W`, `u_2 ∈ K_2∖W` | `k_1` |

**Proof** (necessity + sufficiency).

*Sufficiency*:

- C1: trivial — `S_1 △ S_2 = ∅`, the result is the zero cube (dim −1, §0 object-family definition),
  corresponding to the zero block.
- C2: `S_1 ⊔ S_2 = a_1 + (K_1 ∪ (d + K_1))`, `d = a_1 ⊕ a_2`. It remains to prove
  `K ∪ (d+K) = K + ⟨d⟩` (i.e. closure): take any `x, y ∈ K ∪ (d+K)`,
  - if both are in K: `x⊕y ∈ K`; both in `d+K`: `(d+x')⊕(d+y') = x'⊕y' ∈ K`;
  - one in K, one in `d+K`: `d ⊕ x' ⊕ y ∈ d+K`.
  Hence the union is closed under ⊕ and contains 0, so it is a subspace, and `= K + ⟨d⟩` (dimension
  `k_1+1`, because `d ∉ K_1`). ∎
- C3: `K_1 = K_2 ⊔ (u + K_2)` (codimension 1), so `S_1 ∖ S_2 = t + u + K_2` is a coset. ∎
- C4: **Step 1 (existence of u₁, u₂)**: the C4 conditions give `m = k_1 − 1 = k_2 − 1`; by Lemma 2.0,
  `K_1 = W ⊔ (u_1 + W)`, `K_2 = W ⊔ (u_2 + W)`, so `u_1 ∈ K_1∖W`, `u_2 ∈ K_2∖W` exist.
  **Step 2 (re-represent so that both decompositions share the same W-coset)**: the coset-intersection
  condition of C4 (`a_1 ⊕ a_2 ∈ K_1 + K_2`, equivalent to `S_1 ∩ S_2 ≠ ∅`) gives `t ∈ S_1∩S_2`. A coset can
  be represented by any representative element, so `S_1 = a_1+K_1 = t+K_1 = (t+W) ⊔ (t+u_1+W)`,
  `S_2 = a_2+K_2 = t+K_2 = (t+W) ⊔ (t+u_2+W)`.
  **Step 3 (the common part is exactly the whole intersection and cancels in the symmetric difference)**:
  `W ⊆ K_1∩K_2` ⇒ `t+W ⊆ S_1∩S_2`; moreover `|S_1∩S_2| = 2^m = |t+W|` (§2.1 intersection formula), so
  `S_1∩S_2 = t+W` is exactly the whole intersection. Thus
  `S_1 △ S_2 = (t+u_1+W) ⊔ (t+u_2+W) = (t⊕u_1) + (W ∪ ((u_1⊕u_2)+W))`.
  The same closure argument as C2: `W ∪ ((u_1⊕u_2)+W) = W + ⟨u_1⊕u_2⟩`; and `u_1⊕u_2 ∉ W`
  (if `u_1⊕u_2 ∈ W ⊆ K_1`, then `u_2 = u_1⊕w ∈ K_1∩K_2 = W`, contradicting `u_2 ∉ W`), so
  `dim(W + ⟨u_1⊕u_2⟩) = k_1`, consistent with `dim F = k_1`. ∎

*Necessity*: suppose `F = S_1 △ S_2` is an affine subspace of dimension k.

- If `S_1 = S_2`: C1.
- If `S_1 ∩ S_2 = ∅`: `F = S_1 ⊔ S_2` is an affine set and is covered as the disjoint union of two cosets.
  `|F| = 2^{k_1} + 2^{k_2} = 2^k`. If `k_1 ≠ k_2`, let `k_1 > k_2`; then
  `2^k = 2^{k_2}(2^{k_1−k_2}+1)`, whose parenthesized factor is an odd number greater than 1,
  contradiction; hence `k_1 = k_2` and `k = k_1 + 1`. Now `S_1` is a codimension-1 coset inside F
  (`dim S_1 = k−1 = dim F − 1`), so `F ∖ S_1 = S_2` is the other coset of `K_1` (the complement of a
  hyperplane coset is exactly the other coset), hence `S_2 = a' + K_1`, i.e. `K_2 = K_1`: C2. ∎
- If `S_1 ∩ S_2 ≠ ∅` (intersection `I = t + W`, `|I| = 2^m`):
  `|F| = 2^{k_1} + 2^{k_2} − 2^{m+1}`. Assume `k_1 ≥ k_2` and write
  `|F| = 2^{k_2} · (2^{k_1−k_2} + 1 − 2^{m+1−k_2})`.
  - `m = k_2` (i.e. `S_2 ⊆ S_1`): `|F| = 2^{k_1} + 2^{k_2} − 2^{k_2+1} = 2^{k_2}(2^{k_1−k_2} − 1)`, the
    parenthesized factor is odd, and it is a power of 2 ⟺ `= 1`, i.e. `k_1 = k_2 + 1`. Then
    `|F| = 2^{k_2}`, `F = S_1 ∖ S_2` is a coset of `K_2` (`K_2` has codimension 1 in `K_1`, `K_1` is
    exactly the union of the two cosets of `K_2`, removing `S_2` leaves the other coset): C3. ∎
  - `m ≤ k_2 − 1`: `2^{m+1−k_2} ≤ 1`. If `m ≤ k_2 − 2`, that number is a proper fraction, the parenthesized
    factor is not an integer, `|F|` is not a power of 2, contradiction; hence necessarily `m = k_2 − 1`, the
    parenthesis `= 2^{k_1−k_2}`, `|F| = 2^{k_1}`.
    - If `k_1 > k_2`: excluded by the counterexample below, contradiction;
    - If `k_1 = k_2`: `m = k_1 − 1` and neither contains the other: C4. ∎

*Counterexample (excluding `k_1 > k_2, m = k_2 − 1`)*: `n = 3`, `S_1 = {x_3=0}` (a plane,
`K_1 = span{e_1,e_2}`, `a_1=0`), `S_2 = {x_1=x_2=0}` (a line, `K_2 = span{e_3}`, `a_2=0`). Then
`k_1=2, k_2=1, W = {0}, m=0=k_2−1`, neither contains the other, and `S_1 ∩ S_2 = {000}`.
`F = S_1△S_2 = {100, 010, 110, 001}`. Taking the three points `100 ⊕ 010 ⊕ 001 = 111 ∉ F`, F does not
satisfy the closure condition for affine sets (F is affine ⟺ `u,v,w ∈ F ⟹ u⊕v⊕w ∈ F`), so it is not affine.
Moreover every configuration of the same type (`k_1=2, k_2=1, m=0`, neither contains the other) is equivalent
to this standard configuration under the affine group, so this class is **never** fusible. ∎

> Note: the closure criterion for affine sets `u⊕v⊕w ∈ F` (for any three points) comes from:
> `F = a + K` ⟺ after translation it is a subspace ⟺ closed under ⊕. The criterion applies to **nonempty**
> affine sets; the empty-set case (C1) is handled separately by the zero-cube definition of §0.

### 2.3 Two equivalent forms of the fusion criterion

1. **Direct decision** (used by the branch probe; simple to implement and always correct): compute the
   support of `d = f_A ⊕ f_B` and check whether `supp(d)` is an affine subspace (the closure criterion of 2.2
   or Gaussian elimination). For cross-piece fusion use the shifted `d'` (§1.4 revision).
2. **Structural decision** (for matching, O(poly(n))): compare `(a_i, K_i)` according to §4.2 (including the
   cross-piece shift `r_A ⊕ r_B`) and apply the four cases of Theorem 2.

**Implementation mapping (`four_case_kind` in `backend/fusion/BranchFusion.py` ↔
Theorem 2)**:

| Code number | Document case | Condition |
|---|---|---|
| 1 | C1 + C2 | K_A = K_B (same coset → cancellation; different cosets → C2 fusion) |
| 4 | C3 | K_B ⊊ K_A, dim difference 1, S_B ⊆ S_A (a_A⊕a_B ∈ K_A) |
| 3 | C3 symmetric | K_A ⊊ K_B, dim difference 1, S_A ⊆ S_B (a_A⊕a_B ∈ K_B) |
| 2 | C4 | equal dimensions, dim(K_A∩K_B) = k−1, cosets intersect (a_A⊕a_B ∈ K_A+K_B) |

The authoritative decision is always the direct decision (d affine); the four-case classification is used for
testing and bucket prefiltering.

---

## 3. Algebraic construction of fused segments

### 3.1 Construction for case C2 (the main fusion case)

Given `A, B`: `S_A = a_A + K`, `S_B = a_B + K`, `K_A = K_B = K`, `d = a_A ⊕ a_B ∉ K`.
The support of the fused-segment function `g = f_A ⊕ f_B` is
`F = a_A + (K + ⟨d⟩)`, `dim F = dim K + 1`.

**Construction**: take a basis `ℓ_1, …, ℓ_{n−dim F}` of the annihilator of `K + ⟨d⟩` (i.e.
`n − dim K − 1` linearly independent linear forms whose common zero set is exactly `K + ⟨d⟩`). Then

```
g(x) = Π_j [ℓ_j(x) = ℓ_j(a_A)]
```

i.e.: **apply to x the linear coordinate change L (the first `n − dim F` coordinates are the `ℓ_j`, the rest
completed arbitrarily to a basis), one `(n − dim F)`-controlled MCX (control polarities = the first
`n − dim F` bits of `L(a_A)`), then L⁻¹**. The change L is realized by X/CNOT gates (§3.2), and X/CNOT have
T cost 0, so

**Fusion cost**: `T(g) = t(n − dim K − 1)`, gain
`= t(n − dim K_A) + t(n − dim K_B) − t(n − dim K − 1)` (`dim K_A = dim K_B = dim K`).

**Example (single differing coordinate)**: A and B differ only in the 0/1 choice at variable p (all other
positions identical); suppose A goes low at p (flip mask `U_0`), B goes high (flip mask `U_1`). Then
`K_A = K_B` (the direction subspace is determined only by "which coordinates are decided", not by the
flips), and `d = a_A ⊕ a_B = e_p ⊕ (U_0 ⊕ U_1)` (`U_0⊕U_1` acts only on variables deeper than p). The
**syntactic construction** of the fused segment agrees with the algebraic construction: position p becomes
an ft position with `lim0 = U_0`, `lim1 = U_1`, and the other positions are unchanged — the ft's two
conditional flips `U_0 ⊕ x_p·(U_0⊕U_1)` are exactly the X/CNOT network required by the basis change, and
the MCX control count = original control count − 1. Special case: when `U_0 = U_1`, `d = e_p`, the
annihilator basis can be taken as the standard coordinate functions, the basis change is the identity, and
there is **no extra flip at all**.

**Example (pivot wedge)**: A, B are complete paths (no ft), d arbitrary. The `ℓ_j` corresponding to the
differing coordinates are generally not single-variable projections, so a CNOT basis change is needed —
this is the algebraic content of the x_map/cx_map correction gates in `pivot_to` (in
`limbdd/LimBDDManager.py`): the CNOTs merge two coordinates with different polarities into a single control
coordinate.

### 3.2 Realizability of linear/affine maps by CNOT+X

**Theorem 3**: GL(n,2) is generated by transvections (`y_i ← y_i ⊕ y_j`, one CNOT gate) and coordinate
permutations; the whole affine group is generated by CNOT and X gates.

**Proof**: run Gaussian elimination on any invertible matrix: row swaps = coordinate permutations (on a
circuit this is only renaming, no gates); row adds/subtracts = transvections = CNOTs. Hence `L` decomposes
into O(n²) CNOTs; a constant translation = X gates. ∎

**Corollary**: every affine-cube indicator function can be realized in the branch-segment form
"CNOT/X network + one MCX + inverse network"; in the implementation the network is expanded as a sandwich
(forward network, MCX, inverse network), which guarantees input restoration (involutivity), and segments
preserve XOR additivity among themselves.

### 3.3 Construction for case C3 (nested codimension 1)

`S_2 ⊊ S_1`, `K_2` has codimension 1 in `K_1`, fused segment = `S_1 ∖ S_2` (**direction `K_2`,
`dim = k_2 = k_1 − 1`**). Take `ℓ` to be a linear form that is zero on `K_2` and nonzero on `K_1∖K_2`;
then the indicator function = `1_{S_1}(x) · [ℓ(x) ≠ ℓ(t)]` (`t ∈ S_2`), still a single MCX (control count
`n − k_2`) + X/CNOT basis change. Cost `t(n − k_2)`.

**Example** (n=2): A = the constant-1 segment (`S_A = V`, `k_1 = 2`, control count 0), B = the
single-control segment `[x_1 = 0]` (`S_B` a hyperplane, `k_2 = 1`); the fused segment = `[x_1 = 1]`,
control count `n − k_2 = 1`, CNOT count 0. T: `t(0) + t(1) → t(1)` (the gain in this example is 0; the gain
of C3 = the T of segment A, i.e. `t(n − k_1)`, because B and the fused segment have the same control count).

### 3.4 Construction for case C4 (crossing equal dimension)

`F = (t⊕u_1) + (W + ⟨u_1⊕u_2⟩)` (Theorem 2), `dim F = k_1`.
Take a basis of the annihilator of `W + ⟨u_1⊕u_2⟩` as the control coordinates; again a single MCX (control
count `n − k_1`) + X/CNOT basis change. Cost `t(n − k_1)`.

**Example** (n=2): A = `[x_1=0]`, B = `[x_2=0]` (two lines through the origin with different directions:
`K_1=span{e_2}`, `K_2=span{e_1}`, `W={0}`, `m=0=k−1`). `f_A⊕f_B = [x_1⊕x_2 = 1]`: 1 CNOT
(`y_1 ← x_1⊕x_2`) + 1-controlled MCX (`y_1=1`). T: 7+7 → 7.

### 3.5 Summary of the four cases (unified "annihilator" viewpoint)

The four cases unify as: **fused segment = affine-cube indicator function**; control count = `n − dim F`;
the X/CNOT network = Gaussian elimination that maps the annihilator of F's direction subspace to standard
coordinates. Therefore matching and construction can share the same canonical representation
"direction subspace + coset position".

### 3.6 RREF canonical construction (the construction used by the implementation)

**Theorem 3′ (affine cube → canonical SyntheticPlan)**: given the support `e + K_f` of d (base `e` and any
basis B of `K_f`, B not required to be canonical), there exists a **uniquely determined** synthetic segment

```
CNOT network → MCX(controls, ctrl_state) → inverse CNOT network
```

that realizes d exactly, with `|controls| = n − dim K_f` (attaining the optimum of §1.3.1). Construction
steps:

1. Put the row vectors of B into RREF (pivot = lowest bit, **full elimination**: each pivot column is
   cleared in all other rows), obtaining pivot columns P, free columns F and rows `r_p` (p ∈ P);
2. **constraint rows are indexed by the free columns** (the key direction of the membership equivalence in
   §1.2): `x ∈ e + K_f` ⟺ for every `f ∈ F`:

   ```
   x_f ⊕ Σ_{p ∈ P} (r_p)_f · x_p = e_f ⊕ Σ_{p ∈ P} (r_p)_f · e_p =: c_f
   ```

   (RREF makes each pivot row 0 in the other pivot columns, so the free columns can be parametrized
   independently; the pivot-column values are expressed linearly in terms of the row constants and the free
   columns.)
3. CNOT network: for each set bit `(r_p)_f = 1` add `CNOT(p → f)`. **Controls are all pivot variables and
   targets are all free variables**: the pivot variables are never modified, so the CNOTs do not interfere
   and any order is equivalent; after application `y_f = x_f ⊕ Σ_p (r_p)_f x_p`;
4. MCX: control set = free variables F, polarities = `c_f` (all constants are absorbed into ctrl_state; no X
   shift is needed);
5. Uncomputation: CNOTs are self-inverse; repeat in reverse order, restoring the input (sandwich structure,
   the involutivity requirement of §6.2).

**Correctness**: (2) is the equivalent rewriting of subspace membership; the order-independence of (3) comes
from "control set ∩ target set = ∅"; (4) polarity absorption is the definition of MCX ctrl_state; (5)
self-inverseness restores the input. **Dimension check**: `|controls| = |F| = n − dim K_f`.

**Canonicity and idempotence**: RREF (pivot = lowest bit + full elimination) is deterministic; (cnot_ops in
ascending (p,f) order, controls in ascending f order) is unique. By the RREF uniqueness of Theorem 1(ii),
the output depends only on (e, K) and **not on the chosen basis B** — the same (e, K) produces the same
synthetic segment — so fusion products are hashable and idempotent, and can be re-entered into the fusion
pool safely.

**Example (n=4, tt=42642, the same pair as §7)**: `e = 0100`, `B = (0001, 1110)`. RREF: `r_0 = 0001`
(pivot 0), `r_1 = 1110` (pivot 1) → P = {0,1}, F = {2,3}. Constraint rows:
`f=2: x2 ⊕ x1 = 1`; `f=3: x3 ⊕ x1 = 0`. Synthetic segment =
`CNOT(x1→x2), CNOT(x1→x3), MCX(x2=1, x3=0), inverse network`.
It is equivalent to the annihilator form `[x1⊕x2=1]·[x2⊕x3=1]` of §7 (two coordinate choices of the same
function; the RREF form is the canonical one).

### 3.7 Algebraic closed forms for each case

Let `reduce(·)` be the reduction of a vector family to an independent basis (lowest-bit pivots). The support
`(e, B)` of d can be expressed algebraically directly from the `(a, K)` of A and B, without enumerating 2^n:

- **C2** (K_A = K_B = K, d≠0): `e = a_A`, `B = reduce(basis of K ∪ {a_A ⊕ a_B})`;
- **C4** (equal dimensions, crossing): take `t ∈ S_A ∩ S_B`, `u_1 ∈ K_A∖W`, `u_2 ∈ K_B∖W`
  (W = K_A∩K_B), `e = t ⊕ u_1`, `B = reduce(basis of W ∪ {u_1 ⊕ u_2})`;
- **C3** (K_B ⊊ K_A codimension 1, S_B ⊆ S_A): take `w ∈ K_A∖K_B`, `e = a_B ⊕ w`, `B = basis of K_B`;
  the symmetric case is analogous;
- **C1**: d = 0 (cancellation, no synthetic segment).

The production decision path is the closed form: `backend/fusion/ClosedForm.py` (`canonical_affine`,
`fuse_closed`) is the decision-and-construction engine used by `fuse_pool` / `fuse_pool_global` in
`backend/fusion/BranchFusion.py` (switched to the closed form in stage "T-1", and cross-validated by
`tests/test_closed_form.py`). The old fn path (`fn()`, `is_affine_cube`, `affine_from_fn`, `shift_fn`) still
exists and is used for `LIMBDD_FUSION_DEBUG=1` cross-checks and by research scripts.

### 3.8 Gate-level identities for adjacent wrapper gates (algebraic basis of
`simplify_terms`)

At segment boundaries of a concatenated circuit, adjacent X/CNOT-type flip terms (LimTerm) appear and can
be merged by the following identities (all semantics-preserving):

1. `X(b1)·X(b2) = X(b1⊕b2)` — unconditional X groups merge linearly (X gates commute; bitwise XOR);
2. `[c=p→b1]·[c=p→b2] = [c=p→b1⊕b2]`, provided `c ∉ b1∪b2` — the earlier term does not flip the control
   bit, so the condition of the later term is unchanged by the merge;
3. `[c=0→b]·[c=1→b] = X(b)` — same control, opposite polarities, same target: the two conditions are
   mutually exclusive and exhaustive, so they merge into an unconditional X; this is the gate-level
   reproduction of lifting the common part `lim0∩lim1` in the compressed CompleteBranch representation
   (`x_global` in `backend/CompleteBranch.py`);
4. terms whose merged mask is 0 are deleted.

**Do not merge** multi-control-X ControlTerms: the XOR of two control conditions generally no longer keeps
the single-term form (merging would break the MCX structure).
This is implemented as `simplify_terms()` in `circuit/MCXCircuit.py`, applied to the whole concatenated
circuit — the adjacent wrapper gates at segment boundaries are the main source of gains of that pass.

---

## 4. Algebraic representation of CompleteBranch and the fusion algorithm

### 4.1 Branch → algebraic representation

Each branch segment A corresponds to one pair of data:

1. **Emission signature** (the Position sequence of §1.1; Lim already converted to the absolute variable bit
   space);
2. **Algebraic representation** `(a_A, K_A)`:
   - `a_A` = the canonical shift of `supp(f_A)` (the lexicographically smallest element, for
     disambiguation);
   - `K_A` = `{x ⊕ a_A : x ∈ supp(f_A)}`, stored as a **row-reduced echelon (RREF) basis** (unique
     canonical form).

Their relation: signature → function → algebraic representation (§1.3 Theorem 1); algebraic representation
→ fused-segment construction (§3).

### 4.2 Matching algorithm (structural decision)

For any two segments A, B (if necessary first shift B's support by `r_A ⊕ r_B`, §1.4):

1. **C1/C2**: `K_A = K_B` (RREF bases equal) → always fusible (if equal then cancellation, otherwise the C2
   construction).
2. **C3**: `K_B ⊊ K_A` (subspace criterion: every basis vector of `K_B` lies in `K_A`) and
   `dim K_B = dim K_A − 1` and `S_B ⊆ S_A` (`a_A ⊕ a_B ∈ K_A`); the symmetric case is analogous.
3. **C4**: `dim K_A = dim K_B`, `dim(K_A ∩ K_B) = k − 1`, and neither contains the other.
4. All other cases do not fuse.

Complexity: O(n²) bit operations on n×n bit matrices of canonical bases; together with bucketing by "RREF
hash of K", pool matching is near-linear (only cases 2 and 3 are checked across buckets).

### 4.3 Invariants and termination

- **Invariant**: the XOR sum `⊕ f` of the pool is unchanged under every `{A,B} → {A⊕B}` replacement
  (commutativity/associativity of GF(2) addition; the §3 construction guarantees the new segment's function
  equals `f_A ⊕ f_B` exactly).
- **Termination**: every fusion reduces the segment count by 1 (C1 reduces it by 2); the segment count is
  finite, so termination is inevitable.
- **Cost model**: greedy acceptance condition `gain = t̄(dim K_A) + t̄(dim K_B) − t̄(dim F) > 0`, where
  `t̄(d) := t(n − d)` (d ≥ 0) is the extended cost (X/CNOT are not counted in T; if CNOTs were counted, the
  basis-change network size would enter).
- **Degenerate-dimension convention (consistent with the §0 object-family definition)**:
  when `dim F = −1` (C1), the §0 convention `t̄(−1) := 0` (zero-block cost 0) applies, so the unified
  formula `gain = t̄(dim K_A) + t̄(dim K_B) − t̄(dim F)` also holds for C1, and the cancellation gain
  = `t_A + t_B`; for the zero function the convention `deg 0 := −1` is used, so the degree lower bound
  `t(deg f)` holds trivially for f ≡ 0. The implementation matches this: `is_affine_cube`
  returns `(True, −1)` for `fn == 0`, the cancel branch of `fuse_closed` returns `dim = −1`, and
  `config.t_cost(0) = 0`.

### 4.4 Pairwise-completeness conjecture (decomposability of the k-to-1 rule): partial results and open
problem

**Conjecture** (base case k=3): if the XOR of three affine cubes `S_1, S_2, S_3` is an affine cube
`D = d₀ + L`, then there necessarily exists a pair (i,j) satisfying C1–C4 (i.e. `S_i △ S_j` is a cube,
2-to-1 fusible). If the base case holds, the general k follows by induction (each step fuses one pair; the
sum remains χ_D, the pool size drops by 1, and finally everything merges into a single branch).

**Parts already proved rigorously**:

1. **Covering lemma (Fourier support)**: `K_i⊥ ⊆ L⊥ ∪ ⋃_{j≠i} K_j⊥` (for every i). Proof: take
   `u ∈ K_i⊥ ∖ (L⊥ ∪ ⋃_{j≠i} K_j⊥)`; in the main equation
   `Σ_i 2^{k_i}(−1)^{⟨u,a_i⟩}1_{K_i⊥} = 2^l(−1)^{⟨u,d₀⟩}1_{L⊥}` only one term on the left is nonzero at
   that point, while the right side is zero (u ∉ L⊥), contradiction.
2. **Reduction lemma**: if there exists i with `L ⊆ K_i`, then pairwise completeness holds for that pool.
   Proof: after quotienting by L, D becomes a single point δ (ANF degree n−l), each χ_j quotients to a
   (k_j−l)-dimensional cube (degree n−k_j); degree matching requires some j with `K_j = L` →
   `χ_j ⊕ χ_D` is the difference of two same-direction cosets = a coset of L (C2-type fusible); the
   remaining pool is still a new instance of "sum is a cube", so induction applies.
3. **2-adic valuation lemma**: inclusion-exclusion `2^l = Σ_i 2^{k_i} − 2ΣP + 4Q` (P, Q are intersection
   cardinalities, each 0 or a power of 2) written as `Σ ±2^{e_m} = 2^l`; then the parity of the number of
   lowest-valuation terms is determined by l (odd when μ=l, even otherwise).
4. **Proved subcases**:
   - Three cubes of equal dimension (k₁=k₂=k₃=k): for `l=k`, if no pair is fusible then every pairwise
     intersection ≤ 2^{k−2}, contradicting `ΣP − 2Q = 2^k` → a C4 pair must exist; for `l=k+1` and
     **D = the whole space**, the multiplicity analysis (`N₂+2N₃ = 2^k` and `Q = 2^{k−1} > P_max`) gives a
     contradiction.
   - Unique smallest dimension (k₁ < k₂ ≤ k₃): the valuation lemma forces `l = k₁`; in the `B∩C=∅` branch,
     `|B△C| ≤ 2^{k₁+1}` forces k₂=k₃=k₁+1 and A contains half of each of B and C, contradicting B∩C=∅; in
     the `B∩C≠∅` branch, it forces `B ⊆ C`, `k₃=k₂+1` → C3 fusible.

**Unclosed cases**: equal dimensions with `l=k+1` and **D a proper cube** (the restricted-coset version,
where each `S_i∩(d₀+L)` has direction degenerating to `K_i∩L`, complicates the multiplicity equation); the
double-smallest-dimension subcase (k₁=k₂<k₃, the intermediate branch of `ΣP ≡ 2Q (mod 2^k)` not yet
eliminated one by one); and the reduction for general k ≥ 4 (only Reduction Lemma 2 bridges some cases).

**Counterexample attempts** (all failed, weak evidence): three pairwise disjoint lines with different
directions (d is never a power of 2); two lines meeting in one point and a third separated line (the meeting
pair is exactly C4); three hyperplane cosets (the 2-adic equation does not close); line⊂plane⊂space nesting
(nested codimension 1 forces C3). Each type of configuration either breaks "d is a cube" or necessarily
exposes a C1/C3/C4 pair.

**Random search** (script `experiments/counterexample_search.py`, see Appendix F):

| n | number of triples | d-affine candidates | counterexamples |
|---|---|---|---|
| 5 | 40000 | 1180 (2.95%) | **0** |
| 6 | 40000 | 295 (0.74%) | **0** |
| 7 | 10000 + 100000 (supplementary) | 23 + 206 | **0** |

**Conclusion**: the conjecture has not been disproved, but has not been proved either; it remains an open
problem. **The engineering impact is zero** — the correctness, termination and all measured gains of the
fusion backend do not depend on the conjecture; it only decides whether the 2-to-1 greedy is theoretically
able to reach all k-to-1 opportunities.

---

## 5. Algebraic correspondence with SplitPattern

### 5.1 Two schedulings of the same identity

SplitPattern recognizes branch pairs on the graph and "splits" them; branch fusion recognizes branch pairs
in the pool and "fuses" them. They are two bookkeeping forms of the identity
`{A, B} → {A⊕B}`:

- **Splitting**: materializes `A⊕B` as an independent subgraph (piece), while A and B are rewritten in the
  original graph as residual segments not containing the shared part;
- **Fusion**: keeps `A⊕B` in the pool as one segment.

Since the XOR sum of the pool is the same under both bookkeeping forms, **the two are algebraically
completely equivalent**.

### 5.2 Geometric features → algebraic features

The recognition conditions of each SplitPattern mode translate as follows (`succ_l/succ_r` are the two
children of the same parent node):

| Pattern | Geometric condition | Algebraic meaning | Corresponding theorem case |
|---|---|---|---|
| type 1 | the two branches share the same child | the shared suffix C guarantees `K_A = K_PA ∩ K_C = K_PB ∩ K_C = K_B` (the matched prefixes make `K_PA = K_PB`) | C1/C2; new piece = fused segment (ft chain + C) |
| type 2 | one-child + branch sharing the child | one side's prefix degenerates (one fewer degree of freedom), K equality as in type 1 | C2 |
| type 3 | one-child + fallthrough sharing the child | the fallthrough side's conditional flip is absorbed into the ft, K equality as before | C2 |
| type 4 | the two branches share **two** children | the two branches differ only in polarity: the support of `d` is concentrated on the differing coordinates (± the Lim difference of the edges taken) | C1/C2 (very small basis-change network) |
| type 5 | double fallthrough and `lim_sum` Hamming weight ≤ 2 | `d = a_A⊕a_B` has small support (≤2 bits), the basis change needs at most ≤2 CNOTs | C2 |

General conclusion: **the geometric recognition of SplitPattern ⟺ the algebraic recognition
"direction subspace K equal" (plus the two boundary cases C3/C4)**; the function of the split-off new piece
= the XOR of the recognized branch pair (measured hits, Appendix B).

### 5.3 Canonicalization and fixed point

The in-graph rewriting of SplitPattern simultaneously performs Lim canonicalization
(`eliminate_redundant_lim`, shift), making subsequent K-equal pairs syntactically visible. The
algebraic-layer counterpart:

1. after every fusion, re-canonicalize `(a, K)` with §4.1 (the new segment may match other segments again);
2. iterate to a fixed point (each round strictly reduces the segment count, so it terminates);
3. equivalence acceptance with the old SplitPattern: after the new pass the pool XOR is unchanged and T does
   not increase (differential verification).

---

## 6. Algebraic explanation of the known defects

### 6.1 `pivot_to` is unsound for one-child positions

The wedge construction of `pivot_to` (in `limbdd/LimBDDManager.py`) implicitly assumes that the two branches
have two weight-1 edges at every position (complete branches). For branch pairs containing one-child
positions (w=0), it duplicates subgraphs along the imaginary edge, and the resulting function ≠ `f_A ⊕ f_B`.
Measured contract-failure rates: 4/19 at n=6, 5/78 at n=7 (Appendix C).

**Algebraically this is not a defect of the fusion theory**: for such pairs, as long as
`supp(f_A ⊕ f_B)` is an affine subspace (Theorem 2), the §3 construction still holds exactly. Therefore
general fusion must be implemented natively per §3/§4 and must **not** reuse the graph-level wedge of
`pivot_to`.

### 6.2 `get_pivot_to_circ` does not uncompute its correction gates (the algebraic root of the ancilla=1
bug)

The x_map/cx_map of pivot implement the X/CNOT network of the §3.2 basis change. A fused segment requires
the **sandwich structure** (forward network → MCX → inverse network); otherwise the input is not restored.
`get_pivot_to_circ` (in `backend/OracleSynthesizer.py`) applies the network only once and never uncomputes
it, and with `add_before=False` it appends the network after the nested circuit, causing: the input qubits
are polluted, and the final-state check of `compute_tt` reports `Anomaly Found!!!` (reproduction:
`input_state: 0 → output_state: 20`). **Lesson**: any X/CNOT basis change of a fused segment must be
emitted in the involutive sandwich form.

### 6.3 Free-variable folding of windowed subgraphs

The pieces produced by split_neg cover only part of the variables (windowed var_order). A piece function is
**free** in the uncovered variables: its direction subspace K contains the unit vectors of all uncovered
variables. `get_truth_table()` in `limbdd/LimBDD.py` writes only the coordinate bits of the covered
variables, which is equivalent to folding the free variables to 0 and returns the **restricted function**,
not the full-space function. Verification and the algebraic representation must use the full-space function:
`f_full(x) = f_restricted(x & covered_mask)`. The global verification of the branch probe (Appendix A) uses
exactly the full-space form.

---

## 7. Complete worked example (real data, n=4, tt=42642)

**Two branch segments** (from the branch probe; Lim already converted to the absolute variable space;
position tuple `(var, ft, bit, w0, w1, lim0, lim1)`):

```
A: (1, 0, 0, 1, 1, 0, 4)   → goes low, flip lim0=0, condition x1=0
   (3, 0, 0, 1, 0, 0, 0)   → goes low (one-child), condition x3=0
   (0, 1, -, 1, 1, 0, 0)   → ft, lim0=lim1=0 (identity, produces no condition)
   (2, 0, 1, 0, 1, 0, 0)   → goes high (one-child), condition x2=1
B: (1, 0, 1, 1, 1, 0, 4)   → goes high, flip lim1=4 (i.e. X(x2)), condition x1=1
   (3, 0, 1, 1, 1, 0, 0)   → goes high, condition x3=1
   (0, 1, -, 1, 1, 0, 0)   → ft, identity
   (2, 0, 1, 0, 1, 0, 0)   → goes high, condition (x2 ⊕ 1) == 1  ⟺ x2=0   ← note that B's flip at level 1 changed the effective state
```

**Step 1** (support): `S_A = {0100, 0101}`, `S_B = {1010, 1011}` (binary: x3 x2 x1 x0).
(B's support is x2=0 rather than x2=1 precisely because level 1's `lim1=4` flips x2 before x2 is decided —
§1.2 effective-state evolution.)

**Step 2** (direction subspace):
`a_A = 0100`, `K_A = {0000, 0001} = span{e_0}` (x0 free, from the ft level);
`a_B = 1010`, `K_B = span{e_0}`. `K_A = K_B` ✓, `d = a_A ⊕ a_B = 1110 ∉ K` → case **C2**.

**Step 3** (fused-segment support): `F = a_A + (K + ⟨d⟩) = {0100, 0101, 1010, 1011}`.

**Step 4** (annihilator): `K + ⟨d⟩ = span{e_0, e_1⊕e_2⊕e_3}` (dim 2). Take the annihilator basis (control
coordinates) `ℓ_1 = x_1 ⊕ x_2`, `ℓ_2 = x_2 ⊕ x_3` (both are 0 on `e_0` and on `e_1⊕e_2⊕e_3`). Their values
at `a_A = 0100` are `ℓ_1 = 1`, `ℓ_2 = 1`.

**Step 5** (fused segment):
`g(x) = [x_1⊕x_2 = 1] · [x_2⊕x_3 = 1]`: 2 CNOTs (`y_1 ← x_1⊕x_2`, `y_2 ← x_2⊕x_3`) + 2-controlled MCX
(`y_1=1, y_2=1`) + inverse CNOTs. Equivalent form `[x_1⊕x_3 = 0] · [x_1⊕x_2 = 1]`.
(The RREF canonical construction of §3.6 gives the latter form:
`CNOT(x1→x2), CNOT(x1→x3), MCX(x2=1, x3=0), inverse network` — the same function, unique canonical form.)

**Cost**: `T(A)+T(B) = 16 + 16 = 32` → `T(fused) = t(2) = 7` (the 2 CNOTs are free).

**Comparison**: the difference in this example consists of three parts — the polarity difference at level 1
(x1), the choice difference at level 2 (x3), and the flip difference `U_0⊕U_1 = 4` (x2) of the edges taken
at level 1; the three sum exactly to `d = 1110` (the `d = e_p ⊕ (U_0⊕U_1)` formula of §3.1 is the special
case of "only one differing position"); after the basis-change network merges the three difference
coordinates into two composite coordinates, the MCX control count is determined solely by
`n − dim K − 1 = 2`.

---

## 8. Verification checklist

1. §1.3 Theorem 1: is the affine induction on the state evolution `e_i` correct (the affinity of the ft
   level `A_var(z) = lim0 ⊕ z_var·(lim0⊕lim1)`)?
2. §1.4: is the cross-piece shift formula `f_B(y ⊕ (r_A⊕r_B))` and the argument "the fusion decision is
   independent of the root mask" correct?
3. §2.2 Theorem 2: is the classification complete? In particular check:
   - the closure argument of C2 (`K ∪ (d+K) = K+⟨d⟩`);
   - the cardinality analysis of the "intersecting case" in necessity
     (`|F| = 2^{k1}+2^{k2}−2^{m+1}`, the two branches `m=k2` and `m=k2−1`);
   - the non-affineness of the counterexample `{100,010,110,001}` and its affine-equivalence-class argument;
   - the C4 formula `(t⊕u_1) + (W + ⟨u_1⊕u_2⟩)`.
4. §3.1: check the control-count formula `n − dim K − 1` of the annihilator construction against each
   example (the d=e_p case, the n=4 worked example).
5. §3.2 Theorem 3: generation of GL(n,2) by transvections and the Gaussian-elimination decomposition.
6. §3.3/3.4: the control-count formulas of the boundary cases C3/C4 (C3 is `n − k_2`, C4 is `n − k_1`) and
   the n=2 examples.
7. §4.2: correspondence of the structural decision with the four cases of Theorem 2 (in particular C3's
   inclusion test `a_A⊕a_B ∈ K_A` and C4's mutual-non-containment condition).
8. §5.2 pattern-translation table: do all 5 SplitPattern modes fall into C1–C4, and are there missing
   algebraic cases.
9. §6: the algebraic explanations of the three defects (pivot's complete-branch assumption, the involutive
   sandwich requirement, free-variable folding).
10. Appendices A–C: consistency of the measured numbers with the theorems of this document.
11. §1.3.1 single-segment T optimality lemma: the argument "m controls ⇔ support
    dimension n−m".
12. §3.6 RREF canonical construction: is the direction "constraint rows indexed by free
    columns" consistent with the membership equivalence (pivot rows are 0 in the other pivot columns); the
    order-independence of the CNOT network "controls = pivots, targets = free"; polarity absorption of
    constants; canonical uniqueness (full elimination + ordering).
13. §3.7 closed forms: correspondence of the C2/C3/C4 `(e, B)` formulas with the input
    of §3.6; the cancellation semantics of C1.
14. §4.4 pairwise-completeness conjecture: derivation of the covering lemma, the
    reduction lemma, the 2-adic valuation lemma and the proved subcases; the list of unclosed cases; the
    search data of Appendix F.
14. §1.4 translation sensitivity: cross-piece fusion must re-decide on the shifted d'
    (the coset-position conditions of C3/C4 are translation-sensitive; K invariance is only part of the
    necessary condition).
15. §2.3 mapping table: correspondence of code `four_case_kind` 1–4 with C1–C4 (3/4 are
    mirrors), authoritative decision = direct decision.
16. §3.8 gate-level merge identities: the semantics-preserving conditions of the three
    LimTerm merge rules (especially "the flip bits do not include the control bit") and the reason not to
    merge ControlTerm.

---

## Appendices

### A. Measurement: branch segments are affine cubes

Method: run the complete-pipeline graph phase on a random n=5 case, extract all 38 branch-segment
signatures, compute each `f` with the signature-level simulator (32-bit truth table), and decide "affine
subspace" by "after translating the support, closed under XOR". Result: **38/38 are all affine subspaces**.

### B. Measurement: SplitPattern new piece = XOR of a branch pair

Method: step through `SplitLv2Pattern` / `SplitInterceptorPattern` (in `limbdd/patterns/`), and before/after
each split take the difference of the graph's branch-function pool, checking whether there exists a branch
pair `(A,B)` with `f_A ⊕ f_B` equal to the function of some new piece. Result: 2 hits (both were
SplitLv2Pattern, `dim(result direction subspace) = 2`).

### C. Measurement of the pivot contract (same-graph function-level fusible pairs)

Method: for fusible pairs in the same graph, run `pivot_to` + nested synthesis and verify
`f_pivot(map(x)) == f_A(x) ⊕ f_B(x)` (128-bit full space).

- n=6: 15 of 19 pairs succeeded, 4 failed (the failing pairs contain one-child positions); baseline over
  the successful pairs `Σ2T = 2112 → measured 626` (29.6%).
- n=7: 73 of 78 pairs succeeded, 5 failed; baseline over the successful pairs
  `Σ2T = 19668 → measured 4558` (23.2%).

### D. Reproduction of the ancilla=1 defect

n=7, `ancilla_num=1`, 40 random cases: 2 cases threw in `compute_tt`
`Anomaly Found!!!  input_state: 0, output_state: 20, anc_state: 0` (input qubits polluted by the correction
gates, not uncomputed).

### E. Branch-probe opportunity-statistics summary (reorder=1, 3462 cases)

| n | cases | avg subgraphs / branches | function-level fusible pairs | cross-subgraph share | greedy-match gain | share of baseline T |
|---|---|---|---|---|---|---|
| 4 | 2005 | 3.3 / 3.5 | 1219 | 100% | 2331 | 3.8% |
| 5 | 1002 | 5.3 / 6.5 | 1139 | 100% | 9972 | 9.0% |
| 6 | 303 | 8.1 / 11.7 | 546 | 96.5% | 12458 | 12.2% |
| 7 | 152 | 12.8 / 22.1 | 539 | 85.5% | 37477 | 19.8% |

("function-level fusible pair" = a branch pair with `supp(f_A ⊕ f_B)` an affine subspace, i.e. the direct
decision of Theorem 2.)

### F. Pairwise-completeness counterexample search (script `experiments/counterexample_search.py`)

Method: randomly generate triples of CompleteBranch (random SyntheticPlan: X shift + CNOT chain + random
control subset/polarities), and filter those with `d = f_1⊕f_2⊕f_3` an affine cube and all three pairs
pairwise non-fusible (`is_fusable=False`, i.e. no C1–C4 pair) = counterexample. Output files:
`experiments/phase0_results/cex_search_{n}_{seed}.txt`.

| n | seed | number of triples | d-affine candidates | candidate dim distribution | counterexamples |
|---|---|---|---|---|---|
| 5 | 1 | 40000 | 1180 (2.95%) | {−1:18, 0:8, 1:88, 2:248, 3:453, 4:350, 5:15} | **0** |
| 6 | 2 | 40000 | 295 (0.74%) | {−1:1, 0:1, 1:11, 2:43, 3:76, 4:96, 5:66, 6:1} | **0** |
| 7 | 3 | 10000 | 23 (0.23%) | {2:3, 3:2, 4:4, 5:6, 6:8} | **0** |
| 7 | 4 | 100000 (supplementary) | 206 (0.21%) | {0:1, 2:13, 3:50, 4:50, 5:62, 6:30} | **0** |

In total 190000 random triples and 1704 d-affine candidates, **zero counterexamples** — supporting the §4.4
conjecture as true, but the random search provides only empirical evidence; exhaustive enumeration (all
cube configurations for n ≤ 5) or a complete proof (the unclosed cases of §4.4) remains to be done.
