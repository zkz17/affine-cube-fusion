# System Architecture (current state)

> This document is the design reference for the **current state** of the system. It records the
> module layout, the synthesis pipelines, the fusion engine, the cost model and the data layout;
> it does not restate any proofs.
>
> The mathematics lives in `docs/CompleteBranch_algebra.md` (affine-block algebra, Theorem 1,
> the C1–C4 classification of Theorem 2, and the canonical construction). In particular, the code
> object `CompleteBranch` and the paper term **Affine Block** denote the same concept.
>
> All paths, class names and switches below follow the current repository: the class
> `LimBDDOracleSynthesizer` was renamed `OracleSynthesizer`, and `limbdd/fusion/` together with
> `limbdd/CompleteBranch.py` moved under `backend/`.

---

## 1. Module layout

```
backend/                     synthesizer and fusion engine
  OracleSynthesizer.py       classic synthesize entry and process_cover_route entry
  CompleteBranch.py          CompleteBranch / SyntheticPlan / plan_from_affine / plan_min_cx /
                             t_cost / rref / reduce_basis / in_span / is_affine_cube / affine_from_fn
                             (CompleteBranch == the paper's "Affine Block")
  fusion/
    BranchFusion.py          pool-fusion entry: fuse_pool_global / fuse_pool / pool_t_cost /
                             anf_pool / anf_degree / shift_fn / four_case_kind
    ClosedForm.py            closed-form decision and construction: canonical_affine / fuse_closed / closed_gain
    GLSearch.py              option G: gl_search / fprm_best / apply_linear / climb_best
    FprmProxy.py             option FP: proxy_scores / fprm_proxy_best
    MacroM2.py               macro M2 post-processing: m2_best_move / m2_interleaved / h_candidates
    CoverGenerators.py       minterm / greedy PK covers, PK shift search and proxy prescreen
limbdd/                      BDD core and extraction
  Node.py / Lim.py / LimBDD.py / LimBDDManager.py
  patterns/                  SplitPattern / SplitLv2Pattern / SplitInterceptorPattern (split patterns)
  Extraction.py              abs_lim / extract_complete_branches (subgraph -> affine-block pool)
  FusedBackend.py            limbdd2circ_fused (single-subgraph fused emission, legacy entry)
circuit/
  MCXCircuit.py              MCXCircuit / STGate / order_segments_greedy / simplify_terms /
                             simplify_term_list / boundary_cx_savings
  Control.py                 Control / CtrlSequence (ControlTerm / LimTerm)
utils/
  Qubit.py                   Qubit / QubitManager
  TruthTable.py              TruthTable
  to_qiskit.py               MCXCircuit -> Qiskit circuit (used by evaluate.py for cost counting)
config.py                    single source of truth for t_cost / cx_cost (see §4)
evaluate.py / test.py        evaluation entry / test entry
```

Key points:

- The production path is the **fused backend** (`OracleSynthesizer.fused_backend=True` with
  `ancilla_num=0`). Evaluation explicitly fixes zero ancillas and exposes no ancilla argument.
  The class retains a legacy `ancilla_num=1` default for older callers; direct fused-backend
  callers must pass zero explicitly. Ancilla-assisted synthesis is outside the current experiments.
- `limbdd/patterns/` only serves the split patterns of the BDD graph phase; the fusion engine does
  not depend on BDD and merely consumes pool entries `(CompleteBranch, root_mask, key)`.

---

## 2. Synthesis pipelines and starting-cover routes

### 2.1 Overall flow

```
TruthTable tt
  -> OracleSynthesizer.synthesize(tt)
       fused_backend and ancilla_num == 0:
         allocate_input_qubit(tt.num_var)   # allocate all input qubits explicitly (see section 6.4)
         process_fused_global(tt)
  -> path selection (with gl_search the G option runs alone and `paths` is ignored)
  -> fusion to a fixed point (optional m2_post macro-M2 interleaved schedule)
  -> emission pipeline (section 2.3)
```

`OracleSynthesizer` switches:

- `gl_search=False` (default off): option G, a standalone item;
- `fprm_proxy=False` (default off): option FP, which **replaces** F (a choice relation, not a min);
- `m2_post=True` (default on): the winning pool gets macro-M2 post-processing;
- `paths=('F',)` (default): **single-path discipline** — exactly one of `'A'/'C'/'F'`; multi-path
  combinations (an explicit min) are disabled at this stage. FP and G are selected by switches and
  are never combined with other paths through an implicit min.

### 2.2 The five classic paths

| Label | Entry | Initial pool / search | Emission sandwich |
|---|---|---|---|
| **A** (BDD) | `paths=('A',)` | `construct` (variable order + `try_neg`/`split_neg`/`reorder`/sifting) -> `_collect_pieces_fused` (split patterns; absorb removed) -> `extract_complete_branches` | per-piece root-mask `X(r)` sandwich |
| **C** (ANF) | `paths=('C',)` | `anf_pool`: fast Möbius transform, every monomial `x^S` becomes one affine block (`c=0` semantics) | no shift sandwich |
| **F** (FPRM, production default) | `paths=('F',)` | `_fprm_entries`: select a shift `c` by the pairwise-fused pool cost; `n≤8` full `2^n` enumeration, `n≥9` two-start `climb2` (starts `0` and `2^n−1`) | `X(c)` sandwich (T-free) |
| **FP** (proxy) | `paths=('F',)` with `fprm_proxy=True` | `fprm_proxy_best`: enumerate all `2^n` shifts with Möbius only, proxy score `S1 = Σt(n−|S|) − P`, take top-8 by ascending `(S1,c)` and always include `c=0`, then full fusion on ≤9 candidates | `X(c)` sandwich |
| **G** (GL) | `gl_search=True` (ignores `paths`) | `gl_search`: first compute the F-selected `cF`; start set = `{identity on f, identity on f∘τ_cF} ∪ {all single CNOTs on f∘τ_cF}`; steepest ascent in the CNOT neighborhood, `max_steps=3`; candidates include `cF` itself (zero wrapper) | `X(c) · CNOT(A⁻¹) · [segments] · CNOT(A) · X(c)` (CNOTs counted in CX) |

Path-selection semantics: for non-G paths the paths listed in `paths` are computed in the given
order, the one with the lowest `pool_t_cost` wins, ties keep the earlier one. All paths feed the
same `fuse_pool_global` engine.

For G, the zero-regression relation `G ≤ F` is **structural before macro-M2 post-processing**
(candidates always include `cF`); with `m2_post` on (the default) it becomes empirical — no
violation was observed in the n≤7 measurements, and it is not a design constraint.
The same qualification applies to shift-search candidate inclusion relative to C: M2 runs
only on the winning pool, so pre-M2 dominance does not prove post-M2 dominance.

The evaluator also calls `process_cover_route` after clearing the synthesizer and allocating
all input qubits. **MT** and **PK** feed minterm and greedy pseudo-Kronecker-style covers into
the same fusion engine. **PKS** searches shifted PK covers (all shifts for n≤8, two-start
climb for n≥9); **PKP** prescreens shifts and fuses the top-8 plus c=0. These four routes
honor `m2_post`. **C0 / MT0 / PK0** emit raw ANF / minterm / PK covers in generator order,
with no fusion, M2, or segment reordering. This gives three corresponding raw-cover baselines.

### 2.3 Emission pipeline (tail of `process_fused_global`)

1. **Shift sandwich**: if the winning path carries a shift `c`, apply `X(c)` globally (no-op when
   `shift_c=0`);
2. **GL linear sandwich (G only)**: apply `CNOT(A⁻¹)` (`cnots` in forward order);
3. **Root-sandwich grouping**: group `kept` by root mask `r`, and for each `r` (ascending) apply
   `X(r)`, merge the group's segments in the order given by `order_segments_greedy`, then close `X(r)`;
4. **Standalone segments**: fusion products (`key=None`, self-contained) are likewise ordered by
   `order_segments_greedy` and merged;
5. **Close the GL linear sandwich**: `CNOT(A)` (`cnots` in reverse order);
6. **Close the shift sandwich** `X(c)`;
7. `simplify_terms()`: merge adjacent `LimTerm`s (unconditional X groups XOR together, conditional
   flips with the same control and polarity XOR together, conditional flips with the same control
   and opposite polarity but identical targets become one unconditional X).

Each segment is emitted by `CompleteBranch.to_circuit` as an **involutive sandwich**:
`X(x_trans) → CNOT network → MCX → CNOT network (inverse) → X(x_trans)`. Synthetic segments are
emitted with `plan_min_cx`, while the stored `SyntheticPlan` keeps the RREF canonical form
(needed for idempotence/hashing in the fusion pool). The current basis-candidate pass reduces
all row-equivalent candidates to the same RREF under the same column order. It therefore
provides no additional CX reduction and does not establish CX optimality. Segment reordering
and gate simplification remain the effective CX optimizations.

**B-4 segment reordering** (`order_segments_greedy`): nearest-neighbor greedy using the boundary
cancellation `boundary_cx_savings` as edge weight (all starts tried), candidates include the
original order, and every candidate is evaluated through the real `merge + simplify` pipeline;
the CX-minimal one is taken. It **affects CX only and never T** (the segment set is unchanged).

---

## 3. Fusion engine highlights

- **Representation**: each segment is the indicator function of an affine cube `base + span(basis)`;
  `CompleteBranch.affine()` derives it symbolically using O(n²) operations on n-bit masks
  (no 2^n enumeration). A cross-slice
  shift is exactly `base ^= root_mask` (`canonical_affine`).
- **Closed-form decision C1–C4** (`ClosedForm.fuse_closed`, O(n²) operations on n-bit masks; this is the
  production decision path):
  - `K_A == K_B`: `d ∈ K` -> C1 cancellation (always accepted); otherwise C2 fusion;
  - `K_A ⊂ K_B` or `K_B ⊂ K_A` with codimension 1 and the same large coset -> C3 (set difference,
    codimension-1 nesting);
  - same dimension, intersection of codimension 1, cosets meet -> C4 (one elimination pass via
    `_build_joint`/`_solve_split` finds the intersection point).
- **`fuse_pool_global`** (the production engine): input `[(CompleteBranch, root_mask, key)]`;
  - **bucketing**: bucket by `dim(K)`, enumerate only the same and adjacent buckets (Theorem 2
    implies fusible pairs have dim difference ≤ 1);
  - **incremental matching**: candidate keys `(gain, i, j, cancel)` are globally unique; surviving
    candidates only get their indices remapped, and new candidates involve fusion products only;
  - **plain-cube fast path**: when the direction subspace is a coordinate subspace the test
    degenerates to an O(popcount) mask check;
  - acceptance: cancellation is always accepted; non-cancelling fusions require a **strictly
    positive gain** `gain = t(A)+t(B)−t(F) > 0`;
  - returns `(kept, standalone)`: `kept` entries keep their slice identity (emitted inside the
    corresponding `X(r)` sandwich), `standalone` entries are self-contained fusion products.
- **FPRM budget switch** (`_fprm_entries` / `fprm_best`):
  - `n≤8`: full enumeration of all `2^n` shifts (n=8 means 256 complete fusions);
  - `n≥9`: `climb2`, steepest ascent from the two starts `{0, 2^n−1}` (Hamming neighborhood, strict
    improvement, caching), which evaluates only a few dozen complete fusions.
    **This explains why the F path is faster at n=9 than at n=8** (1368 ms vs 1984 ms per case).
- **gl_search (G)**: deterministic GL(n,2) local search, a small-n enhancement option; validated for
  n≤7, about **1.05 s/case at n=6**. Most of that time is re-running `fuse_pool_global` for every
  coordinate change (~94%).
- **fprm_proxy (option FP)**: proxy prescreen replacing F. The saved experiments use the legacy
  heuristic `S0 = Σt(n−|S|)`, with `P` = number of one-variable-apart monomial pairs and
  `S1 = S0 − P`; this is not the raw ANF cost, which is `Σt(|S|)`. Top-8 candidates get full
  fusion. Always including `c=0` implies the result is ≤ the C path before M2.
  Measured over n=3..10: it retains **85–98%** of F's Exorcism T reduction; it is **8–13× faster at
  n=7/8** but only about **2× faster at n=9/10** (the proxy pass itself costs O(n·4^n) because every
  one of the `2^n` shifts needs a Möbius transform, whereas F's `climb2` evaluates only tens of full
  fusions).
- **MacroM2 (`m2_post`, on by default)**:
  - schedule: `fuse_pool_global` to a fixed point -> enumerate the best macro M2 -> re-fuse, at most
    `max_rounds=30` iterations;
  - move: for a segment pair `(A,B)` and a candidate `h` (taken from the single-row divergent family
    of the pair's constraint rows: all shared rows plus one differing equation, with each polarity
    tried separately for an opposite row), first apply `{A,B} -> {A⊕h, B⊕h}`, then fuse
    `A⊕h` with a third segment `C`; the result replaces `{A,B,C}` by `{B⊕h, (A⊕h)⊕C}`;
  - acceptance uses the **true total gain** `imm + g_fuse > 0`, with
    `imm = t(A)+t(B)−t(A⊕h)−t(B⊕h)` and
    `g_fuse = t(A⊕h)+t(C)−t((A⊕h)⊕C)`; the pair rewrite itself may be T-neutral or slightly
    worsening — only the total is required to be positive. The implementation requires
    nonzero rewrite and final fusion products; it implements this three-block subset of the
    broader macro definition, not a separate immediate two-block rewrite pass;
  - C prescreen: a necessary condition for `gain(A′,C) > 0` is a `dim(K)` difference ≤ 1, so `C` is
    scanned per dim bucket;
  - the XOR-invariant assertion runs only under `LIMBDD_FUSION_DEBUG=1`; every step strictly
    decreases, hence per-case T never regresses.

---

## 4. Cost model and data layout

### 4.1 Cost functions (`config.py`, single source of truth)

```python
t_cost  = lambda x: 0 if x <= 1 else 8 * (x - 1) - 1
cx_cost = lambda x: 0 if x <= 0 else (1 if x == 1 else 6 * (x - 1))
```

Convention: an MCX/CCX with `k` controls costs `t_cost(k)` / `cx_cost(k)`; an explicit CX costs one
CX; X gates are free.

### 4.2 Data directories

```
data/test_cases/test_cases_{n}                    # one truth-table integer per line
data/exorcism_cost/exorcism_{t,cx}_cost_{n}       # Exorcism baseline (aligned by line number)
data/sshr_cost/sshr_{t,cx}_cost_{n}               # SSHR baseline (n>=7 empty -> skipped automatically)
data/our_cost/
  {path}/    our_t_cost_{tag}_{n}_{start}_{stop}.txt
             our_cx_cost_{tag}_{n}_{start}_{stop}.txt
             our_results_{tag}_{n}_{start}_{stop}.txt
  deg_bound/ # historical lower-bound files; new runs report the bound in the summary only
  no_M2_vs_default.txt
```

- `path` is one of `A C F FP G MT PK PKS PKP C0 MT0 PK0`. `tag` is the route name, with
  `_noM2` appended when M2 is disabled; both variants share the base route directory.
  Raw C0/MT0/PK0 always use their plain tag and ignore the M2 switch.
- Classic campaign: n=3 all 256 cases, n=4 all 65536, n=5–7 10000 each, n=8–11 1000 each;
  G stops at n=7. Cover routes use the first 1000 cases for each n=5..11. Saved no-M2
  comparisons cover A/C/F/FP at n=4..11, 1000 cases each. See `data/README.md` for exact coverage.

---

## 5. Testing discipline and acceptance conventions

1. **Routine changes are tested for n≤7 only**; n≥8 runs are reserved for the final aggregation round
   (time budget).
2. **Single-path discipline**: each evaluation measures one of the twelve routes, with no
   min taken across routes; FP and G are never implicitly combined with the default F.
3. **Determinism**: full enumeration / deterministic start sets (`climb2`, the GL start set, the
   `(S1,c)` proxy ordering) / deterministic tie-breaking; no random sampling (the old rand64 sampler
   was replaced by `climb2`).
4. **XOR invariant**: under `LIMBDD_FUSION_DEBUG=1` the fusion engine and macro M2 assert round by
   round that the pool XOR equals the target function (guaranteed by `h⊕h=0` for M2 rewrites).
5. **Time measurement**: `time.perf_counter_ns()` around synthesis (including reset/allocation
   for cover routes), accumulated per case and averaged as `avg synthesize wall`. Verification,
   Qiskit conversion, and file I/O are excluded. Historical timings come from separate runs;
   controlled timing comparisons require the same environment and execution conditions.
6. **Emission verification**: the evaluator rejects a circuit unless `circ.verify(tt)` succeeds.
   `test.py -y -z` covers raw ANF semantics/costs, all evaluation routes with both M2 settings,
   directory conventions, and summary consumers on multiple nonzero input intervals.

---

## 6. Final decisions

### 6.1 Graph phase and path positioning

- **absorb removed**: `LimBDDManager.collect_branches` no longer calls `try_absorb_branch` (the
  function body is kept for research). A dedicated measurement
  (`experiments/bdd_split_absorb_probe.py`) showed its isolated effect is exactly zero (±8 noise
  level): it only performs C1 cancellations, which `fuse_pool_global` covers unconditionally.
- **split patterns stay active**: they are the core commitment mechanism of the BDD path (removing
  them degrades n≤6 by 10–13%).
- **The BDD path is a cheap supplement for n≤6**: A's small-n advantage comes from the
  Shannon/LIM-structured coarse pool plus split commitment; from n≥7 on, A degrades noticeably
  relative to F (A's Exorcism CX reduction even turns negative, −3.1% → −9.2%), so it is not a
  workhorse.
- **Production backend = F (with m2_post on)**; **FP** serves time-sensitive runs (slightly lower
  quality, faster); **G** is a small-n quality mode (best quality up to n=6, but 1051.6 ms/case at
  n=6).
- **A / C degrade for n≥7**: compared with F their T reduction narrows markedly; A's Exorcism CX
  reduction turns negative (−3.1% → −9.2%) and C's is only +4~9%. Both are kept as path controls and
  for coarse-pool research, not as production defaults.

### 6.2 Historical classic-route summary (n=3..10)

The following numbers describe the earlier classic-route campaign. They do not include
the starting-cover comparison. `experiments/data_summary.py` reads current per-route files;
it prints the chosen sample interval because totals from different intervals are not comparable.

**F path vs Exorcism T (n=3..10)**: `+30.9% / +38.4% / +41.4% / +38.4% / +32.2% / +27.3% / +21.7% /
+20.8%`, **with zero higher cases**.

**F path vs Exorcism CX (n=3..10)**: `+18.9% / +24.6% / +28.8% / +26.6% / +22.0% / +19.0% / +14.9% /
+15.5%` (at most 10 higher cases for n≥7).

**F path vs SSHR (n=3..6)**: T `+20~29%`, CX `+18~27%`.

**Wall (ms/case, F)**:

| n | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| F | 0.5 | 2.2 | 11.4 | 61.0 | 340.9 | 1984.4 | 1368.2 | 5411.5 |

(FP: 0.5 / 1.3 / 3.7 / 12.0 / 41.6 / 155.2 / 639.5 / 2776.4; A/C lie between 0.3 and 1158. The fact
that F at n=9 is faster than at n=8 is explained by the FPRM budget switch in §3.)

### 6.3 Quality/time switch conclusions

- **GL**: best quality for n≤6 (G at n=6: Exorcism T +50.0%), but a heavy wall time (1051.6 ms/case
  at n=6), so it remains a small-n quality demonstration/enhancement option.
- **FP**: retains 85–98% of F's T reduction and is 8–13× faster at n=7/8 (about 2× at n=9/10); it is
  the intended substitute for F in time-sensitive runs.
- **M2**: on by default; accepted moves strictly decrease T. CX is not part of the macro
  acceptance condition and can increase (the saved A comparisons at n=10/11 include such cases).
  Use `data/our_cost/no_M2_vs_default.txt` for the saved A/C/F/FP comparisons; no saved G
  no-M2 campaign is included in that report.

### 6.4 Input-qubit allocation fix (fused backend)

The fused backend skips `construct` (no BDD is built) and previously allocated input qubits lazily:
if a variable appeared in no emitted operator (for example f independent of x₀; regression case n=4,
tt=64560), `compute_tt` enumerated over a shrunken variable space and the emitted truth table was
misaligned. `synthesize` now **explicitly allocates all input qubits** at the fused-path entry
(`allocate_input_qubit(tt.num_var)`). Data files produced by the fused backend before this fix may
contain such silently wrong entries and must be re-verified before any final aggregation (the current
`data/our_cost/` files are products of the re-run after the fix).

---

## 7. Relation to the algebra document

`docs/CompleteBranch_algebra.md` is the mathematical foundation of the fusion decision (Theorem 1:
affine block ⟺ affine-cube function; Theorem 2: the C1–C4 classification of symmetric differences;
§3.6–3.7: the canonical construction and the algebraic closed forms; §1.4: shift invariance). It is
the reference for `backend/CompleteBranch.py` and `backend/fusion/ClosedForm.py`. Macro M2 has no
separate theory document; its definition, acceptance condition and invariants are the ones described
in §3 above and implemented in `backend/fusion/MacroM2.py`.
