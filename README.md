# Affine-Block Fusion for T-Count Optimization of Oracle Synthesis

A deterministic framework for synthesizing Boolean oracles (one target qubit, `n` control
qubits) that minimizes the T-count as the primary objective and the CNOT-count as the
secondary objective. The core idea is to represent every branch of the decomposition as an
**affine block** — the indicator function of an affine subspace of `GF(2)^n` — and to fuse
pairs of blocks with a closed-form test using `O(n^2)` operations on `n`-bit masks, without enumerating the
`2^n` truth table.

## Key ideas

- **Affine block (a.k.a. `CompleteBranch`).** A single segment of the circuit has the form
  `(X/CNOT network) → one multi-controlled X → (inverse network)`; its flip function is
  exactly the indicator of an affine subspace (Theorem 1 in `docs/CompleteBranch_algebra.md`).
- **Closed-form fusion.** The symmetric difference of two affine subspaces is again an affine
  subspace **iff** one of four cases holds (C1–C4, Theorem 2). Each case has an algebraic
  closed form for the fused block, so both the test and the construction are purely symbolic.
- **Pool engine.** A greedy maximal-matching engine fuses blocks to a fixed point with a
  dimension-bucketed, incrementally maintained candidate set; a plain-cube `O(popcount)`
  fast path covers ANF monomials and most BDD walk segments. The XOR of the pool is an
  invariant, and every accepted fusion strictly decreases the segment count.
- **Committed macro M2.** After the fixed point, a distance-2 pair can be rewritten
  `{A,B} → {A⊕h, B⊕h}` and fused against a third block `C` whenever the total gain
  `imm + g_fuse > 0` — a strictly-decreasing move that recovers gains beyond pairwise greed.
- **Emission.** Fused segments are emitted with root-mask sandwiches, a canonical RREF plan,
  greedy segment reordering (B-4), and a final gate-level simplification.
  Reordering and simplification affect CX only, never T.

## Cost model

Defined in `config.py` (single source of truth, linear fits of Qiskit MCX decompositions):

```
t_cost(k)  = 0 if k <= 1 else 8*(k-1) - 1
cx_cost(k) = 0 if k <= 0 else (1 if k == 1 else 6*(k-1))
```

X and CNOT gates are T-free; each explicit CNOT costs one CX, while X gates are free.

## Synthesis pipelines

The evaluation entry supports twelve routes and explicitly uses zero ancillas. The five
classic pipelines below use `OracleSynthesizer.synthesize` with `ancilla_num=0`;
the default route is **FS**, with M2 enabled (`m2_post=True`).

| Path | Front-end | Notes |
|---|---|---|
| **F** | ANF decomposition (`c = 0`) | the unshifted ANF starting cover followed by fusion |
| **FS** | FPRM shift search | full `2^n` enumeration for `n ≤ 8`, deterministic two-start climb2 for `n ≥ 9` |
| **FP** | fusion-free proxy prescreen | top-`k` proxy candidates (always including `c = 0`) replace the full FS search; a quality/time trade-off |
| **FG** | deterministic `GL(n,2)` local search | steepest ascent over single-CNOT (transvection) moves; a slow, small-`n` enhancement option |
| **A** | LIMDD/BDD graph phase + branch extraction | demonstrates that the same engine consumes BDD branches as well as ANF monomials |

The evaluation entry also supports starting-cover comparisons: **MT** (minterms),
**PK** (a greedy pseudo-Kronecker-style cover), **PKS** (PK with shift search), and
**PKP** (PK with proxy prescreening). **F0**, **MT0**, and **PK0** emit the raw ANF,
minterm, and PK covers without fusion, M2, or segment reordering.

Route names follow the manuscript: the former C0/C/F/G routes are now F0/F/FS/FG,
respectively. The default remains ANF with shift search, now called **FS**. See
`data/README.md` for the migration and historical-data naming rules.

Shift search scores candidates after pairwise fusion and applies M2 only to the selected
pool. Its candidate inclusion guarantees therefore concern the cost before M2.
The recorded FP experiments use the existing heuristic score `Σt(n−|S|)−P`;
this is not the raw ANF cost `Σt(|S|)`. Also, the experimental `plan_min_cx`
basis changes collapse to the same RREF plan, so they provide no additional CX optimization.

## Repository layout

```
backend/            synthesizer + fusion engine (production code)
  OracleSynthesizer.py
  CompleteBranch.py
  fusion/           BranchFusion, ClosedForm, GLSearch, FprmProxy, MacroM2, CoverGenerators
limbdd/             LIMDD/BDD core, patterns, branch extraction, fused backend
circuit/            MCXCircuit, Control, segment reordering and simplification
utils/              Qubit, TruthTable, to_qiskit
tests/              test suites (also exposed through test.py)
experiments/        probe/measurement scripts used during development
docs/               architecture.md, CompleteBranch_algebra.md
data/               test cases, baselines, and per-case results (see data/README.md)
```

## Getting started

```bash
pip install -r requirements.txt

python test.py -f -b -a -c -g -e -p -q -r -s -d -x -u -w -m -y -z
python evaluate.py <n> <start> <stop> <path> [m2] [reorder]
#   path: A F FS FP FG MT PK PKS PKP F0 MT0 PK0
#   m2=0 adds _noM2 to filenames within the base route directory;
#   F0/MT0/PK0 ignore m2; reorder affects only A; all routes use zero ancillas.
python evaluate.py 8 0 1000 F0   # raw ANF baseline on the same 1000-case cover subset
python evaluate.py 8 0 1000 FS   # ANF + shift search + pairwise fusion + M2
python evaluate.py 8 0 1000 FS 0 # same route without M2; saves FS_noM2 files under FS/
```

Dependencies (numpy, qiskit, scipy) are declared in `requirements.txt`; the code is
standard-library-only beyond that.

The synthesizer is deterministic (no randomness in any path), asserts the pool XOR
invariant under `LIMBDD_FUSION_DEBUG=1`, and every emitted circuit is verified against its
truth table (`compute_tt`) during evaluation, outside the reported synthesis timing.

## Documentation

- `docs/architecture.md` — system design: module layout, synthesis routes, the fusion
  engine, the cost model, and the data layout.
- `docs/CompleteBranch_algebra.md` — the mathematics: Theorem 1 (branch = affine block),
  the C1–C4 classification and proofs of Theorem 2, the canonical RREF construction, and
  the single-block T-optimality lemma.

## Results and data

Precomputed outputs over `n` up to 11, together with Exorcism and SSHR baselines and the
test-case sets, live under `data/`. Per-case T/CX costs and run summaries are stored in
`data/our_cost/{path}/`; default and `_noM2` files share the same base route directory.
Historical degree-bound files remain under `data/our_cost/deg_bound`; new runs include
the bound only in their summary. The format and coverage are documented in `data/README.md`.
