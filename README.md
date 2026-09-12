# Affine-Block Fusion for T-Count Optimization of Oracle Synthesis

A deterministic framework for synthesizing Boolean oracles (one target qubit, `n` control
qubits) that minimizes the T-count as the primary objective and the CNOT-count as the
secondary objective. The core idea is to represent every branch of the decomposition as an
**affine block** — the indicator function of an affine subspace of `GF(2)^n` — and to fuse
pairs of blocks with a closed-form, `O(n^2)`-bit-operation test that never enumerates the
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
- **Emission.** Fused segments are emitted with root-mask sandwiches, a CX-optimal basis
  (`plan_min_cx`), greedy segment reordering (B-4), and a final gate-level simplification.
  Reordering and simplification affect CX only, never T.

## Cost model

Defined in `config.py` (single source of truth, linear fits of Qiskit MCX decompositions):

```
t_cost(k)  = 0 if k <= 1 else 8*(k-1) - 1
cx_cost(k) = 0 if k <= 0 else (1 if k == 1 else 6*(k-1))
```

X and CNOT gates are T-free and are only charged in the CX objective.

## Synthesis pipelines

One entry point, `backend/OracleSynthesizer`, exposes five pipelines; the production default
is **F** with the M2 post-processing enabled (`m2_post=True`).

| Path | Front-end | Notes |
|---|---|---|
| **C** | ANF decomposition (`c = 0`) | the base pipeline and the `c = 0` anchor of the dominance chain |
| **F** | FPRM-optimal shift search | full `2^n` enumeration for `n ≤ 8`, deterministic two-start climb2 for `n ≥ 9` |
| **FP** | fusion-free proxy prescreen | top-`k` proxy candidates (always including `c = 0`) replace the full F search; a quality/time trade-off |
| **G** | deterministic `GL(n,2)` local search | steepest ascent over single-CNOT (transvection) moves; a slow, small-`n` enhancement option |
| **A** | LIMDD/BDD graph phase + branch extraction | demonstrates that the same engine consumes BDD branches as well as ANF monomials |

## Repository layout

```
backend/            synthesizer + fusion engine (production code)
  OracleSynthesizer.py
  CompleteBranch.py
  fusion/           BranchFusion, ClosedForm, GLSearch, FprmProxy, MacroM2
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
./venv/bin/python test.py -f -b -a -c -g -e -p -q -r -s -d -x -u -w -m   # full test suite
./venv/bin/python evaluate.py <n> 1 0 <start> <stop> <gl> <proxy> <paths> [m2]
#   <paths> : A | C | F (single path); gl=1 selects G; proxy=1 with F selects FP
```

The synthesizer is deterministic (no randomness in any path), asserts the pool XOR
invariant under `LIMBDD_FUSION_DEBUG=1`, and every emitted circuit is verified against its
truth table (`compute_tt`).

## Documentation

- `docs/architecture.md` — system design: module layout, the five pipelines, the fusion
  engine, the cost model, and the data layout.
- `docs/CompleteBranch_algebra.md` — the mathematics: Theorem 1 (branch = affine block),
  the C1–C4 classification and proofs of Theorem 2, the canonical RREF construction, and
  the single-block T-optimality lemma.

## Results and data

Precomputed outputs for the five pipelines over `n` up to 11, together with Exorcism and
SSHR baselines and the test-case sets, live under `data/` (per-case T and CX costs under
`data/our_cost/t_cost` and `data/our_cost/cx_cost`, degree bounds under
`data/our_cost/deg_bound`, and per-run summaries `data/our_cost/our_results_*`). The
format and file-naming conventions are documented in `data/README.md`.
