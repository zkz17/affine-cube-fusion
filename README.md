# Quantum Oracle Synthesis via Affine Cube Fusion

Affine Cube Fusion (ACF) is a deterministic framework for synthesizing Boolean oracles (one target qubit, `n` control
qubits) that minimizes the T-count as the primary objective and the CNOT-count as the
secondary objective. A starting cover represents the Boolean function as the XOR of
**affine cubes**, the indicator functions of affine subspaces of `GF(2)^n`. ACF fuses
pairs of cubes with a closed-form test using `O(n^2)` operations on `n`-bit masks, without enumerating the
`2^n` truth table.
The optimized cubes are then emitted as affine circuit blocks. The `limbdd/` module
retains the project's historical DD front end.

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

The evaluation entry supports thirteen routes and explicitly uses zero ancillas. The five
classic pipelines below use `OracleSynthesizer.synthesize`. Both the Python constructor
and the evaluator default to zero-ancilla **FS**, with M2 enabled (`m2_post=True`).
Explicit nonzero `ancilla_num` values retain the historical ancilla-assisted backend,
which is outside the paper's experiments.

**Known limitation:** The historical backend with `ancilla_num > 0` has known
correctness issues and may fail to restore input qubits. Use `ancilla_num=0` for
supported synthesis; fixing the nonzero-ancilla backend is future work. See
[the algebra notes](docs/CompleteBranch_algebra.md), Section 6.2 and Appendix D.

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
**A0** emits A's preprocessed DD cover with the same three optimizations disabled.
It retains all of A's DD operations, including variable reordering and splitting,
and preserves the extracted branches' root-mask transformations. All four baselines
retain the shared basic circuit simplification. The A0-to-A CX difference therefore
includes both fusion and segment reordering, as for the other raw-to-fused pairs.

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

For application-derived experiments on DES, AES, Ascon and SPEEDY S-box output functions, use
the separate `evaluate_structured.py` entry. Dataset sources,
bit ordering, regeneration, external backend setup and complete-dataset reporting
are described in [the structured benchmark guide](data/test_cases/structured/README.md).
The existing `evaluate.py` continues to handle the random/exhaustive datasets.

Use Python 3.12 (the paper runs used 3.12.3). From the repository root, create and
activate an environment on Linux or WSL:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `py -3.12 -m venv venv`, then
`.\venv\Scripts\Activate.ps1` and the same `python -m pip install` command.

On Windows, printing Qiskit's Unicode circuit diagrams (for example, in `main.py`)
may raise `UnicodeEncodeError` when output uses a legacy encoding, especially when
redirected. Run `python -X utf8 main.py` to enable Python's UTF-8 mode.

`requirements.txt` includes NumPy, Qiskit, SciPy, Graphviz's Python package, and
sortedcontainers. DD image rendering additionally needs the Graphviz system executable
`dot` on PATH (for example, `sudo apt install graphviz` on Ubuntu); synthesis and tests
do not invoke `dot` unless drawing is requested.

`requirements-paper.txt` records the tested direct dependency versions, including
the optional plotting packages. To use those versions, install it instead:
`python -m pip install -r requirements-paper.txt`. This is a direct-dependency
snapshot, not a lock of every transitive package.

```bash
python test.py -f -b -a -c -g -e -p -q -r -s -d -x -u -w -m -y -z
python evaluate.py --help
# Small run on eight saved functions, writing outside the paper data:
python evaluate.py 3 0 8 FS --output-dir runs/reproduction

# Usage: python evaluate.py <n> <start> <stop> <path> [m2] [reorder] [--output-dir DIR] [--overwrite]
#   path: A A0 F FS FP FG MT PK PKS PKP F0 MT0 PK0
#   m2=0 adds _noM2 to filenames within the base route directory;
#   A0/F0/MT0/PK0 ignore m2; reorder affects DD variables in A/A0 only;
#   all routes use zero ancillas.
# Complete batches (substantially slower than the small run):
python evaluate.py 8 0 1000 A0 --output-dir runs/reproduction
python evaluate.py 8 0 1000 F0 --output-dir runs/reproduction
python evaluate.py 8 0 1000 FS --output-dir runs/reproduction
python evaluate.py 8 0 1000 FS 0 --output-dir runs/reproduction
```

The output root defaults to `data/our_cost`; `--output-dir DIR` writes to `DIR/<route>/`
with the existing filenames, including `_noM2` when appropriate. Existing batch files
are protected by default. Use a fresh output root for reruns; `--overwrite` explicitly
allows replacement. Results are written only after the entire batch passes circuit
verification, so a synthesis failure leaves existing results intact. Input ranges
must satisfy `0 <= start < stop <= number of saved cases`; they are not silently clipped.
Baseline truth-table keys are checked before synthesis. Missing or insufficient baseline
data are skipped, while malformed or mismatched records cause an error.

The synthesizer is deterministic (no randomness in any path), asserts the pool XOR
invariant under `LIMBDD_FUSION_DEBUG=1`, and every emitted circuit is verified against its
truth table (`compute_tt`) during evaluation, outside the reported synthesis timing.

## Documentation

- `docs/architecture.md` — system design: module layout, synthesis routes, the fusion
  engine, the cost model, and the data layout.
- `docs/CompleteBranch_algebra.md` — the mathematics: Theorem 1 (branch = affine block),
  the C1–C4 classification and proofs of Theorem 2, the canonical RREF construction, and
  the single-block T-optimality lemma. Exploratory conjectures and historical probe
  appendices are explicitly distinguished from the current algorithm's guarantees.

## Results and data

Precomputed outputs over `n` up to 11, together with Exorcism and SSHR baselines and the
test-case sets, live under `data/`. Per-case T/CX costs and run summaries are stored in
`data/our_cost/{path}/`; default and `_noM2` files share the same base route directory.
Historical degree-bound files remain under `data/our_cost/deg_bound`; new runs include
the bound only in their summary. The format and coverage are documented in `data/README.md`.

## Paper figures

Install the optional plotting dependencies in the same Python environment:

```bash
python -m pip install -r requirements-plot.txt
python -m unittest discover -s tests -p test_plot_results.py
python -m experiments.plot_results
python -m experiments.plot_results --metric cx
python -m experiments.plot_results --metric time
```

Figures use the same first 1000 functions for each n=5..11 and are exported as
vector PDF and 300-dpi PNG. The default produces the T-cost comparison; `--metric cx`
uses the same layout for CX cost, and `--metric time` produces a linear time plot.
Colors identify the minterm, PK, ANF, and DD families; line styles and markers distinguish
raw covers, fusion, shift search, and proxy search. Points are joined without smoothing.
Figures label the saved A0/A routes as **DD0/DD**; source filenames and CLI route
names remain A0/A. Summary CSVs retain both the source route and its display name.

Outputs live in `figure/results/`, already ignored by Git. The companion
`starting_cover_summary.csv` contains total T/CX costs, reported mean synthesis time,
the exact case interval, and source filenames for all 84 route/size combinations.
It is regenerated from the original TXT files. The dependency file and plotting
script belong in Git; generated figures and summaries do not need to be committed.

The T/CX figures compare the eight optimized routes MT, PK, PKS, PKP, F, FS, FP, and DD.
The left panel shows absolute total cost on a linear axis; the right panel divides each route's total
by F's total **at the same n**. This is a ratio of batch totals, not an average of
per-function ratios. F is the horizontal reference at 1; lower is better. The
normalized panel has a nonzero lower axis limit to show relative differences.
The two panels share the same route colors, line styles, and markers.

`starting_cover_total_{t,cx}_comparison.pdf` and `.png` contain the combined figures.
`starting_cover_{t,cx}_comparison.csv` records all 56 plotted values and denominators;
`starting_cover_{t,cx}_fusion_reductions.csv` records the 28 per-size reductions from
MT0 to MT, PK0 to PK, F0 to F, and DD0 to DD. `starting_cover_{t,cx}_notes.md` supplies suggested
captions and reduction ranges. Raw-cover costs remain in the CSV files; the earlier
ten-route T-cost overview exports have been removed.

The time plot, `starting_cover_avg_wall_ms_linear.pdf` and `.png`, defaults to the same
eight optimized routes as the cost figures and reports **mean synthesis time in
ms/function** from each 1000-case batch. MT0, PK0, F0, and DD0 are reserved for a separate
comparison; their data remain in the full summary CSV. Explicit `--routes` selections
can include these baselines (use source name A0 for DD0).
Its linear axis starts at zero. For exploratory plots, `--view overview` accepts a
route subset and `--scale linear`, `log`, or `both`. The earlier `--view t-comparison`
command remains an alias for the default T-cost comparison.

To inspect the larger-n timing differences with a broken linear axis:

```bash
python -m experiments.plot_results --metric time --time-break 13000 27000
```

This writes `starting_cover_avg_wall_ms_broken.pdf` and `.png`, with two slashes on
the left axis and a narrow unlabeled gap. The empty 13000--27000 ms/function interval
is omitted and documented in `starting_cover_time_broken_notes.md`; all measured
points of the selected routes remain visible. `brokenaxes` manages the axis layout and break marks.
Cross-band segments use Matplotlib `ConnectionPatch` to join the actual endpoints
in display coordinates, keeping the route visually continuous after compression.
These segments are visual guides, not interpolated measurements within the omitted
interval. The two retained bands use the same linear scale per unit of physical
height. The script rejects a break containing
any observed point. The ordinary linear figure remains available for comparison.

For a compact version of the complete starting-cover figure:

```bash
python -m experiments.plot_results --view compact --scale log
python -m experiments.plot_results --view compact --time-break 13000 27000
```

These commands create `starting_cover_compact_log.{pdf,png}` and
`starting_cover_compact_broken.{pdf,png}`. Both retain the eight optimized routes
and all five original comparisons in one 7.16-by-3-inch figure: total T-count and
CNOT-count above their F-normalized ratios, with synthesis time spanning both rows
in the third column. The figure has one shared legend and 8-point tick labels at
its intended two-column width. Cost totals are expressed in millions; normalization
still divides batch totals by F at the same input size. No data points are removed.

The logarithmic time axis shows the full runtime range without a break; its spacing
represents ratios. The broken version keeps the original linear bands and uses
`ConnectionPatch` to join the observed endpoints of each cross-band segment
continuously in display coordinates. These links are visual guides, not
interpolated measurements within the omitted interval. Both bands retain the same
linear scale per unit of height, and the left axis retains its break marks. The data source is
the same regenerated `starting_cover_summary.csv` used by the separate figures.
Sample size and timing conventions belong in the paper caption, avoiding repeated
titles inside the combined figure. Suggested caption for the logarithmic version:

> Starting-cover and shift-search comparisons on 1,000 matched Boolean functions
> at each input size n=5,...,11. In (a) and (b), upper panels show total costs and
> lower panels show costs relative to F. Panel (c) shows mean synthesis times on
> a logarithmic axis. Timings are from separate batch runs.

For the broken version, replace "on a logarithmic axis" with "with the empty
13,000--27,000 ms interval omitted". Embed the combined vector PDF at full text
width in one figure rather than adding the previous three subfigure captions.

The script requires exact-interval T, CX, and summary files and validates their input
keys. It never substitutes a larger batch's mean runtime. Timing figures combine
separate batch runs and should be read as descriptive comparisons. To fill a missing
batch, use `python evaluate.py <n> 0 1000 <route>` before regenerating the figure.
