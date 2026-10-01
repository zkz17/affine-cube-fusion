# Cryptographic S-box experiment results

The datasets are DES S1--S8 (32 output functions, n=6), the forward AES S-box
(8 output functions, n=8), Ascon (5 output functions, n=5), and the forward
SPEEDY S-box (6 output functions, n=6). The manuscript's cryptographic S-box
panel includes all four datasets. Each output is synthesized independently.
These are component-oracle experiments, not complete cipher implementations or
joint multi-output S-box synthesis. Sources and the measurement protocol are in
[the dataset guide](../test_cases/structured/README.md).

## Authoritative measurements

`<dataset>/<method>/results.csv` contains one record per output, aligned by index
and truth-table integer with `../test_cases/structured/case_index.csv`. Methods
are F, FS, FP, Exorcism and SSHR-H. Costs use the same `config.py` model as the
random-function experiments; synthesis timings are arithmetic means over every
output in the named dataset, without verification, imports or file I/O.

- DES: all five methods completed all 32 outputs.
- AES: F, FS, FP and Exorcism completed all 8 outputs. SSHR-H attempted all 8 and
  each reached the 1200-second synthesis limit; no SSHR-H costs or completed-run
  average time are available. Empty SSHR cost exports are intentional.
- The original DES/AES data have 192 successful measurements and 8 timeout records, with no unrun or
  error records. A timeout must not be treated as zero cost or a completed
  1200-second synthesis time.
- Those 120 ACF and 32 completed SSHR-H circuits were exhaustively verified as
  phase-free X/CX/MCX permutations with input restoration and target XOR action.
  Exorcism's external interface supplies costs only, so its records explicitly
  mark circuit verification unavailable.
- Ascon and SPEEDY: all five methods completed all 5 and 6 outputs, respectively.
  All 55 new records are successful, without timeouts or errors. Their 33 ACF
  and 11 SSHR-H circuits passed the same exhaustive verification; the 11 Exorcism
  records use its cost-only interface. Across all four datasets there are now
  247 successful measurements and the original 8 AES SSHR-H timeouts.

The recorded implementation fingerprints and environment strings describe the
original measurements. Extending or cleaning up dataset generators changes the current
fingerprint, but does not change these measurements or the synthesis algorithms.
Keep existing CSV fingerprints unchanged. New measurements belong in a separate
output directory; do not replace or reinterpret historical timings.

The 2026-09-30 cleanup audit regenerated both ordered datasets, validated all
200 CSV records and 33 text exports, and reproduced the saved T/CNOT costs of
all 120 ACF circuits with exhaustive oracle verification. Source tables, truth
tables and measurement CSVs remained byte-identical; no recorded timing was
replaced by these verification-only reruns. The 10 focused regressions passed.

The subsequent Ascon/SPEEDY runs used the same sequential, fresh-worker protocol
and the same WSL interpreter/package environments as the corresponding DES/AES
methods. Source tables were checked against their independent Boolean definitions
over all inputs. All 55 new CSV records and 34 new text exports passed alignment
and completeness checks; all 67 text exports across the four datasets agree with
their CSV sources. The 59 existing DES/AES data/source files, including measured
timings, remained byte-identical. The updated 10 focused regressions passed.

`<dataset>/summary.txt` is the full current aggregate. Per-method `summary.txt`
files are historical snapshots from the end of that method's run and may precede
other methods' completion. Recompute an aggregate without running synthesis:

```bash
python evaluate_structured.py summarize des
python evaluate_structured.py summarize aes
python evaluate_structured.py summarize ascon
python evaluate_structured.py summarize speedy
```

## Text exports and analysis

Exports use the existing `tt:integer_cost` format and live next to the CSV data:

- `<dataset>/{F,FS,FP}/our_{t_cost,cx_cost,results}_{route}.txt`;
- `<dataset>/exorcism_cost/exorcism_{t,cx}_cost`, `exorcism_time`,
  and `exorcism_results.txt`;
- `<dataset>/sshr_cost/sshr_{t,cx}_cost` and `sshr_results.txt`, plus `sshr_time`
  only for a fully completed group.

```bash
python experiments/export_structured_results.py --check
python experiments/analyze_structured_results.py --output-dir runs/sbox-analysis
```

The exporter refuses differing existing files; it never reruns synthesis. Its
`--source-dir` and `--data-dir` both default to `data/structured_cost` and denote
roots containing dataset subdirectories. The analyzer creates a new directory
with source checksums, completeness counts, group means, aligned comparisons,
and per-output results. Incomplete groups have no numeric mean or reduction.

## Interpretation for the paper

FS reduces aggregate T/CNOT costs against Exorcism on DES and AES, and against
SSHR-H on DES. AES supports a timeout comparison with SSHR-H, not a gate-cost
comparison. F's DES aggregate CNOT cost is slightly higher than Exorcism's, and
one DES output also has a higher FS CNOT cost than Exorcism; avoid statements
that all configurations or all individual outputs improve both costs.

The Ascon and SPEEDY groups have the following full-dataset observations. These
are means across scalar output functions, not costs for a jointly optimized S-box.

- **Ascon:** F and FS have identical per-output T/CNOT costs, averaging 8.4/13.2.
  FP has the same mean T-count and a lower mean CNOT-count of 11.4. FS reduces
  T/CNOT costs by 51.16%/17.50% versus Exorcism and 65.85%/54.48% versus SSHR-H.
  Mean synthesis times are F 0.903 ms, FS 3.687 ms, FP 1.544 ms, Exorcism
  0.505 ms, and SSHR-H 4.184 ms. Shift search does not improve T-count here.
- **SPEEDY:** FS and FP both average 62.67 T gates, with 56.67 and 55.50 CNOTs,
  respectively. FS reduces T/CNOT costs by 15.88%/4.23% versus Exorcism and
  26.71%/25.76% versus SSHR-H. Mean synthesis times are F 1.385 ms, FS 22.516 ms,
  FP 4.619 ms, Exorcism 0.650 ms, and SSHR-H 386.940 ms. F's mean CNOT-count is
  10.14% higher than Exorcism's; its T-count is only 2.24% lower.

The two new datasets thus retain both improvements and limitations. In these
groups FP matches FS's mean T-count while using fewer CNOTs and less synthesis
time. FS still has a higher CNOT-count than Exorcism on 1/5 Ascon outputs and
2/6 SPEEDY outputs, despite the lower aggregate costs. Do not infer a universal
ranking of F, FS and FP from these small groups.
The manuscript reports F, FS and FP against both baselines in panel (c) of its
external-baseline table, ordered Ascon, SPEEDY, DES, AES by input count.
All means and reductions are computed
from the unrounded CSV records. AES SSHR-H is marked TO, with no cost estimate
or completed-run time. All 51 output functions are included in the manuscript;
the compact starting-cover figure leaves room for all four S-box groups.
