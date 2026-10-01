# Application-derived Boolean functions

These datasets extend the random-function experiments without changing their
inputs or results. Each output bit is synthesized as a **separate single-output
oracle**. They do not measure a complete cipher, a joint multi-output oracle, or
an end-to-end quantum algorithm.

- `test_cases_des_6`: 32 functions, four output bits for each of DES S1--S8.
- `test_cases_aes_8`: 8 functions, the eight bits of the forward AES S-box.
- `test_cases_ascon_5`: 5 functions, the five bits of the Ascon S-box.
- `test_cases_speedy_6`: 6 functions, the six bits of the forward SPEEDY S-box.

The manuscript's cryptographic S-box panel includes all four datasets: Ascon,
SPEEDY, DES, and AES, ordered by input count, with 51 output functions in total.
All five methods use every output in each group.

Every line is a decimal truth-table integer, exactly as in `../test_cases_{n}`.
Bit `a` is the value `f(a)`, and variable `x0` is the least significant bit of `a`.
**The fixed input counts are DES: 6, AES: 8, Ascon: 5, and SPEEDY: 6.**
All named outputs are retained in their original order; no support reduction or
deduplication is applied.
`case_index.csv` maps every one-based data line to its named output; evaluator
`--start` and `--stop` instead use zero-based, half-open indices.

## Sources and bit ordering

The compact S-box tables under `sources/` come from the following standards and
original specifications. The generator uses the checked-in text, so PDF parsing and network
access are not required to reproduce these datasets.

- **DES**: [FIPS 46-3](https://csrc.nist.gov/files/pubs/fips/46-3/final/docs/fips46-3.pdf),
  S1--S8 on printed pages 17--18 (PDF pages 22--23). Reference PDF SHA-256:
  `38dc009ca59d391814328fbbf3df0dfe30c69e75dc22b280efd807621e0244b1`.
  An input integer `a` is the six-bit word `b1...b6` in the standard, with `b1`
  the most significant bit. Its row is `2*b1+b6`, and its column is `b2...b5`.
  Outputs are ordered S1 through S8; within each box, bits 0 through 3 are
  **least significant first**. Thus `x0=b6`, `x5=b1`; `S1[0]` is the least
  significant bit of the four-bit output. This is the DES S-box component,
  without expansion, key mixing, or the DES P permutation.
- **AES**: [FIPS 197-upd1](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf),
  Table 4 on printed page 14 (PDF page 22). Reference PDF SHA-256:
  `62c86eb567f13edb8f71826e985da870b04ef6381634f303cdb16e84d47becd1`.
  Input `a` is the byte value; the high nibble selects the table row and the low
  nibble selects the column. Output bits 0 through 7 are least significant first.
  This is the forward SubBytes S-box, not its inverse.
- **Ascon**: [NIST SP 800-232](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-232.pdf)
  (August 2025), Section 3.3, Table 6 on printed page 10 (PDF page 21).
  The 32 hexadecimal values are checked against all five Boolean expressions in
  Eq. (7) on printed page 9. The specification writes the five-bit input tuple
  with its `x0` most significant. Our input `xj` therefore corresponds to its
  `x(4-j)`, and our `Sbox[j]` to its `y(4-j)`. This measures one five-bit S-box,
  not the 320-bit permutation, the linear diffusion layer, or an AEAD/hash oracle.
- **SPEEDY**: [Leander et al., The SPEEDY Family of Block Ciphers](https://eprint.iacr.org/2021/960.pdf),
  Table 3 and the DNF expressions in Section 3.3, printed page 14 (PDF page 14).
  All 64 table entries were also checked against the
  [authors' reference implementation](https://github.com/Chair-for-Security-Engineering/SPEEDY/blob/5f1b2c01f690e2cdb81a1d136496b36726e8022e/Software/Reference%20Implementation/C%20Code/SPEEDY.c).
  The specification uses most-significant-first indices: our `xj` corresponds
  to its `x(5-j)`, and our `Sbox[j]` to its `y(5-j)`. This is the forward six-bit
  substitution only, without key mixing, ShiftColumns or MixColumns.

The generator pins SHA-256 hashes of the S-box source text, using LF-normalized
text to accommodate Windows checkouts. AES is additionally checked at every byte
against its GF(256) inversion and affine-map definition. DES uses the standard's
row/column rule, row-permutation checks, and the published `S1(011011)=0101` example.
Ascon and SPEEDY are checked over their entire domains against the independent
Boolean definitions above, as well as for bijectivity. The hexadecimal input
and output values are preserved; only component naming uses our LSB-first convention.

## Generate and validate

Run from the repository root, with any Python 3.12 interpreter (standard library
only):

```bash
python experiments/generate_structured_cases.py
python experiments/generate_structured_cases.py --check
```

Generation refuses to replace differing files unless `--overwrite` is explicit.
The sources are intentionally small and checked in so all four truth-table files
can be regenerated offline. No reduced-support variants are generated.

## Evaluate

`evaluate_structured.py` is independent of `evaluate.py`. It fixes M2 on and ACF
synthesis ancillas to zero, using the same route settings and `config.py` cost
model as the existing experiments. Default commands run **every output** in the
specified dataset, sequentially; there is no sampling or repeated subset timing.

```bash
python evaluate_structured.py run des F
python evaluate_structured.py run des FS
python evaluate_structured.py run des FP
python evaluate_structured.py run des exorcism
python evaluate_structured.py run des sshr --sshr-root ../quantum-sshr
python evaluate_structured.py summarize des
```

Replace `des` with `aes`, `ascon` or `speedy` for the other groups.
ACF needs the project's ordinary requirements. Exorcism needs the existing
`caterpillar_bindings` native module in `data/` (or `--exorcism-dir`); it must match
the operating system and Python ABI. The supplied local `.so` works in Linux/WSL,
not native Windows. Truth-table words are padded to their full fixed input width,
including zero high words.

SSHR-H uses the primitives from [quantum-sshr](https://github.com/grrain/quantum-sshr)
(the local reference checkout is commit `ec670bb3c8374da7e3c39965d6985096a931593c`).
The small adapter scans dimensions from `n-1` down to 1 and applies adjacent-X
cancellation, matching the paper-profile interface used for our earlier baseline
experiments. It does **not** call the repository's legacy `f1` entry, which starts
at `n-2`, and does not require our private `sshr_test` wrapper. Its imports require
Qiskit Aer even though the evaluator uses its own exhaustive permutation check.
Use SSHR's existing environment if its dependencies are separate:

```bash
../quantum-sshr/.venv/bin/python evaluate_structured.py run des sshr \
  --sshr-root ../quantum-sshr
```

Workers use the interpreter that launches the evaluator. Each worker records
Python/package versions, OS, and a digest of the evaluator,
ACF source and selected external backend; a resumed run rejects changes.
Each output is evaluated in a fresh worker, so process-local caches are not
shared across different functions. All workers execute sequentially.

### Time limits and retained results

SSHR-H defaults to a **1,200-second synthesis limit per function** when requested.
The worker is terminated when this limit is reached. Imports/setup and
verification each have a separate 180-second guard; those failures are errors,
never synthesis timeouts. ACF and Exorcism have no default synthesis limit.
`--timeout SECONDS` is available for diagnostic runs; use a separate output
directory when changing the limit.

Each completed case, timeout or algorithm error is atomically recorded in
`data/structured_cost/<dataset>/<method>/results.csv`. It contains the original
output index, full truth table, fixed n, method/settings, source digest,
environment, status, costs, time and verification status. Costs and accepted
times are **blank** for failed cases. Failures stop the batch for investigation;
synthesis timeouts allow it to continue. Dependency/setup errors stop before a
case result is accepted. No existing random-function file is read or replaced.

Use the same command with `--resume` after an interruption; already recorded
timeouts and errors are also retained and skipped. Remeasurements belong in a
new `--output-dir`, not in a mixed run. A `.running` lock prevents concurrent
writers; if the parent is forcibly killed, confirm that its worker has stopped
before removing the stale lock. To exercise a few cases without polluting the
paper results, use e.g. `--start 0 --stop 1 --output-dir tmp/structured-smoke`.

Synthesis times exclude imports, process startup, verification, accounting and
file I/O. Exorcism returns only costs through two separate synthesis interfaces;
its single time is the arithmetic mean of those two calls. Its circuit cannot be
independently verified through that cost-only interface. Every ACF and SSHR
circuit is exhaustively checked for the correct target XOR and restored inputs,
for both initial target values. This verifies the abstract X/CX/MCX oracle,
not a physical Clifford+T decomposition.

`summarize` recomputes arithmetic means across the **entire named dataset** and
aligned percentage cost reductions from its total costs. A group with any
unrun, timed-out or failed function has **no full-dataset mean or reduction**.
The report gives completed/total and timeout/error counts instead. Do not compare
an easy completed SSHR subset against all outputs of another method. The
per-method `summary.txt` is a snapshot taken at that run's end; recompute with
`summarize` after other methods finish. Partial smoke runs are not paper data.

The focused evaluator/extraction regression checks can be run in the ACF
environment with `python -m unittest discover -s tests -p test_structured.py -v`.
They cover timeout termination, crashed workers, resume protection, truth-table
alignment, input restoration and suppression of incomplete baseline averages.
