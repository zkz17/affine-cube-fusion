import argparse
from contextlib import ExitStack
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from qiskit.circuit.library.standard_gates.x import XGate, CXGate, MCXGate, CCXGate

from config import config
from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import anf_degree
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit

T_COST_FN = config['t_cost']
CX_COST_FN = config['cx_cost']

# All supported synthesis routes, selected by a single `path` argument.
# A/F/FS are the BDD/ANF/FPRM routes, FP is FS with the proxy prescreen, FG the GL search,
# MT/PK the engine routes over the minterm / pseudo-Kronecker covers, PKS/PKP their
# shift-search variants, and A0/F0/MT0/PK0 the covers emitted without the fusion engine.
ROUTES = ('A', 'F', 'FS', 'FP', 'FG', 'MT', 'PK', 'PKS', 'PKP', 'A0', 'F0', 'MT0', 'PK0')
ENGINE_ROUTES = ('A', 'F', 'FS', 'FP', 'FG', 'MT', 'PK', 'PKS', 'PKP')
RAW_ROUTES = ('A0', 'F0', 'MT0', 'PK0')
COVER_ROUTES = ('MT', 'PK', 'PKS', 'PKP') + RAW_ROUTES


def count_costs(circ):
    """
    Circuit (T, CX) cost. The scope matches data/exorcism_cost.py / sshr data:
      MCX/CCX are charged T_COST_FN(k) and CX_COST_FN(k) by their control count k;
      an explicit CX gate counts as 1 CX; X gates are free.
    """
    t = cx = 0
    for instruction in circ.data:
        gate = instruction.operation

        if isinstance(gate, MCXGate):
            t += T_COST_FN(gate.num_ctrl_qubits)
            cx += CX_COST_FN(gate.num_ctrl_qubits)
        elif isinstance(gate, CCXGate):
            t += T_COST_FN(2)
            cx += CX_COST_FN(2)
        elif isinstance(gate, CXGate):
            cx += 1
        elif isinstance(gate, XGate):
            pass
        else:
            print(f'Unrecognized gate type {type(gate)}.')
    return t, cx


def count_t_cost(circ):
    """T-only scope (compatible with existing callers)."""
    return count_costs(circ)[0]


def count_cx_cost(circ):
    """CX-only scope."""
    return count_costs(circ)[1]


def load_aligned(cases_path, cost_path, start, stop):
    """
    Read baseline costs only after checking their truth-table keys against the inputs.
    Missing, empty, or shorter baselines return costs=None; malformed or misaligned
    records in the requested interval are errors, including a shorter file's prefix.
    """
    with open(cases_path) as f:
        lines = f.readlines()
    if not 0 <= start < stop <= len(lines):
        raise ValueError(f'Require 0 <= start < stop <= {len(lines)} input cases.')
    cases = [int(line.rstrip('\n')) for line in lines[start:stop]]

    try:
        with open(cost_path) as f:
            clines = f.readlines()
    except FileNotFoundError:
        return cases, None
    costs = []
    for index, (case, line) in enumerate(zip(cases, clines[start:stop]), start=start):
        try:
            key, cost = map(int, line.split(':'))
        except ValueError as exc:
            raise ValueError(f'Malformed baseline record: {cost_path}, line {index + 1}') from exc
        if key != case:
            raise ValueError(f'Baseline truth-table mismatch: {cost_path}, line {index + 1}')
        if cost < 0:
            raise ValueError(f'Negative baseline cost: {cost_path}, line {index + 1}')
        costs.append(cost)
    if stop > len(clines):
        return cases, None
    return cases, costs


def _compare(name, cases, ours, baseline):
    """Single-scope comparison; skipped when baseline is None. Returns a list of report lines."""
    if baseline is None:
        return [f"===== {name}:baseline missing, skipped ====="]
    better, worse, same = 0, 0, 0
    worse_cases = []
    s_base = s_ours = 0
    for i, tt in enumerate(cases):
        t_ours = ours[i]
        t_base = baseline[i]
        s_base += t_base
        s_ours += t_ours
        if t_ours > t_base:
            worse += 1
            worse_cases.append(tt)
        elif t_ours == t_base:
            same += 1
        else:
            better += 1
    return [
        f"===== {name} =====",
        f"lower  cost in {better}({better / len(cases) * 100}%) cases",
        f"higher cost in {worse}({worse / len(cases) * 100}%) cases",
        f"same   cost in {same}({same / len(cases) * 100}%) cases",
        f"cases with higher cost: {worse_cases[:min(len(worse_cases), 20)]}",
        f"total cost: baseline {s_base}, ours {s_ours} "
        f"(reduction {1 - s_ours / max(1, s_base):.2%})",
    ]


def evaluate_all(num_var=3, start=0, stop=None, path='FS', m2_post=True, reorder=True,
                 *, output_dir='data/our_cost', overwrite=False):
    """
    Compare a synthesis route against Exorcism / SSHR (T and CX dual scopes).

    Usage: evaluate.py <n> <start> <stop> <path> [m2] [reorder]
                       [--output-dir DIR] [--overwrite]

    path    : one of A / A0 / F / FS / FP / FG / MT / PK / PKS / PKP / F0 / MT0 / PK0.
              A = BDD decomposition, F = plain ANF, FS = FPRM (shift search), FP = FS with the
              proxy prescreen, FG = deterministic GL(n,2) search, MT/PK = minterm /
              pseudo-Kronecker covers fused by the engine, PKS/PKP = their shift-search
              variants, F0/MT0/PK0 = raw ANF/minterm/PK covers emitted without fusion (baselines).
              A0 = A's complete DD preprocessing, then emission without fusion or segment reordering.
    m2      : macro M2 interleaved post-processing (1 = on, default; 0 = off, tagged `_noM2`).
              Ignored for A0/F0/MT0/PK0, which skip the engine entirely.
    reorder : DD variable reordering for A and A0 (1 = on, default); not segment reordering.

    Outputs are written to `{output_dir}/{path}/our_{t,cx}_cost_{tag}_{n}_{a}_{b}.txt` and
    `{output_dir}/{path}/our_results_{tag}_{n}_{a}_{b}.txt`. Both M2 settings share the
    base route directory; only filenames carry the `_noM2` suffix. The G-3 degree lower bound is
    reported inside the results file but no longer written as a separate file (it depends
    only on the test set, not on the route). Existing outputs require overwrite=True.
    Synthesis and verification finish before any result files are published.
    """
    if path not in ROUTES:
        raise ValueError(f'illegal route: {path}')
    cases_path = f'data/test_cases/test_cases_{num_var}'
    with open(cases_path) as f:
        n_total = len(f.readlines())
    stop = n_total if stop is None else stop
    if not 0 <= start < stop <= n_total:
        raise ValueError(f'Require 0 <= start < stop <= {n_total} input cases.')
    total = stop - start

    cases, exor_t = load_aligned(cases_path, f'data/exorcism_cost/exorcism_t_cost_{num_var}', start, stop)
    _, exor_cx = load_aligned(cases_path, f'data/exorcism_cost/exorcism_cx_cost_{num_var}', start, stop)
    _, sshr_t = load_aligned(cases_path, f'data/sshr_cost/sshr_t_cost_{num_var}', start, stop)
    _, sshr_cx = load_aligned(cases_path, f'data/sshr_cost/sshr_cx_cost_{num_var}', start, stop)

    tag = path if path in RAW_ROUTES else path + ('' if m2_post else '_noM2')
    outdir = Path(output_dir) / path
    suffix = f'{tag}_{num_var}_{start}_{stop}.txt'
    output_paths = [outdir / f'our_{kind}_{suffix}' for kind in ('t_cost', 'cx_cost', 'results')]
    if not overwrite:
        for target in output_paths:
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite {target}; use --output-dir or --overwrite.')

    # Ancilla-assisted synthesis is not an evaluation option: every route uses zero ancillas.
    kwargs = dict(reorder=reorder, ancilla_num=0, fused_backend=True, m2_post=m2_post)
    if path == 'FG':
        kwargs['gl_search'] = True
    else:
        if path in ('A', 'F', 'FS', 'FP'):
            kwargs['paths'] = ('A',) if path == 'A' else ('F',) if path == 'F' else ('FS',)
            if path == 'FP':
                kwargs['fprm_proxy'] = True
    synthesizer = OracleSynthesizer(**kwargs)
    cover = path if path in COVER_ROUTES else None

    ours_t, ours_cx, bounds = [], [], []
    wall_ns = 0
    for i, tt in enumerate(cases):
        if i % 1000 == 0:
            print(i, flush=True)
        truth_table = TruthTable(tt, num_var)
        t0 = time.perf_counter_ns()
        if cover:
            synthesizer.clear()
            synthesizer.allocate_input_qubit(num_var)
            synthesizer.process_cover_route(truth_table, cover)
        else:
            synthesizer.synthesize(truth_table)
        wall_ns += time.perf_counter_ns() - t0
        # Verify every emitted circuit, outside the reported synthesis timing.
        if not synthesizer.circ.verify(truth_table):
            raise AssertionError(f'circuit verification failed: path={tag}, n={num_var}, tt={tt}')
        t, cx = count_costs(to_qiskit(synthesizer.circ))
        bound = T_COST_FN(anf_degree(truth_table.bitstr, num_var))

        ours_t.append(t)
        ours_cx.append(cx)
        bounds.append(bound)

    avg_wall_ms = wall_ns / max(total, 1) / 1e6
    report = ["****** Comparison of oracle synthesis routes ******",
              f"number of input variables: {num_var} ({total} cases); path={tag}",
              f"avg synthesize wall: {avg_wall_ms:.2f} ms/case"]
    report += _compare("Exorcism T", cases, ours_t, exor_t)
    report += _compare("Exorcism CX", cases, ours_cx, exor_cx)
    report += _compare("SSHR T", cases, ours_t, sshr_t)
    report += _compare("SSHR CX", cases, ours_cx, sshr_cx)
    report.append(f"G-3 lower bound: Σ t(deg f) = {sum(bounds)} "
                  f"({sum(bounds) / max(1, sum(ours_t)):.2%} of ours ΣT)")
    report.append("****** ****** ******")

    text = '\n'.join(report) + '\n'
    contents = [''.join(f'{tt}:{cost}\n' for tt, cost in zip(cases, costs))
                for costs in (ours_t, ours_cx)] + [text]
    outdir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.evaluation-', dir=outdir) as staging:
        for target, content in zip(output_paths, contents):
            (Path(staging) / target.name).write_text(content, encoding='utf-8')
        if overwrite:
            for target in output_paths:
                os.replace(Path(staging) / target.name, target)
        else:
            # Exclusive creation also protects against another run finishing between
            # the initial existence check and publication. Roll back only our own files.
            created = []
            try:
                with ExitStack() as stack:
                    streams = []
                    for target in output_paths:
                        streams.append(stack.enter_context(target.open('x', encoding='utf-8')))
                        created.append(target)
                    for stream, content in zip(streams, contents):
                        stream.write(content)
            except BaseException:
                for target in created:
                    target.unlink()
                raise
    print(text, end='')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate and verify one oracle synthesis route.')
    parser.add_argument('n', nargs='?', type=int, default=3)
    parser.add_argument('start', nargs='?', type=int, default=0)
    parser.add_argument('stop', nargs='?', type=int, default=None)
    parser.add_argument('path', nargs='?', choices=ROUTES, default='FS')
    parser.add_argument('m2', nargs='?', type=int, choices=(0, 1), default=1)
    parser.add_argument('reorder', nargs='?', type=int, choices=(0, 1), default=1)
    parser.add_argument('--output-dir', type=Path, default=Path('data/our_cost'),
                        help='result root; each route gets a subdirectory (default: data/our_cost)')
    parser.add_argument('--overwrite', action='store_true', help='allow replacing an existing batch')
    args = parser.parse_args(argv)
    try:
        evaluate_all(args.n, args.start, args.stop, args.path, m2_post=bool(args.m2),
                     reorder=bool(args.reorder), output_dir=args.output_dir, overwrite=args.overwrite)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
