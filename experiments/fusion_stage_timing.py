"""Serial, matched timing of raw / pairwise / full synthesis.

The default experiment uses 20 functions per size (two from each of the ten
original sampling intervals), three repetitions, and all four cover families.
Each function visits all three execution positions in successive repetitions;
pairwise/full precedence is balanced across functions. No synthesis state is
shared between runs. Verification and gate counting are outside the timer.

Example (run from the repository root, with fusion debugging disabled)::

    python experiments/fusion_stage_timing.py

Outputs are a human-readable summary and a tab-separated TXT run log under
data/our_cost/fusion_timing/. Existing outputs are never overwritten. For a
short validation run, pass --sizes 5 --cases 2 --prefix smoke --output-dir ...
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import FUSION_DEBUG
from evaluate import COVER_ROUTES, count_costs
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit

MODES = ('raw', 'pairwise', 'full')
FIELDS = ('route n case_index repeat position mode wall_ns cpu_ns T CX tt').split()


def execution_order(rank, repeat):
    base = list(MODES if rank % 2 == 0 else ('raw', 'full', 'pairwise'))
    offset = (rank + repeat - 1) % 3
    return base[offset:] + base[:offset]


def source_hash():
    paths = [Path(__file__), ROOT / 'config.py', ROOT / 'evaluate.py']
    for folder in ('backend', 'circuit', 'limbdd', 'utils'):
        paths.extend((ROOT / folder).rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_expected(route, n, functions):
    result = {}
    for mode in MODES:
        folder = route + '0' if mode == 'raw' else route
        tag = route + '_noM2' if mode == 'pairwise' else folder
        costs = []
        for metric in ('t', 'cx'):
            path = ROOT / f'data/our_cost/{folder}/our_{metric}_cost_{tag}_{n}_0_1000.txt'
            rows = [tuple(map(int, line.split(':'))) for line in path.read_text().splitlines()]
            assert len(rows) == 1000 and [row[0] for row in rows] == functions[:1000], path
            costs.append([row[1] for row in rows])
        result[mode] = list(zip(*costs))
    return result


def run_one(route, n, fn, mode):
    path = route + '0' if mode == 'raw' else route
    kwargs = dict(reorder=True, ancilla_num=0, fused_backend=True, m2_post=mode == 'full')
    if path in ('A', 'F'):
        kwargs['paths'] = (path,)
    synth = OracleSynthesizer(**kwargs)
    tt = TruthTable(fn, n)
    gc.collect()
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    if path in COVER_ROUTES:
        synth.clear()
        synth.allocate_input_qubit(n)
        synth.process_cover_route(tt, path)
    else:
        synth.synthesize(tt)
    wall_ns = time.perf_counter_ns() - start
    cpu_ns = time.process_time_ns() - cpu_start
    assert synth.circ.verify(tt), (route, n, mode, fn)
    t, cx = count_costs(to_qiskit(synth.circ))
    return dict(route=route, n=n, mode=mode, wall_ns=wall_ns, cpu_ns=cpu_ns, T=t, CX=cx, tt=fn)


def summary(rows, routes, sizes):
    lines = ['route n runs_per_mode raw_ms pairwise_ms full_ms M2_time_change_pct repeat_changes_pct']
    for route in routes:
        for n in sizes:
            group = [r for r in rows if r['route'] == route and r['n'] == n and r['repeat'] > 0]
            groups = {m: [r for r in group if r['mode'] == m] for m in MODES}
            # Partial triplets are not summarized while the process is running.
            counts = [len(groups[m]) for m in MODES]
            if not min(counts) or len(set(counts)) != 1:
                continue
            means = {m: statistics.mean(r['wall_ns'] for r in groups[m]) / 1e6 for m in MODES}
            change = 100 * (means['full'] / means['pairwise'] - 1)
            repeats = []
            for repeat in sorted({r['repeat'] for r in group}):
                pair = [r['wall_ns'] for r in groups['pairwise'] if r['repeat'] == repeat]
                full = [r['wall_ns'] for r in groups['full'] if r['repeat'] == repeat]
                repeats.append(100 * (statistics.mean(full) / statistics.mean(pair) - 1))
            lines.append(f'{route} {n} {counts[0]} ' + ' '.join(f'{means[m]:.6f}' for m in MODES)
                         + f' {change:+.6f} ' + ','.join(f'{v:+.6f}' for v in repeats))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=list(range(5, 12)))
    parser.add_argument('--routes', nargs='+', choices=('MT', 'PK', 'F', 'A'), default=['MT', 'PK', 'F', 'A'])
    parser.add_argument('--cases', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--prefix', default='fusion_stage_timing_20260926')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/our_cost/fusion_timing')
    args = parser.parse_args()
    assert not FUSION_DEBUG, 'set LIMBDD_FUSION_DEBUG=0 for timing'
    assert 2 <= args.cases < 1000 and args.cases % 2 == 0
    assert args.repeats >= 3 and args.repeats % 3 == 0
    assert len(set(args.sizes)) == len(args.sizes) and all(5 <= n <= 11 for n in args.sizes)
    assert len(set(args.routes)) == len(args.routes)
    assert Path(args.prefix).name == args.prefix
    indices = [i * 1000 // args.cases for i in range(args.cases)]
    warmup = next(i for i in range(1000) if i not in indices)
    # Each function appears equally often in every execution position.
    for rank in range(args.cases):
        for mode in MODES:
            positions = [execution_order(rank, r).index(mode) for r in range(1, args.repeats + 1)]
            assert [positions.count(p) for p in range(3)] == [args.repeats // 3] * 3
    precedence = [execution_order(k, r).index('pairwise') < execution_order(k, r).index('full')
                  for k in range(args.cases) for r in range(1, args.repeats + 1)]
    assert sum(precedence) * 2 == len(precedence)
    functions = {n: [int(line) for line in (ROOT / f'data/test_cases/test_cases_{n}').read_text().splitlines()]
                 for n in args.sizes}
    for n in args.sizes:
        width = (1 << (1 << n)) // 10
        assert all(min(functions[n][i] // width, 9) == i // 100 for i in indices)
    expected = {(route, n): load_expected(route, n, functions[n]) for route in args.routes for n in args.sizes}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f'{args.prefix}.txt'
    raw_path = args.output_dir / f'{args.prefix}_runs.txt'
    assert not report_path.exists() and not raw_path.exists(), 'choose a new output prefix'
    signature = source_hash()
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                   if line.startswith('model name'))
    header = ['Serial matched three-stage synthesis timing',
              f'started UTC: {datetime.now(timezone.utc).isoformat()}',
              f'git revision: {subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}',
              f'source SHA256 (including probe): {signature}',
              f'platform: {platform.platform()}; CPU: {cpu}; Python: {platform.python_version()}',
              'packages: ' + ', '.join(f'{name}={version(name)}' for name in ('numpy', 'qiskit', 'scipy')),
              f'routes={args.routes}; sizes={args.sizes}; indices={indices}; repeats={args.repeats}',
              f'counts in original strata: {[sum(i // 100 == j for i in indices) for j in range(10)]}',
              f'warmup index={warmup}; one excluded run per configuration and size (repeat=0)',
              'one worker; fresh complete synthesis per run; zero ancillas; DD reorder/splitting enabled',
              'three-stage execution positions balanced per function; pairwise/full precedence balanced overall',
              'function order rotated by seven positions per repeat; normal Python GC enabled',
              'constructor, explicit pre-run GC, verification, Qiskit conversion, cost counting and I/O untimed',
              'all circuits verified; every T/CX pair checked against the corresponding saved 1000-case result',
              'means include all measured repeats; M2 change = 100*(mean(full)/mean(pairwise)-1)',
              'these sample means do not replace or represent timings of complete 1000-case batches',
              f'raw run log: {raw_path.name}']
    for n in args.sizes:
        header.append(f'input n={n} SHA256: ' + hashlib.sha256((ROOT / f'data/test_cases/test_cases_{n}').read_bytes()).hexdigest())
    rows = []

    def save(status):
        report_path.write_text('\n'.join(header + [f'status: {status}', '']
                                        + summary(rows, args.routes, args.sizes)) + '\n', encoding='utf-8')

    save('RUNNING')
    try:
        with raw_path.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter='\t')
            writer.writeheader()
            stream.flush()
            for n in args.sizes:
                for route in args.routes:
                    print(f'{route} n={n}: warmup, then {args.cases} functions x {args.repeats} repeats x 3 stages', flush=True)
                    for repeat in range(args.repeats + 1):
                        rotated = indices[7 * (repeat - 1) % len(indices):] + indices[:7 * (repeat - 1) % len(indices)]
                        work = [(warmup, list(MODES))] if repeat == 0 else [
                            (i, execution_order(indices.index(i), repeat)) for i in rotated]
                        for index, order in work:
                            for position, mode in enumerate(order, 1):
                                row = run_one(route, n, functions[n][index], mode)
                                assert (row['T'], row['CX']) == expected[route, n][mode][index], (route, n, index, mode)
                                row.update(case_index=index, repeat=repeat, position=position)
                                writer.writerow(row)
                                stream.flush()
                                rows.append(row)
                            save('RUNNING')
                        print(f'{route} n={n} repeat={repeat} complete', flush=True)
                    print(summary(rows, [route], [n])[-1], flush=True)
                    assert source_hash() == signature, 'production source changed during timing'
        save('COMPLETE; all circuits verified and all costs match saved results')
    except BaseException as exc:
        save(f'INCOMPLETE: {type(exc).__name__}: {exc}')
        raise
    print(f'Complete: {report_path}', flush=True)


if __name__ == '__main__':
    main()
