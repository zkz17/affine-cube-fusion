"""Sequential, resumable evaluation of fixed-width application-derived functions.

Examples (run from the repository root):
  python evaluate_structured.py run des F
  python evaluate_structured.py run aes exorcism
  python evaluate_structured.py run des sshr --sshr-root ../quantum-sshr
  python evaluate_structured.py summarize des

External backends are imported only in their worker process. See
data/test_cases/structured/README.md for dependencies and timeout/reporting rules.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

from experiments.generate_structured_cases import DATASETS, ROOT, load_cases


METHODS = ('F', 'FS', 'FP', 'exorcism', 'sshr')
OUTPUT_DIR = ROOT / 'data/structured_cost'
PROTOCOL = 'structured-v1;M2=on;ancilla=0;sshr-H:k=n-1;Exorcism:mean-of-two-calls'
FIELDS = ('dataset', 'n', 'index', 'truth_table', 'method', 'protocol', 'timeout_s',
          'implementation', 'environment', 'status', 't_count', 'cx_count',
          'synthesis_ms', 'verification', 'error', 'finished_utc')


def cost_and_verify(circuit, tt, n, *, reversed_inputs=False):
    """Check the exact X/CX/MCX permutation on both target values; charge config.py.

    Truth tables for all wires are bit-packed, so this includes input restoration,
    target XOR semantics and open controls without expensive statevector simulation.
    Only phase-free X-family gates are accepted; unsupported gates are an error.
    """
    from qiskit.circuit.library import XGate, CXGate, CCXGate, MCXGate
    from config import config

    if circuit.num_qubits != n + 1 or circuit.num_clbits or float(circuit.global_phase) % (2 * math.pi):
        raise ValueError('Expected an n+1-qubit, phase-free oracle without classical bits')
    assignments = 1 << (n + 1)
    mask = (1 << assignments) - 1
    original = [sum(((a >> j) & 1) << a for a in range(assignments)) for j in range(n + 1)]
    wires = list(original)
    t_count = cx_count = 0
    for instruction in circuit.data:
        gate = instruction.operation
        if not isinstance(gate, (XGate, CXGate, CCXGate, MCXGate)):
            raise ValueError(f'Unsupported gate in verification/cost model: {gate.name}')
        indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        k = len(indices) - 1
        ctrl_state = getattr(gate, 'ctrl_state', (1 << k) - 1)
        active = mask
        for j, wire in enumerate(indices[:-1]):
            active &= wires[wire] if (ctrl_state >> j) & 1 else mask ^ wires[wire]
        wires[indices[-1]] ^= active
        t_count += config['t_cost'](k)
        cx_count += config['cx_cost'](k)
    expected_flip = 0
    for a in range(assignments):
        x = a & ((1 << n) - 1)
        if reversed_inputs:
            x = int(f'{x:0{n}b}'[::-1], 2)
        expected_flip |= ((tt >> x) & 1) << a
    if wires[:n] != original[:n] or wires[n] != original[n] ^ expected_flip:
        raise ValueError('Circuit does not implement the requested oracle or restore inputs')
    return int(t_count), int(cx_count)


def tt_words(tt, n):
    """Kitty's little-endian uint64 blocks, including leading zero blocks."""
    return [(tt >> offset) & ((1 << 64) - 1) for offset in range(0, 1 << n, 64)]


def _sshr_h(tt, n, primitives):
    """Same paper-profile scan as the existing SSHR-H comparison interface.

    Reuse upstream primitives; start at n-1 rather than the legacy f1's n-2.
    Keep this small adapter here so no private sshr_test wrapper is required.
    """
    from qiskit import QuantumCircuit
    minterms = primitives.string2list(format(tt, f'0{1 << n}b'))
    circuit = QuantumCircuit(n + 1)
    while minterms:
        k = n - 1
        while k >= 1:
            found, block = primitives.check_k(k, minterms, n)
            if found:
                circuit = circuit.compose(block)
            else:
                k -= 1
        if minterms:
            circuit = circuit.compose(primitives.term2MCT(format(minterms[0], f'0{n}b')))
            minterms = []
    return primitives.optimize_adjacent_x(circuit)


def _implementation_digest(extra_files):
    # Resume refuses a changed evaluator, production code, or external backend.
    files = [Path(__file__), ROOT / 'config.py', ROOT / 'experiments/generate_structured_cases.py']
    for folder in ('backend', 'circuit', 'utils', 'limbdd'):
        files.extend(sorted((ROOT / folder).rglob('*.py')))
    files.extend(extra_files)
    digest = hashlib.sha256()
    for path in files:
        content = path.read_bytes()
        if path.suffix == '.py':
            content = content.replace(b'\r\n', b'\n')
        digest.update(path.name.encode())
        digest.update(content)
    return digest.hexdigest()


def _prepare_backend(method, options):
    extra_files = []
    if method in ('F', 'FS', 'FP'):
        from backend.OracleSynthesizer import OracleSynthesizer
        backend = OracleSynthesizer(ancilla_num=0, fused_backend=True, m2_post=True,
                                    paths=('F',) if method == 'F' else ('FS',),
                                    fprm_proxy=method == 'FP')
    elif method == 'exorcism':
        sys.path.insert(0, options['exorcism_dir'])
        backend = importlib.import_module('caterpillar_bindings')
        extra_files.append(Path(backend.__file__))
        for name in ('synthesize_stg_from_tt_tcost', 'synthesize_stg_from_tt_cxcost'):
            if not hasattr(backend, name):
                raise ValueError(f'Exorcism binding missing {name}')
    else:
        if not options['sshr_root']:
            raise ValueError('SSHR requires --sshr-root pointing to the quantum-sshr checkout')
        checkout = Path(options['sshr_root'])
        sys.path.insert(0, str(checkout))
        backend = importlib.import_module('q_lib.package.functional')
        if not Path(backend.__file__).resolve().is_relative_to(checkout):
            raise ValueError('Imported SSHR does not belong to --sshr-root')
        extra_files.extend(sorted((checkout / 'q_lib').rglob('*.py')))
    from importlib.metadata import version, PackageNotFoundError
    packages = []
    for name in ('numpy', 'qiskit', 'qiskit-aer', 'scipy'):
        try:
            packages.append(f'{name}={version(name)}')
        except PackageNotFoundError:
            pass
    info = {'implementation': _implementation_digest(extra_files),
            'environment': f'{platform.platform()};Python={platform.python_version()};' + ';'.join(packages)}
    return backend, info


def _worker(connection, method, tt, n, options):
    phase = 'setup'
    try:
        backend, info = _prepare_backend(method, options)
        # Imports, object creation and truth-table wrapping are outside synthesis time.
        if method in ('F', 'FS', 'FP'):
            from utils.TruthTable import TruthTable
            from utils.to_qiskit import to_qiskit
            truth_table = TruthTable(tt, n)
        if method == 'exorcism':
            words = tt_words(tt, n)
        connection.send(('ready', info))
        if connection.recv() != 'go':
            return
        phase = 'synthesis'
        start = time.perf_counter_ns()
        if method in ('F', 'FS', 'FP'):
            backend.synthesize(truth_table)
        elif method == 'exorcism':
            t_count = backend.synthesize_stg_from_tt_tcost(n, 0, words)
            cx_count = backend.synthesize_stg_from_tt_cxcost(n, 0, words)
        else:
            circuit = _sshr_h(tt, n, backend)
        elapsed = time.perf_counter_ns() - start
        synthesis_ms = elapsed / 1e6 / (2 if method == 'exorcism' else 1)
        connection.send(('synthesized', synthesis_ms))
        phase = 'verification'
        if method in ('F', 'FS', 'FP'):
            if not backend.circ.verify(truth_table):
                raise ValueError('ACF circuit verification failed')
            t_count, cx_count = cost_and_verify(to_qiskit(backend.circ), tt, n)
        elif method == 'sshr':
            t_count, cx_count = cost_and_verify(circuit, tt, n, reversed_inputs=True)
        if t_count < 0 or cx_count < 0:
            raise ValueError('Negative circuit cost')
        connection.send(('result', dict(status='complete', t_count=int(t_count), cx_count=int(cx_count),
                                        synthesis_ms=synthesis_ms,
                                        verification='unavailable-cost-only-interface' if method == 'exorcism'
                                        else 'exhaustive-X-permutation', error='')))
    except BaseException:
        connection.send(('error', f'{phase}: {traceback.format_exc()}'))
    finally:
        connection.close()


def _receive(connection, process, timeout, phase):
    if not connection.poll(timeout):
        if phase == 'synthesis':
            return 'timeout', None
        raise RuntimeError(f'Worker exceeded {phase} time limit ({timeout}s)')
    try:
        return connection.recv()
    except EOFError as exc:
        raise RuntimeError(f'Worker exited unexpectedly during {phase} (exit={process.exitcode})') from exc


def run_case(method, tt, n, options, timeout, expected_info=None, *, worker=_worker):
    """Hard per-case synthesis timeout; setup/verification failures are not TOs."""
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=worker, args=(child, method, tt, n, options))
    process.start()
    child.close()
    try:
        kind, info = _receive(parent, process, 180, 'setup')
        if kind != 'ready':
            raise RuntimeError(f'Backend setup failed: {info}')
        if expected_info and info != expected_info:
            raise ValueError('Implementation/environment changed; use a separate output directory')
        parent.send('go')
        try:
            kind, value = _receive(parent, process, timeout, 'synthesis')
        except RuntimeError as exc:
            return dict(status='error', error=str(exc)), info
        if kind == 'timeout':
            result = dict(status='timeout', error=f'Synthesis exceeded {timeout:g}s')
        elif kind == 'error':
            result = dict(status='error', error=value)
        elif kind == 'synthesized':
            try:
                kind, value = _receive(parent, process, 180, 'verification')
            except RuntimeError as exc:
                return dict(status='error', error=str(exc)), info
            result = value if kind == 'result' else dict(status='error', error=value)
        else:
            raise RuntimeError(f'Unexpected worker message: {kind}')
        return result, info
    finally:
        parent.close()
        process.join(0.2)
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def read_records(path, name, method, cases):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f'Unexpected record schema: {path}')
        rows = list(reader)
    seen = set()
    for row in rows:
        index = int(row['index'])
        if (row['dataset'] != name or row['method'] != method or int(row['n']) != DATASETS[name].n
                or row['protocol'] != PROTOCOL or index in seen or not 0 <= index < len(cases)
                or int(row['truth_table']) != cases[index]):
            raise ValueError(f'Misaligned/incompatible result record: {path}, index={index}')
        seen.add(index)
        if row['status'] not in ('complete', 'timeout', 'error'):
            raise ValueError(f'Unknown status: {path}')
        if row['status'] == 'complete':
            if (int(row['t_count']) < 0 or int(row['cx_count']) < 0
                    or not math.isfinite(float(row['synthesis_ms'])) or float(row['synthesis_ms']) < 0):
                raise ValueError(f'Invalid completed measurement: {path}')
        elif any(row[key] for key in ('t_count', 'cx_count', 'synthesis_ms')):
            raise ValueError(f'Failed cases must not contain accepted costs/times: {path}')
    if rows and any((row['implementation'], row['environment'], row['timeout_s']) !=
                    (rows[0]['implementation'], rows[0]['environment'], rows[0]['timeout_s']) for row in rows):
        raise ValueError(f'Mixed run settings: {path}')
    return sorted(rows, key=lambda row: int(row['index']))


def save_records(path, rows):
    """Atomic snapshot: interrupted writes preserve every previously completed row."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', newline='',
                                         delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: int(row['index'])))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def complete_means(rows, count):
    if len(rows) != count or any(row['status'] != 'complete' for row in rows):
        return None
    return {key: sum(float(row[key]) for row in rows) / count
            for key in ('t_count', 'cx_count', 'synthesis_ms')}


def summarize(name, output_dir=OUTPUT_DIR):
    cases = load_cases(name)
    spec = DATASETS[name]
    lines = [f'{spec.title}: n={spec.n}, {len(cases)} output functions',
             'Full-dataset arithmetic means only; incomplete groups have no reported mean.',
             'Time is synthesis wall time (ms), excluding imports, verification and I/O.',
             'Exorcism time is the mean of its T-cost and CX-cost synthesis calls.', '']
    means = {}
    for method in METHODS:
        rows = read_records(Path(output_dir) / name / method / 'results.csv', name, method, cases)
        counts = Counter(row['status'] for row in rows)
        lines.append(f'{method}: complete={counts["complete"]}/{spec.count}, timeout={counts["timeout"]}, '
                     f'error={counts["error"]}, unrun={spec.count - len(rows)}')
        if rows:
            lines.append(f'  Per-function synthesis limit: {rows[0]["timeout_s"] or "none"} s')
        mean = complete_means(rows, spec.count)
        if mean is not None:
            means[method] = mean
            lines.append('  ' + ', '.join(f'mean {key}={value:.6f}' for key, value in mean.items()))
    for method in ('F', 'FS', 'FP'):
        if method not in means:
            continue
        for baseline in ('exorcism', 'sshr'):
            if baseline not in means:
                continue
            for metric in ('t_count', 'cx_count'):
                base = means[baseline][metric]
                reduction = f'{100 * (1 - means[method][metric] / base):.3f}%' if base else 'undefined (zero baseline)'
                lines.append(f'{method} vs {baseline}: {metric} reduction={reduction}')
    return '\n'.join(lines) + '\n'


def run(args):
    name, method = args.dataset, args.method
    cases = load_cases(name)
    stop = len(cases) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(cases):
        raise ValueError(f'Require 0 <= start < stop <= {len(cases)}')
    timeout = args.timeout
    if timeout is None and method == 'sshr':
        timeout = 1200.0
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('--timeout must be a finite positive number')
    limit = '' if timeout is None else f'{timeout:g}'
    directory = args.output_dir / name / method
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / '.running'
    # Exclusive creation prevents concurrent writers. A hard-killed parent leaves
    # a visible lock: confirm no old process is running before manually removing it.
    with lock_path.open('x') as handle:
        handle.write(f'pid={os.getpid()}\n')
    try:
        path = directory / 'results.csv'
        if path.exists() and not args.resume:
            raise FileExistsError(f'{path} exists; use --resume or a new --output-dir')
        rows = read_records(path, name, method, cases)
        if rows and rows[0]['timeout_s'] != limit:
            raise ValueError('Timeout changed; use a new output directory')
        info = {key: rows[0][key] for key in ('implementation', 'environment')} if rows else None
        existing = {int(row['index']) for row in rows}
        options = dict(exorcism_dir=str(args.exorcism_dir.resolve()),
                       sshr_root=str(args.sshr_root.resolve()) if args.sshr_root else None)
        mp.set_executable(sys.executable)
        for index in range(args.start, stop):
            if index in existing:
                continue  # Preserve timeout/error records too; reruns go in a new directory.
            print(f'{name}/{method}: output {index + 1}/{len(cases)}', flush=True)
            result, info = run_case(method, cases[index], DATASETS[name].n, options, timeout, info)
            row = {key: '' for key in FIELDS}
            row.update(dataset=name, n=DATASETS[name].n, index=index, truth_table=cases[index], method=method,
                       protocol=PROTOCOL, timeout_s=limit, finished_utc=datetime.now(timezone.utc).isoformat())
            row.update(info)
            row.update(result)
            rows.append(row)
            save_records(path, rows)
            print(f'  {result["status"]}' + (f', {result["synthesis_ms"]:.3f} ms' if result['status'] == 'complete'
                                            else f': {result["error"]}'), flush=True)
            if result['status'] == 'error':
                raise RuntimeError('Case error saved; investigate before continuing with --resume')
        report = summarize(name, args.output_dir)
        (directory / 'summary.txt').write_text(report, encoding='utf-8', newline='\n')
        print(report)
    finally:
        lock_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    runner = commands.add_parser('run', help='Run one method sequentially on one benchmark')
    runner.add_argument('dataset', choices=DATASETS)
    runner.add_argument('method', choices=METHODS)
    runner.add_argument('--start', type=int, default=0, help='Zero-based output index (inclusive)')
    runner.add_argument('--stop', type=int, help='Exclusive output index; default is all outputs')
    runner.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    runner.add_argument('--resume', action='store_true', help='Skip all previously recorded cases, including failures')
    runner.add_argument('--timeout', type=float, help='Synthesis seconds per case; SSHR default 1200, others unlimited')
    runner.add_argument('--exorcism-dir', type=Path, default=ROOT / 'data')
    runner.add_argument('--sshr-root', type=Path)
    summary = commands.add_parser('summarize', help='Recompute read-only, aligned full-dataset means')
    summary.add_argument('dataset', choices=DATASETS)
    summary.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.command == 'run':
        run(args)
    else:
        print(summarize(args.dataset, args.output_dir))


if __name__ == '__main__':
    main()
