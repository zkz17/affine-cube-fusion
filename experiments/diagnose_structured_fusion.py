"""Cost-only stage diagnostics; never replace the formal synthesis timings.

Capture the production search winner before and after M2, verify every captured
pool and emitted circuit, and split CNOT cost into MCX, target-CX and linear CX.
PK/PKS are diagnostic alternate frontends, not additional paper measurements.
"""
import argparse
from collections import Counter
import csv
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.fusion.BranchFusion import anf_pool, shift_fn, fuse_pool_global, pool_t_cost
from backend.fusion.CoverGenerators import pk_entries
from config import config
from evaluate_structured import cost_and_verify, read_records
from experiments.generate_structured_cases import DATASETS, generate, load_cases
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit

synthesis = importlib.import_module('backend.OracleSynthesizer')


def pool_stats(kept, standalone, tt, n, shift):
    entries = list(kept) + [(cb, 0, None) for cb in standalone]
    function = 0
    for cb, root, _ in entries:
        function ^= shift_fn(cb.fn(), root, n) if root else cb.fn()
    assert function == shift_fn(tt, shift, n), 'Captured pool changed the function'
    cubes = [cb for cb, _, _ in entries]
    histogram = Counter(cb.nctrl() for cb in cubes)
    return dict(cubes=len(cubes), t=sum(cb.t_cost() for cb in cubes),
                ordinary_cubes=sum(all(free == 0 for _, free, _ in cb.affine_rows()) for cb in cubes),
                controls=dict(sorted(histogram.items())))


def circuit_stats(circuit, tt, n):
    qc = to_qiskit(circuit)
    t, cx = cost_and_verify(qc, tt, n)
    parts = dict(t=t, cx=cx, mcx_cx=0, target_cx=0, linear_cx=0)
    for inst in qc.data:
        k = len(inst.qubits) - 1
        if k >= 2:
            parts['mcx_cx'] += config['cx_cost'](k)
        elif k == 1:
            target = qc.find_bit(inst.qubits[-1]).index
            parts['target_cx' if target == n else 'linear_cx'] += 1
    assert cx == sum(parts[key] for key in ('mcx_cx', 'target_cx', 'linear_cx'))
    return parts


class Probe(synthesis.OracleSynthesizer):
    def _emit_pool(self, kept, standalone, shift_c=0, cnots=(), reorder=True):
        self.winner = (list(kept), list(standalone), shift_c, cnots)
        return super()._emit_pool(kept, standalone, shift_c, cnots, reorder)


def emit(kept, standalone, shift, n):
    engine = synthesis.OracleSynthesizer(ancilla_num=0)
    engine.clear()
    engine.allocate_input_qubit(n)
    engine._emit_pool(kept, standalone, shift)
    return engine.circ


def diagnose(tt, n, route):
    captured = []
    original = synthesis.m2_interleaved

    def macro(kept, standalone, size):
        captured.append((list(kept), list(standalone)))
        return original(kept, standalone, size)

    engine = Probe(ancilla_num=0, fused_backend=True, m2_post=True,
                   paths=('F',) if route == 'F' else ('FS',), fprm_proxy=route == 'FP')
    truth_table = TruthTable(tt, n)
    with patch.object(synthesis, 'm2_interleaved', macro):
        if route in ('PK', 'PKS'):
            engine.clear()
            engine.allocate_input_qubit(n)
            engine.process_cover_route(truth_table, route)
        else:
            engine.synthesize(truth_table)
    assert len(captured) == 1
    kept, standalone, shift, cnots = engine.winner
    assert not cnots
    pair_kept, pair_standalone = captured[0]
    generator = pk_entries if route in ('PK', 'PKS') else anf_pool
    raw = generator(shift_fn(tt, shift, n), n)
    stats = dict(shift=shift,
                 raw=pool_stats(raw, [], tt, n, shift),
                 pairwise=pool_stats(pair_kept, pair_standalone, tt, n, shift),
                 full=pool_stats(kept, standalone, tt, n, shift),
                 pairwise_circuit=circuit_stats(emit(pair_kept, pair_standalone, shift, n), tt, n),
                 full_circuit=circuit_stats(engine.circ, tt, n))
    assert stats['pairwise']['t'] == stats['pairwise_circuit']['t']
    assert stats['full']['t'] == stats['full_circuit']['t']
    return stats


def exhaustive_shifts(tt, n, index, name, output_dir):
    """Exhaust all shifts under the same pre-M2 objective; smallest shift wins ties."""
    best = None
    with (output_dir / f'shift_scores_{index}.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['shift', 'pairwise_t'])
        for shift in range(1 << n):
            kept, standalone = fuse_pool_global(anf_pool(shift_fn(tt, shift, n), n), n)
            cost = pool_t_cost(kept, standalone)
            writer.writerow([shift, cost])
            if best is None or cost < best[0]:
                best = (cost, shift, kept, standalone)
            if (shift + 1) % 128 == 0:
                handle.flush()
                print(f'{name}: checked {shift + 1}/{1 << n} shifts; best pairwise T={best[0]}', flush=True)
    cost, shift, kept, standalone = best
    pairwise = pool_stats(kept, standalone, tt, n, shift)
    kept, standalone = synthesis.m2_interleaved(kept, standalone, n)
    return dict(index=index, output=name, shifts_checked=1 << n, shift=shift,
                pairwise=pairwise, full=pool_stats(kept, standalone, tt, n, shift),
                full_circuit=circuit_stats(emit(kept, standalone, shift, n), tt, n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--routes', nargs='+', choices=('F', 'FS', 'FP', 'PK', 'PKS'),
                        default=['F', 'FS', 'FP', 'PK', 'PKS'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--exhaustive-indices', nargs='+', type=int,
                        help='Instead of stage tracing, exhaust all shifts for these zero-based outputs')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = load_cases(args.dataset)
    names = generate(args.dataset)[1]
    n = DATASETS[args.dataset].n
    if args.exhaustive_indices is not None:
        if any(not 0 <= i < len(cases) for i in args.exhaustive_indices):
            raise ValueError('Output index outside the dataset')
        results = []
        for index in args.exhaustive_indices:
            results.append(exhaustive_shifts(cases[index], n, index, names[index], args.output_dir))
            (args.output_dir / 'exhaustive_winners.json').write_text(json.dumps(results, indent=2) + '\n')
        return
    formal = {}
    for route in args.routes:
        if route in ('F', 'FS', 'FP'):
            path = ROOT / 'data/structured_cost' / args.dataset / route / 'results.csv'
            formal[route] = {int(r['index']): r for r in read_records(path, args.dataset, route, cases)}
    details, rows = [], []
    for index, tt in enumerate(cases):
        for route in args.routes:
            stats = diagnose(tt, n, route)
            recorded = formal.get(route, {}).get(index)
            if recorded:
                assert recorded['status'] == 'complete'
                assert (stats['full_circuit']['t'], stats['full_circuit']['cx']) == (
                    int(recorded['t_count']), int(recorded['cx_count'])), 'Formal cost mismatch'
            detail = dict(dataset=args.dataset, index=index, output=names[index], route=route, **stats)
            details.append(detail)
            row = dict(dataset=args.dataset, index=index, output=names[index], route=route, shift=stats['shift'])
            for stage in ('raw', 'pairwise', 'full'):
                row.update({f'{stage}_{k}': v for k, v in stats[stage].items() if k != 'controls'})
            for stage in ('pairwise_circuit', 'full_circuit'):
                row.update({f'{stage}_{k}': v for k, v in stats[stage].items()})
            rows.append(row)
            (args.output_dir / 'stages.json').write_text(json.dumps(details, indent=2) + '\n')
            with (args.output_dir / 'stages.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f'{names[index]} {route}: shift={stats["shift"]} '
                  f'T {stats["raw"]["t"]} -> {stats["pairwise"]["t"]} -> {stats["full"]["t"]}; '
                  f'CX {stats["full_circuit"]}', flush=True)


if __name__ == '__main__':
    main()
