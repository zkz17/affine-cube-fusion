# -*- coding: utf-8 -*-
"""Paired F-route timing with/without M2; production algorithms are unchanged.

Default: n=6/9/11, indices 0,50,...,950 (two cases per original sampling
stratum), three repeats, balanced off/on and on/off order in every repeat.
Each run performs a fresh, complete synthesis; no winning pool or shift-search
cache is shared. One held-out case warms up both modes before each size.

Timing wrappers call the real production methods. Search includes cover
generation and candidate fusion. M2 is split into re-fusion and remaining
macro search/rewrite/bookkeeping time. Emission includes segment reordering
and simplification. Allocation/reset and wrapper overhead remain in 'other'.
Verification, Qiskit cost conversion, GC collection before a run, and I/O are
outside timing. Normal Python GC remains enabled during synthesis.

Writes only phase0_results/: a human-readable summary and a tab-separated TXT
run log, flushed after every verified run (repeat=0 denotes excluded warmup).
Example: ./venv/bin/python experiments/m2_paired_timing_probe.py
Smoke:   ... --sizes 6 --cases 2 --repeats 2 --prefix m2_paired_smoke
Checks:  ... --self-test
"""
import argparse
import csv
import gc
import hashlib
import importlib
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import FUSION_DEBUG, pool_t_cost
from evaluate import count_costs
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit

ORACLE = importlib.import_module('backend.OracleSynthesizer')
MACRO = importlib.import_module('backend.fusion.MacroM2')
OUT = ROOT / 'experiments' / 'phase0_results'
FIELDS = ('n case_index repeat position m2 wall_ms cpu_ms search_ms macro_ms '
          'refusion_ms m2_total_ms emit_ms other_ms shift blocks_before '
          'blocks_after moves pool_T_before T CX verified tt').split()


class TimedSynthesizer(OracleSynthesizer):
    def __init__(self, m2):
        super().__init__(ancilla_num=0, fused_backend=True, paths=('F',), m2_post=m2)
        self.timing = dict(search_ns=0, emit_ns=0, m2_ns=0, refusion_ns=0, moves=0)
        self.initial_pool = self.final_pool = None
        self.shift = None

    def _fprm_entries(self, tt, n):
        start = time.perf_counter_ns()
        result = super()._fprm_entries(tt, n)
        self.timing['search_ns'] = time.perf_counter_ns() - start
        self.initial_pool = result[:2]
        self.shift = result[2]
        return result

    def _emit_pool(self, kept, standalone, shift_c=0, cnots=(), reorder=True):
        self.final_pool = (kept, standalone)
        start = time.perf_counter_ns()
        super()._emit_pool(kept, standalone, shift_c, cnots, reorder)
        self.timing['emit_ns'] = time.perf_counter_ns() - start


def run_one(n, fn, m2):
    """Measure a real synthesize call, then verify and count gates outside timing."""
    s = TimedSynthesizer(m2)
    tt = TruthTable(fn, n)
    original_m2, original_fuse = ORACLE.m2_interleaved, MACRO.fuse_pool_global

    def timed_m2(*args, **kwargs):
        start = time.perf_counter_ns()
        result = original_m2(*args, **kwargs)
        s.timing['m2_ns'] += time.perf_counter_ns() - start
        return result

    def timed_refusion(*args, **kwargs):
        start = time.perf_counter_ns()
        result = original_fuse(*args, **kwargs)
        s.timing['refusion_ns'] += time.perf_counter_ns() - start
        s.timing['moves'] += 1
        return result

    gc.collect()
    with patch.object(ORACLE, 'm2_interleaved', timed_m2), \
            patch.object(MACRO, 'fuse_pool_global', timed_refusion):
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        s.synthesize(tt)
        wall_ns = time.perf_counter_ns() - start
        cpu_ns = time.process_time_ns() - cpu_start

    assert s.circ.verify(tt), f'wrong circuit: n={n}, m2={m2}, tt={fn}'
    t, cx = count_costs(to_qiskit(s.circ))
    before = pool_t_cost(*s.initial_pool)
    assert t == pool_t_cost(*s.final_pool) <= before
    times = s.timing
    other_ns = wall_ns - times['search_ns'] - times['m2_ns'] - times['emit_ns']
    assert other_ns >= 0 and times['m2_ns'] >= times['refusion_ns']
    return dict(n=n, m2=int(m2), wall_ms=wall_ns / 1e6, cpu_ms=cpu_ns / 1e6,
                search_ms=times['search_ns'] / 1e6,
                macro_ms=(times['m2_ns'] - times['refusion_ns']) / 1e6,
                refusion_ms=times['refusion_ns'] / 1e6, m2_total_ms=times['m2_ns'] / 1e6,
                emit_ms=times['emit_ns'] / 1e6, other_ms=other_ns / 1e6,
                shift=s.shift, blocks_before=sum(map(len, s.initial_pool)),
                blocks_after=sum(map(len, s.final_pool)), moves=times['moves'],
                pool_T_before=before, T=t, CX=cx, verified=1, tt=fn)


def indices_for(cases):
    return [i * 1000 // cases for i in range(cases)]


def schedule(indices, repeat):
    # Rotate case order between repeats; balance both run orders in each repeat.
    offset = (repeat - 1) * 7 % len(indices)
    ranks = {index: rank for rank, index in enumerate(indices)}
    for index in indices[offset:] + indices[:offset]:
        # Base order on the original rank: rotation must not cancel the per-case flip.
        yield index, (False, True) if (ranks[index] + repeat) % 2 else (True, False)


def expected_costs(n, functions):
    result = {}
    for mode in (False, True):
        tag = 'F' if mode else 'F_noM2'
        metrics = []
        for metric in ('t', 'cx'):
            matches = []
            for path in (ROOT / 'data/our_cost/F').glob(f'our_{metric}_cost_{tag}_{n}_0_*.txt'):
                if int(path.stem.rsplit('_', 1)[1]) >= 1000:
                    matches.append(path)
            assert matches, f'no saved costs: {tag} n={n} {metric}'
            path = min(matches, key=lambda p: int(p.stem.rsplit('_', 1)[1]))
            pairs = [tuple(map(int, line.split(':')))
                     for line in path.read_text().splitlines()[:1000]]
            assert [tt for tt, _ in pairs] == functions[:1000], path
            metrics.append([cost for _, cost in pairs])
        result[mode] = list(zip(*metrics))
    return result


def paired_rows(rows, n):
    grouped = {}
    for row in rows:
        if row['n'] == n and row['repeat'] > 0:
            grouped.setdefault((row['repeat'], row['case_index']), {})[row['m2']] = row
    pairs = []
    for modes in grouped.values():
        if len(modes) != 2:
            continue
        off, on = modes[0], modes[1]
        assert all(off[k] == on[k] for k in ('tt', 'shift', 'blocks_before', 'pool_T_before'))
        assert on['T'] <= off['T']
        pairs.append((off, on))
    return pairs


def ratio_delta(pairs, field='wall_ms'):
    return sum(on[field] for off, on in pairs) / sum(off[field] for off, on in pairs) - 1


def summarize(rows, n):
    pairs = paired_rows(rows, n)
    if not pairs:
        return [f'n={n}: no complete measured pairs yet.']
    mean = statistics.mean
    by_case = {}
    for off, on in pairs:
        by_case.setdefault(off['case_index'], []).append((off, on))
    lines = [f'====== n={n}: {len(by_case)} functions, {len(pairs)} measured pairs ======',
             'Means below give equal weight to each completed measurement; warmups excluded.',
             'stage                         off ms        on ms       delta ms']
    for field in ('wall_ms', 'cpu_ms', 'search_ms', 'macro_ms', 'refusion_ms',
                  'm2_total_ms', 'emit_ms', 'other_ms'):
        a, b = (mean(p[mode][field] for p in pairs) for mode in (0, 1))
        lines.append(f'{field:<24} {a:13.6f} {b:13.6f} {b-a:+14.6f}')
    lines += [f'aggregate wall change (on/off - 1): {ratio_delta(pairs):+.4%}',
              f'aggregate CPU change (on/off - 1): {ratio_delta(pairs, "cpu_ms"):+.4%}',
              f'M2 + emission delta: {mean(b["m2_total_ms"] + b["emit_ms"] - a["emit_ms"] for a,b in pairs):+.6f} ms/case',
              f'on faster in {sum(b["wall_ms"] < a["wall_ms"] for a,b in pairs)}/{len(pairs)} measurement pairs',
              f'mean blocks before/after: {mean(a["blocks_before"] for a,b in pairs):.2f} / {mean(b["blocks_after"] for a,b in pairs):.2f}',
              f'mean accepted M2 moves: {mean(b["moves"] for a,b in pairs):.2f}']
    for repeat in sorted({a['repeat'] for a, b in pairs}):
        group = [(a, b) for a, b in pairs if a['repeat'] == repeat]
        lines.append(f'repeat {repeat}: {len(group)} pairs, off={mean(a["wall_ms"] for a,b in group):.6f} ms, '
                     f'on={mean(b["wall_ms"] for a,b in group):.6f} ms, delta={ratio_delta(group):+.4%}')
    for first in (0, 1):
        group = [(a, b) for a, b in pairs if (a if first == 0 else b)['position'] == 1]
        if group:
            lines.append(f'{"off" if first == 0 else "on"}-first: {len(group)} pairs, wall delta={ratio_delta(group):+.4%}')
    for mode in (0, 1):
        cvs = [statistics.stdev(p[mode]['wall_ms'] for p in group) /
               mean(p[mode]['wall_ms'] for p in group) for group in by_case.values() if len(group) > 1]
        if cvs:
            lines.append(f'median within-function repeat CV, {"on" if mode else "off"}: {statistics.median(cvs):.4%}')
    unique = [group[0] for group in by_case.values()]
    for metric in ('T', 'CX'):
        before, after = (sum(p[mode][metric] for p in unique) for mode in (0, 1))
        reduction = f'{1-after/before:.4%}' if before else 'n/a'
        lines.append(f'{metric} (functions counted once): {before} -> {after}, reduction={reduction}; '
                     f'better/same/worse={sum(b[metric]<a[metric] for a,b in unique)}/'
                     f'{sum(b[metric]==a[metric] for a,b in unique)}/'
                     f'{sum(b[metric]>a[metric] for a,b in unique)}')
    return lines


def source_hash():
    paths = [ROOT / 'config.py', ROOT / 'evaluate.py', Path(__file__).resolve()]
    for folder in ('backend', 'circuit', 'limbdd', 'utils'):
        paths.extend((ROOT / folder).rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def self_test():
    indices = indices_for(20)
    assert len(set(indices)) == 20
    assert [sum(i // 100 == j for i in indices) for j in range(10)] == [2] * 10
    for repeat in (1, 2, 3):
        assert sum(order[0] for _, order in schedule(indices, repeat)) == 10
    orders = [dict(schedule(indices, repeat)) for repeat in (1, 2, 3)]
    assert all(orders[0][i] == orders[1][i][::-1] == orders[2][i] for i in indices)
    # Use a saved F case with a real M2 move, plus zero/constant/inactive-input cases.
    base = ROOT / 'data/our_cost/F'
    off = [tuple(map(int, l.split(':'))) for l in (base / 'our_t_cost_F_noM2_4_0_1000.txt').read_text().splitlines()]
    on = [tuple(map(int, l.split(':'))) for l in (base / 'our_t_cost_F_4_0_65536.txt').read_text().splitlines()[:1000]]
    fn_move = next(a[0] for a, b in zip(off, on) if a[1] > b[1])
    saw_move = False
    original = (ORACLE.m2_interleaved, MACRO.fuse_pool_global)
    for fn in (0, 65535, 64560, fn_move):
        for m2 in (False, True):
            measured = run_one(4, fn, m2)
            ref = OracleSynthesizer(ancilla_num=0, m2_post=m2)
            ref.synthesize(TruthTable(fn, 4))
            assert ref.circ.verify(TruthTable(fn, 4))
            assert (measured['T'], measured['CX']) == count_costs(to_qiskit(ref.circ))
            assert measured['m2_total_ms'] == 0 if not m2 else measured['m2_total_ms'] > 0
            saw_move |= measured['moves'] > 0
    assert saw_move and original == (ORACLE.m2_interleaved, MACRO.fuse_pool_global)
    assert ratio_delta([({'wall_ms': 10}, {'wall_ms': 8})]) < 0
    print('Self-check passed: sampling/order balance, wrappers restored, real M2 moves, '
          'unmodified production costs, timing accounting, negative deltas retained.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[6, 9, 11])
    parser.add_argument('--cases', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--prefix', default='m2_paired_timing_probe')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    assert not FUSION_DEBUG, 'timing requires LIMBDD_FUSION_DEBUG=0'
    assert 1 <= args.cases < 1000 and args.repeats >= 1
    assert len(set(args.sizes)) == len(args.sizes) and all(5 <= n <= 11 for n in args.sizes)
    assert Path(args.prefix).name == args.prefix, 'prefix must be a filename, not a path'
    indices = indices_for(args.cases)
    warmup_index = next(i for i in range(1000) if i not in indices)
    functions = {n: [int(l) for l in (ROOT / f'data/test_cases/test_cases_{n}').read_text().splitlines()]
                 for n in args.sizes}
    expected = {n: expected_costs(n, functions[n]) for n in args.sizes}
    for n in args.sizes:
        interval = (1 << (1 << n)) // 10
        assert all(min(functions[n][i] // interval, 9) == i // 100 for i in indices)
    OUT.mkdir(parents=True, exist_ok=True)
    report_path, raw_path = OUT / f'{args.prefix}.txt', OUT / f'{args.prefix}_runs.txt'
    assert not report_path.exists() and not raw_path.exists(), 'output exists; use a different --prefix'
    signature = source_hash()
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next(l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                   if l.startswith('model name'))
    header = ['M2 paired timing probe: F, zero ancillas, production search/emission',
              f'started UTC: {datetime.now(timezone.utc).isoformat()}',
              f'git revision: {subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}',
              f'source SHA256 (including probe): {signature}',
              f'platform: {platform.platform()}; CPU: {cpu}; Python: {platform.python_version()}',
              'packages: ' + ', '.join(f'{name}={version(name)}' for name in ('numpy', 'qiskit', 'scipy')),
              f'sizes={args.sizes}; cases/size={args.cases}; repeats={args.repeats}; indices={indices}',
              f'counts in 10 original strata: {[sum(i//100==j for i in indices) for j in range(10)]}',
              f'warmup index={warmup_index}, one run/mode/size; repeat=0 excluded from all summaries',
              'fresh full synthesis per run; alternating balanced order; case order rotated between repeats',
              'normal GC enabled; explicit pre-run GC, verification, Qiskit conversion and I/O untimed',
              'times: wall/search/M2/refusion/emission; CPU is whole-process CPU time',
              'search includes initial cover and candidate fusion; macro excludes timed re-fusion',
              'wall = search + macro + refusion + emission + other; m2_total = macro + refusion',
              'cost totals count each function once; timings average all completed repeats',
              'raw run log: ' + raw_path.name]
    for n in args.sizes:
        path = ROOT / f'data/test_cases/test_cases_{n}'
        header.append(f'input n={n} SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}')
    rows = []

    def save_report(status):
        lines = header + [f'status: {status}', f'updated UTC: {datetime.now(timezone.utc).isoformat()}', '']
        for n in args.sizes:
            lines += summarize(rows, n) + ['']
        report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    save_report('RUNNING')
    try:
        with raw_path.open('x', encoding='utf-8', newline='') as raw:
            writer = csv.DictWriter(raw, fieldnames=FIELDS, delimiter='\t')
            writer.writeheader()
            raw.flush()
            for n in args.sizes:
                print(f'n={n}: warmup (excluded), then {args.cases * args.repeats} pairs', flush=True)
                for repeat in range(args.repeats + 1):
                    batches = [(warmup_index, (False, True))] if repeat == 0 else schedule(indices, repeat)
                    for index, order in batches:
                        pair = {}
                        for position, mode in enumerate(order, 1):
                            row = run_one(n, functions[n][index], mode)
                            assert (row['T'], row['CX']) == expected[n][mode][index], \
                                f'saved cost mismatch: n={n}, index={index}, m2={mode}'
                            row.update(case_index=index, repeat=repeat, position=position)
                            writer.writerow(row)
                            raw.flush()
                            rows.append(row)
                            pair[int(mode)] = row
                        a, b = pair[0], pair[1]
                        print(f'n={n} repeat={repeat} index={index:3d} '
                              f'{"on/off" if order[0] else "off/on"}: '
                              f'off={a["wall_ms"]:.2f} on={b["wall_ms"]:.2f} ms '
                              f'({b["wall_ms"]/a["wall_ms"]-1:+.2%}); '
                              f'M2={b["m2_total_ms"]:.2f}, emit delta={b["emit_ms"]-a["emit_ms"]:+.2f} ms; '
                              f'T={a["T"]}->{b["T"]}', flush=True)
                        save_report('RUNNING')
                    if repeat:
                        print('\n'.join(summarize(rows, n)), flush=True)
                assert source_hash() == signature, 'production source changed during the experiment'
        save_report('COMPLETE; all circuits verified and all T/CX costs match saved results')
    except BaseException as exc:
        save_report(f'INCOMPLETE: {type(exc).__name__}: {exc}')
        raise
    print(f'Completed. Summary: {report_path}\nRaw runs: {raw_path}', flush=True)


if __name__ == '__main__':
    main()
