"""Export saved structured CSV records using the existing cost/report formats.

No synthesis is performed. Exports live beside the source CSV records under
data/structured_cost/<dataset>/, separate from random-function data. Incomplete groups get empty
cost files and explicit status reports, never partial or fabricated averages.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import _compare, anf_degree, T_COST_FN
from evaluate_structured import METHODS, OUTPUT_DIR, complete_means, read_records
from experiments.generate_structured_cases import DATASETS, load_cases


def costs(rows, count, key):
    if complete_means(rows, count) is None:
        return None
    return [int(row[key]) for row in rows]


def cost_text(cases, values):
    return '' if values is None else ''.join(f'{tt}:{value}\n' for tt, value in zip(cases, values))


def time_text(rows):
    values = [float(row['synthesis_ms']) for row in rows]
    return (f'total runtime  : {sum(values)} ms\n'
            f'average runtime: {sum(values) / len(values)} ms\n')


def prepare(name, source_dir, data_dir):
    cases = load_cases(name)
    spec, count = DATASETS[name], len(cases)
    records = {}
    for method in METHODS:
        records[method] = read_records(source_dir / name / method / 'results.csv', name, method, cases)
    outputs = {}
    # ACF exports require the complete requested output set. No silent truncation.
    for route in ('F', 'FS', 'FP'):
        if complete_means(records[route], count) is None:
            raise ValueError(f'{name}/{route}: a full successful dataset is required')
    for method in ('exorcism', 'sshr'):
        rows = records[method]
        directory = data_dir / name / f'{method}_cost'
        for metric, key in (('t', 't_count'), ('cx', 'cx_count')):
            outputs[directory / f'{method}_{metric}_cost'] = cost_text(cases, costs(rows, count, key))
        mean = complete_means(rows, count)
        time_path = directory / f'{method}_time'
        if mean is not None:
            outputs[time_path] = time_text(rows)
        elif time_path.exists():
            raise ValueError(f'Incomplete source has a stale numeric time file: {time_path}')
        counts = Counter(row['status'] for row in rows)
        lines = [f'{method} structured dataset: {name}',
                 f'number of input variables: {spec.n} ({count} cases)',
                 f'complete={counts["complete"]}/{count}, timeout={counts["timeout"]}, '
                 f'error={counts["error"]}, unrun={count-len(rows)}',
                 f'source: data/structured_cost/{name}/{method}/results.csv']
        if rows and rows[0]['timeout_s']:
            lines.append(f'per-function synthesis limit: {rows[0]["timeout_s"]} s')
        if mean is None:
            lines.append('Full-dataset costs and mean synthesis time unavailable; cost files are empty.')
        else:
            lines += [f'mean T cost: {mean["t_count"]}', f'mean CX cost: {mean["cx_count"]}']
        if method == 'exorcism':
            lines.append('Time convention: each recorded time is the mean of the T-cost and CX-cost calls; '
                         'total runtime sums those per-function means.')
        outputs[directory / f'{method}_results.txt'] = '\n'.join(lines) + '\n'
    bound = sum(T_COST_FN(anf_degree(tt, spec.n)) for tt in cases)
    for route in ('F', 'FS', 'FP'):
        rows = records[route]
        directory = data_dir / name / route
        suffix = f'{route}.txt'
        ours_t, ours_cx = (costs(rows, count, key) for key in ('t_count', 'cx_count'))
        outputs[directory / f'our_t_cost_{suffix}'] = cost_text(cases, ours_t)
        outputs[directory / f'our_cx_cost_{suffix}'] = cost_text(cases, ours_cx)
        report = ['****** Comparison of oracle synthesis routes ******',
                  f'number of input variables: {spec.n} ({count} cases); path={route}',
                  f'avg synthesize wall: {complete_means(rows, count)["synthesis_ms"]:.2f} ms/case']
        for method, label in (('exorcism', 'Exorcism'), ('sshr', 'SSHR')):
            report += _compare(f'{label} T', cases, ours_t, costs(records[method], count, 't_count'))
            report += _compare(f'{label} CX', cases, ours_cx, costs(records[method], count, 'cx_count'))
        report += [f'G-3 lower bound: Σ t(deg f) = {bound} ({bound / max(1,sum(ours_t)):.2%} of ours ΣT)',
                   '****** ****** ******', f'structured dataset: {name}; M2=on; synthesis ancillas=0',
                   f'source: data/structured_cost/{name}/{route}/results.csv']
        if complete_means(records['sshr'], count) is None:
            counts = Counter(row['status'] for row in records['sshr'])
            report.append(f'SSHR-H comparison unavailable: complete={counts["complete"]}/{count}, '
                          f'timeout={counts["timeout"]}, error={counts["error"]}, '
                          f'unrun={count-len(records["sshr"])}; see ../sshr_cost/sshr_results.txt.')
        outputs[directory / f'our_results_{suffix}'] = '\n'.join(report) + '\n'
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--source-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--data-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--check', action='store_true', help='Verify existing exports without writing')
    args = parser.parse_args()
    outputs = {}
    for name in args.datasets:
        outputs.update(prepare(name, args.source_dir, args.data_dir))
    # Preflight every target. Existing exports may be reused only if identical.
    for path, content in outputs.items():
        same = path.exists() and path.read_text(encoding='utf-8') == content
        if args.check and not same:
            raise ValueError(f'Missing or stale export: {path}')
        if path.exists() and not same:
            raise FileExistsError(f'Refusing to replace a different export: {path}')
    if not args.check:
        for path, content in outputs.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8', newline='\n') as handle:
                handle.write(content)
    print(f'{"Verified" if args.check else "Exported"} {len(outputs)} files for {", ".join(args.datasets)}; no synthesis performed.')


if __name__ == '__main__':
    main()
