"""Read-only analysis of completed structured experiments; never runs synthesis.

Creates a timestamped snapshot of aggregate and aligned per-output comparisons.
Only complete datasets receive means; timed-out groups retain explicit status
counts without numeric cost or runtime averages.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_structured import METHODS, OUTPUT_DIR, complete_means, read_records
from experiments.generate_structured_cases import DATASETS, DATA_DIR, load_cases

METRICS = ('t_count', 'cx_count', 'synthesis_ms')
LABELS = {'des': 'DES', 'aes': 'AES', 'ascon': 'Ascon', 'speedy': 'SPEEDY'}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison(name, method, baseline, ours, theirs, scope):
    if set(ours) != set(theirs):
        raise ValueError('Comparison subsets do not match')
    result = []
    for metric in METRICS:
        values = [(float(ours[i][metric]), float(theirs[i][metric])) for i in sorted(ours)]
        a, b = sum(x for x, _ in values), sum(y for _, y in values)
        result.append(dict(dataset=name, scope=scope, count=len(values), method=method,
                           baseline=baseline, metric=metric, method_mean=a / len(values),
                           baseline_mean=b / len(values), reduction_pct=100 * (1 - a / b) if b else None,
                           method_over_baseline=a / b if b else None,
                           lower=sum(x < y for x, y in values), same=sum(x == y for x, y in values),
                           higher=sum(x > y for x, y in values)))
    return result


def collect(output):
    captured = datetime.now(timezone.utc).isoformat()
    mapping = {}
    with (DATA_DIR / 'case_index.csv').open() as handle:
        for row in csv.DictReader(handle):
            mapping[(row['dataset'], int(row['line_1based']) - 1)] = row['output']
    means, comparisons, per_output, inventory, groups = [], [], [], [], []
    for name, spec in DATASETS.items():
        cases = load_cases(name)
        records, aggregates = {}, {}
        for method in METHODS:
            path = OUTPUT_DIR / name / method / 'results.csv'
            before = path.read_bytes() if path.exists() else None
            rows = read_records(path, name, method, cases)
            after = path.read_bytes() if path.exists() else None
            if before != after:
                raise RuntimeError(f'Source changed during snapshot; rerun analysis: {path}')
            if before is not None:
                inventory.append(f'{path.relative_to(ROOT).as_posix()}  sha256={hashlib.sha256(before).hexdigest()}')
            records[method] = {int(row['index']): row for row in rows}
            counts = Counter(row['status'] for row in rows)
            mean = complete_means(rows, spec.count)
            aggregates[method] = mean
            means.append(dict(dataset=name, n=spec.n, total=spec.count, method=method,
                              complete=counts['complete'], timeout=counts['timeout'], error=counts['error'],
                              unrun=spec.count - len(rows),
                              mean_t_count=mean['t_count'] if mean else None,
                              mean_cx_count=mean['cx_count'] if mean else None,
                              mean_synthesis_ms=mean['synthesis_ms'] if mean else None))
        for method in ('F', 'FS', 'FP'):
            if not aggregates[method]:
                continue
            for baseline in ('exorcism', 'sshr', 'F', 'FS'):
                if baseline != method and aggregates[baseline]:
                    comparisons += comparison(name, method, baseline, records[method], records[baseline], 'full_dataset')
        details = []
        for i in range(spec.count):
            detail = dict(dataset=name, n=spec.n, index=i, output=mapping[(name, i)])
            compact = dict(index=i, output=mapping[(name, i)], methods={})
            for method in METHODS:
                row = records[method].get(i)
                state = row['status'] if row else 'unrun'
                detail[method + '_status'] = state
                compact['methods'][method] = {'status': state}
                for metric in METRICS:
                    value = float(row[metric]) if state == 'complete' else None
                    detail[f'{method}_{metric}'] = value
                    compact['methods'][method][metric] = value
            per_output.append(detail)
            details.append(compact)
        groups.append(dict(name=name, label=LABELS[name], n=spec.n, count=spec.count,
                           methods=[row for row in means if row['dataset'] == name], details=details))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'means.csv', means)
    write_csv(output / 'comparisons.csv', comparisons)
    write_csv(output / 'per_output.csv', per_output)
    (output / 'sources.txt').write_text(
        f'Captured UTC: {captured}\nSource result hashes at capture; sources may continue to grow afterward.\n'
        'Mean: arithmetic mean over all nominal output functions in a dataset.\n'
        'Reduction: 100*(1-total_method/total_baseline); negative values indicate an increase.\n'
        'No full-dataset means for incomplete groups. Timing units: milliseconds.\n'
        + '\n'.join(inventory) + '\n', encoding='utf-8')
    return dict(captured=captured, groups=groups, comparisons=comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=OUTPUT_DIR / 'analysis' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--canvas-data', type=Path, help='Optional scratch JSON for an embedded-data review canvas')
    args = parser.parse_args(argv)
    data = collect(args.output_dir)
    if args.canvas_data:
        args.canvas_data.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    print(args.output_dir)
    print(f'Captured {data["captured"]}; {sum(r["mean_t_count"] is not None for g in data["groups"] for r in g["methods"])} complete dataset/method groups.')


if __name__ == '__main__':
    main()
