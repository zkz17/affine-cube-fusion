# -*- coding: utf-8 -*-
"""
Data summary for the per-path layout of data/our_cost/{path}/.

Each route folder contains our_{t,cx}_cost_{tag}_{n}_{a}_{b}.txt and
our_results_{tag}_{n}_{a}_{b}.txt (tags may carry the _noM2 suffix). Aggregates
ΣT / ΣCX / avg wall / Exorcism reductions per (tag, n); the degree bound is computed
on the fly from the test cases (it is test-set dependent only). For each (tag, n),
show the largest saved batch and its interval; totals across different intervals
must not be compared directly.

Read-only on data/; writes experiments/phase0_results/data_summary.txt.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR = os.path.join(ROOT, 'data', 'our_cost')
BASE = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'data_summary.txt')


def load_vals(path, start, stop):
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return [int(line.split(':')[1]) for line in lines[start:stop]]


def anf_bound(fn, n):
    from backend.fusion.BranchFusion import anf_degree
    from config import config
    return config['t_cost'](anf_degree(fn, n))


def main():
    rows = {}
    for tag in sorted(os.listdir(OUR)):
        d = os.path.join(OUR, tag)
        if not os.path.isdir(d) or tag == 'deg_bound':
            continue
        for f in sorted(os.listdir(d)):
            m = re.match(r'our_t_cost_([A-Za-z0-9_]+)_(\d+)_(\d+)_(\d+)\.txt$', f)
            if not m:
                continue
            ftag, n, a, b = m.groups()
            n, a, b = int(n), int(a), int(b)
            key = (ftag, n, a, b)
            # Run files already contain only [a, b); do not slice them a second time.
            tc = load_vals(os.path.join(d, f), 0, None)
            cx = load_vals(os.path.join(d, f'our_cx_cost_{ftag}_{n}_{a}_{b}.txt'), 0, None)
            assert len(tc) == len(cx) == b - a, f'incomplete cost batch: {f}'
            wall = None
            rp = os.path.join(d, f'our_results_{ftag}_{n}_{a}_{b}.txt')
            if os.path.exists(rp):
                with open(rp, encoding='utf-8', errors='replace') as report:
                    mm = re.search(r'avg synthesize wall: ([\d.]+) ms', report.read())
                if mm:
                    wall = float(mm.group(1))
            rows[key] = dict(sum_t=sum(tc), sum_cx=sum(cx), wall=wall, n=len(tc), start=a, stop=b)

    ns = sorted({k[1] for k in rows})
    tags = sorted({k[0] for k in rows}, key=lambda t: (t.replace('_noM2', '~'), t))
    lines = ['Data summary (per-path layout; largest saved batch per route)',
             'Compare totals only for the same input interval; timings may come from different runs.', '']

    def pick(n):
        d = {}
        for (tag, nn, a, b), r in rows.items():
            if nn != n:
                continue
            if tag not in d or r['n'] > d[tag]['n']:
                d[tag] = r
        return d

    lines.append('=== input interval [start, stop) ===')
    lines.append('  n    ' + ' '.join(f'{t:>12}' for t in tags))
    for n in ns:
        d = pick(n)
        intervals = [f'[{d[t]["start"]},{d[t]["stop"]})' if t in d else '--' for t in tags]
        lines.append(f'  {n:>3}  ' + ' '.join(f'{v:>12}' for v in intervals))
    lines.append('')
    lines.append('=== ΣT ===')
    lines.append('  n    ' + ' '.join(f'{t:>9}' for t in tags))
    for n in ns:
        d = pick(n)
        lines.append(f'  {n:>3}  ' + ' '.join(f'{d[t]["sum_t"]:>9}' if t in d else f'{"--":>9}' for t in tags))
    lines.append('')
    lines.append('=== ΣCX ===')
    lines.append('  n    ' + ' '.join(f'{t:>9}' for t in tags))
    for n in ns:
        d = pick(n)
        lines.append(f'  {n:>3}  ' + ' '.join(f'{d[t]["sum_cx"]:>9}' if t in d else f'{"--":>9}' for t in tags))
    lines.append('')
    lines.append('=== avg wall (ms/case) ===')
    lines.append('  n    ' + ' '.join(f'{t:>9}' for t in tags))
    for n in ns:
        d = pick(n)
        lines.append(f'  {n:>3}  ' + ' '.join(f'{d[t]["wall"]:>9.2f}' if t in d and d[t]['wall'] is not None else f'{"--":>9}' for t in tags))
    lines.append('')
    lines.append('=== Exorcism ΣT reduction (negative = worse) ===')
    lines.append('  n    ' + ' '.join(f'{t:>9}' for t in tags))
    for n in ns:
        ex = os.path.join(BASE, 'exorcism_cost', f'exorcism_t_cost_{n}')
        ex_vals = load_vals(ex, 0, None) if os.path.exists(ex) else None
        d = pick(n)
        out = []
        for t in tags:
            if t not in d or ex_vals is None:
                out.append('--')
                continue
            r = d[t]
            if len(ex_vals) < r['stop']:
                out.append('--')
                continue
            base = sum(ex_vals[r['start']:r['stop']])
            out.append(f'{1 - r["sum_t"] / max(1, base):+.1%}')
        lines.append(f'  {n:>3}  ' + ' '.join(f'{v:>9}' for v in out))
    lines.append('')
    lines.append('=== degree bound Σ t(deg f) (computed on the fly; test-set dependent only) ===')
    for n in ns:
        a = b = None
        for (tag, nn, aa, bb) in rows:
            if nn == n and (b is None or (bb - aa) > (b - a)):
                a, b = aa, bb
        with open(os.path.join(BASE, f'test_cases/test_cases_{n}'), encoding='utf-8') as cases:
            fns = [int(l) for l in cases.readlines()[a:b]]
        s = sum(anf_bound(fn, n) for fn in fns)
        lines.append(f'  n={n} [{a},{b}) ({b - a} cases): Σ t(deg f) = {s}')

    text = '\n'.join(lines) + '\n'
    print(text)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'\nsummary written to {OUT}')


if __name__ == '__main__':
    main()
