# -*- coding: utf-8 -*-
"""
Data summary: a simple summary of the three data/our_cost folders (t_cost/cx_cost/deg_bound + root-level results).
Read-only over the data directory; output is written to experiments/phase0_results/.

Scope:
  - A/C/F/FP n=3..10 (n=3 all 256, n=4 all 65536, n=5-7 10000, n=8-10 1000);
  - G only n=3..6 (n=7 omitted, does not participate for n≥8);
  - baseline Exorcism all n; SSHR only n=3..6 (no data for n≥7, skipped automatically).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'our_cost')
OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'data_summary.txt')
BASE = os.path.join(ROOT, 'data')

TAGS = {'A': list(range(3, 11)), 'C': list(range(3, 11)), 'F': list(range(3, 11)),
        'FP': list(range(3, 11)), 'G': list(range(3, 7))}
NS = {3: 256, 4: 65536, 5: 10000, 6: 10000, 7: 10000, 8: 1000, 9: 1000, 10: 1000}


def load_vals(path):
    out = []
    for line in open(path):
        if ':' in line:
            out.append(int(line.split(':')[1].strip()))
    return out


def parse_results(tag, n, start, stop):
    p = os.path.join(DATA, f'our_results_{tag}_{n}_{start}_{stop}.txt')
    wall = None
    ex = {}          # ('exor','t'|'cx') -> (lower, higher, same, ratio)
    sshr = {}
    bound_ratio = None
    try:
        text = open(p).read()
    except FileNotFoundError:
        return None
    m = re.search(r'avg synthesize wall: ([\d.]+) ms', text)
    if m:
        wall = float(m.group(1))
    for name in ('exorcism', 'sshr'):
        for k in ('t', 'cx'):
            label = 'Exorcism' if name == 'exorcism' else 'SSHR'
            sec = re.search(rf'===== {label} {k.upper()} =====\n'
                            rf'(.*?)(?=====|$)', text, re.S)
            if not sec:
                continue
            body = sec.group(1)
            lm = re.search(r'lower  cost in (\d+)\(', body)
            hm = re.search(r'higher cost in (\d+)\(', body)
            sm = re.search(r'same   cost in (\d+)\(', body)
            rm = re.search(r'缩减 ([-\d.]+)%', body)
            if lm and rm:
                ex[name, k] = (int(lm.group(1)), int(hm.group(1)), int(sm.group(1)),
                               float(rm.group(1)) / 100)
    m = re.search(r'G-3 下界: Σ t\(deg f\) = (\d+) \(为 ours ΣT 的 ([\d.]+)%\)', text)
    if m:
        bound_ratio = float(m.group(2)) / 100
    return dict(wall=wall, ex=ex, bound_ratio=bound_ratio)


def main():
    rows = []
    for tag, ns in TAGS.items():
        for n in ns:
            start, stop = 0, NS[n]
            t_vals = load_vals(os.path.join(DATA, 't_cost', f'our_t_cost_{tag}_{n}_{start}_{stop}.txt'))
            cx_vals = load_vals(os.path.join(DATA, 'cx_cost', f'our_cx_cost_{tag}_{n}_{start}_{stop}.txt'))
            b_vals = load_vals(os.path.join(DATA, 'deg_bound', f'our_deg_bound_{tag}_{n}_{start}_{stop}.txt'))
            res = parse_results(tag, n, start, stop)
            ex_t = os.path.exists(os.path.join(BASE, 'exorcism_cost', f'exorcism_t_cost_{n}'))
            ex_cx = os.path.exists(os.path.join(BASE, 'exorcism_cost', f'exorcism_cx_cost_{n}'))
            sshr_t = os.path.exists(os.path.join(BASE, 'sshr_cost', f'sshr_t_cost_{n}'))
            sshr_cx = os.path.exists(os.path.join(BASE, 'sshr_cost', f'sshr_cx_cost_{n}'))
            rows.append(dict(tag=tag, n=n, cases=len(t_vals), st=sum(t_vals), scx=sum(cx_vals),
                             sb=sum(b_vals), res=res, ex_t=ex_t, ex_cx=ex_cx,
                             sshr_t=sshr_t, sshr_cx=sshr_cx))
    rows.sort(key=lambda r: (r['n'], r['tag']))

    lines = ['Data summary (2025-08-24, fused backend = current production default, m2_post on)', '']
    lines.append('=== ΣT(ours)===')
    lines.append('  n   ' + ' '.join(f'{t:>9}' for t in ('A', 'C', 'F', 'FP', 'G')))
    for n in sorted({r['n'] for r in rows}):
        d = {r['tag']: r['st'] for r in rows if r['n'] == n}
        lines.append(f'  {n:>2}  ' + ' '.join(f'{d.get(t, -1):>9}' for t in ('A', 'C', 'F', 'FP', 'G')))
    lines.append('')
    lines.append('=== ΣCX(ours)===')
    lines.append('  n   ' + ' '.join(f'{t:>9}' for t in ('A', 'C', 'F', 'FP', 'G')))
    for n in sorted({r['n'] for r in rows}):
        d = {r['tag']: r['scx'] for r in rows if r['n'] == n}
        lines.append(f'  {n:>2}  ' + ' '.join(f'{d.get(t, -1):>9}' for t in ('A', 'C', 'F', 'FP', 'G')))
    lines.append('')
    lines.append('=== avg wall (ms/case) ===')
    lines.append('  n   ' + ' '.join(f'{t:>9}' for t in ('A', 'C', 'F', 'FP', 'G')))
    for n in sorted({r['n'] for r in rows}):
        d = {r['tag']: (r['res']['wall'] if r['res'] and r['res']['wall'] is not None else None)
             for r in rows if r['n'] == n}
        lines.append(f'  {n:>2}  ' + ' '.join(f'{d.get(t, -1):>9.1f}' if d.get(t) is not None else '        -'
                                              for t in ('A', 'C', 'F', 'FP', 'G')))
    lines.append('')
    lines.append('=== Exorcism ΣT reduction (negative = higher; higher-case count in parens) ===')
    lines.append('  n   ' + ' '.join(f'{t:>13}' for t in ('A', 'C', 'F', 'FP', 'G')))
    for n in sorted({r['n'] for r in rows}):
        d = {}
        for r in rows:
            if r['n'] != n:
                continue
            e = r['res'] and r['res']['ex'].get((('exorcism', 't')))
            d[r['tag']] = f'{e[3]:+.2%}(h{e[1]})' if e else '--'
        lines.append(f'  {n:>2}  ' + ' '.join(f'{d.get(t, "--"):>13}' for t in ('A', 'C', 'F', 'FP', 'G')))
    lines.append('')
    lines.append('=== Exorcism ΣCX reduction (higher-case count in parens) ===')
    lines.append('  n   ' + ' '.join(f'{t:>13}' for t in ('A', 'C', 'F', 'FP', 'G')))
    for n in sorted({r['n'] for r in rows}):
        d = {}
        for r in rows:
            if r['n'] != n:
                continue
            e = r['res'] and r['res']['ex'].get((('exorcism', 'cx')))
            d[r['tag']] = f'{e[3]:+.2%}(h{e[1]})' if e else '--'
        lines.append(f'  {n:>2}  ' + ' '.join(f'{d.get(t, "--"):>13}' for t in ('A', 'C', 'F', 'FP', 'G')))
    lines.append('')
    lines.append('=== SSHR ΣT / ΣCX reduction (n=3..6) ===')
    lines.append('  n   ' + ' '.join(f'{t:>13}' for t in ('A', 'C', 'F', 'FP', 'G')))
    for n in (3, 4, 5, 6):
        d = {}
        for r in rows:
            if r['n'] != n:
                continue
            e_t = r['res'] and r['res']['ex'].get(('sshr', 't'))
            e_c = r['res'] and r['res']['ex'].get(('sshr', 'cx'))
            d[r['tag']] = f'{e_t[3]:+.1%}/{e_c[3]:+.1%}' if (e_t and e_c) else '--'
        lines.append(f'  {n:>2}  ' + ' '.join(f'{d.get(t, "--"):>13}' for t in ('A', 'C', 'F', 'FP', 'G')))
    lines.append('')
    lines.append('=== G-3 bound share of ours ΣT ===')
    lines.append('  n   ' + ' '.join(f'{t:>9}' for t in ('A', 'C', 'F', 'FP', 'G')))
    for n in sorted({r['n'] for r in rows}):
        d = {r['tag']: r['res']['bound_ratio'] if r['res'] and r['res']['bound_ratio'] is not None else None
             for r in rows if r['n'] == n}
        lines.append(f'  {n:>2}  ' + ' '.join(f'{d.get(t, -1)*100:>8.1f}%' if d.get(t) is not None else '        -'
                                              for t in ('A', 'C', 'F', 'FP', 'G')))
    text = '\n'.join(lines) + '\n'
    print(text)
    with open(OUT, 'w') as f:
        f.write(text)
    print(f'\nSummary written to {OUT}')


if __name__ == '__main__':
    main()
