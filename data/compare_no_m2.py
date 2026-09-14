#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the no-macro-M2 runs (data/our_cost/no_M2) against the default M2-enabled runs
(data/our_cost/t_cost and data/our_cost/cx_cost) on a per-case basis, and emit the
lower/higher/same counts, their percentages, and the total cost in the same style as
evaluate.py's per-case comparison block.

The no_M2 files are keyed by truth table (`tt:cost`); the default files cover a superset
of cases (e.g. n=4 default is the exhaustive 65536-case file while no_M2 holds 1000 cases),
so matching is done by the `tt` key, not by line index.

Output: data/our_cost/no_M2_vs_default.txt (also printed to stdout).
"""
import os
import re
from glob import glob

DATA = os.path.dirname(os.path.abspath(__file__))
NO_M2 = os.path.join(DATA, 'our_cost', 'no_M2')
DEF_T = os.path.join(DATA, 'our_cost', 't_cost')
DEF_CX = os.path.join(DATA, 'our_cost', 'cx_cost')
OUT = os.path.join(DATA, 'our_cost', 'no_M2_vs_default.txt')

NAME_RE = re.compile(r'our_t_cost_(\w+)_noM2_(\d+)_\d+_\d+\.txt')


def load(path):
    """Read a `tt:cost` file, preserving order, returning [(tt, cost), ...]."""
    pairs = []
    for line in open(path):
        tt, cost = line.split(':')
        pairs.append((int(tt), int(cost)))
    return pairs


def load_default_map(pattern):
    """Load the default file for the given glob pattern as {tt: cost}.

    There is exactly one default t_cost/cx_cost file per (tag, n); the no_M2 run covers a
    prefix of it, so we pick the single match and index it by tt.
    """
    matches = glob(pattern)
    assert matches, f'no default file for {pattern}'
    return dict(load(matches[0]))


def compare(name, base_pairs, ours_map):
    """Per-case comparison; returns the report block in evaluate.py's style."""
    better = worse = same = 0
    worse_cases = []
    s_base = s_ours = 0
    for tt, c_base in base_pairs:
        assert tt in ours_map, f'tt {tt} missing from default file for {name}'
        c_ours = ours_map[tt]
        s_base += c_base
        s_ours += c_ours
        if c_ours > c_base:
            worse += 1
            worse_cases.append(tt)
        elif c_ours == c_base:
            same += 1
        else:
            better += 1
    n = len(base_pairs)

    def pct(k):
        return f'{k / n * 100:.1f}%'

    return [
        f'===== {name} =====',
        f'lower  cost in {better}({pct(better)}) cases',
        f'higher cost in {worse}({pct(worse)}) cases',
        f'same   cost in {same}({pct(same)}) cases',
        f'cases with higher cost: {worse_cases[:20]}',
        f'total cost: baseline {s_base}, ours {s_ours} '
        f'(reduction {1 - s_ours / max(1, s_base):.2%})',
    ]


def main():
    lines = ['****** M2 (default) vs no-M2 per-case comparison ******', '']
    for no_t in sorted(glob(os.path.join(NO_M2, 'our_t_cost_*_noM2_*.txt'))):
        m = NAME_RE.match(os.path.basename(no_t))
        tag, n = m.group(1), m.group(2)
        base_t = load(no_t)
        ours_t = load_default_map(os.path.join(DEF_T, f'our_t_cost_{tag}_{n}_*.txt'))

        no_cx_glob = glob(os.path.join(NO_M2, f'our_cx_cost_{tag}_noM2_{n}_*.txt'))
        assert no_cx_glob, f'no no_M2 CX file for {tag} n={n}'
        base_cx = load(no_cx_glob[0])
        ours_cx = load_default_map(os.path.join(DEF_CX, f'our_cx_cost_{tag}_{n}_*.txt'))

        lines.append(f'****** {tag} n={n} ({len(base_t)} cases) ******')
        lines += compare(f'{tag} n={n} T', base_t, ours_t)
        lines += compare(f'{tag} n={n} CX', base_cx, ours_cx)
        lines.append('')

    text = '\n'.join(lines) + '\n'
    print(text)
    with open(OUT, 'w') as f:
        f.write(text)
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
