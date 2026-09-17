#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the no-macro-M2 runs against the default M2-enabled runs on a per-case basis,
and emit the lower/higher/same counts, their percentages, and the total cost in the
same style as evaluate.py's per-case comparison block. Both variants live in the same
per-path folder under data/our_cost/{tag}/; the no-M2 files carry the `_noM2` tag suffix.

The no_M2 files are keyed by truth table (`tt:cost`); the default files cover a superset
of cases (e.g. n=4 default is the exhaustive 65536-case file while no_M2 holds 1000 cases),
so matching is done by the `tt` key, not by line index.

Output: data/our_cost/no_M2_vs_default.txt (also printed to stdout).
"""
import os
import re
from glob import glob

DATA = os.path.dirname(os.path.abspath(__file__))
OUR_COST = os.path.join(DATA, 'our_cost')
OUT = os.path.join(OUR_COST, 'no_M2_vs_default.txt')

NAME_RE = re.compile(r'our_t_cost_(\w+)_noM2_(\d+)_(\d+)_(\d+)\.txt')


def load(path):
    """Read a `tt:cost` file, preserving order, returning [(tt, cost), ...]."""
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            tt, cost = line.split(':')
            pairs.append((int(tt), int(cost)))
    return pairs


def load_default_map(pattern):
    """Combine all default batches, rejecting inconsistent overlapping measurements."""
    matches = sorted(glob(pattern))
    assert matches, f'no default file for {pattern}'
    costs = {}
    for path in matches:
        for tt, cost in load(path):
            assert tt not in costs or costs[tt] == cost, f'conflicting default cost in {path}'
            costs[tt] = cost
    return costs


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
    batches = glob(os.path.join(OUR_COST, '*', 'our_t_cost_*_noM2_*.txt'))
    # Sort by path components so Windows and Linux put F before FP consistently.
    for no_t in sorted(batches, key=lambda p: (os.path.basename(os.path.dirname(p)), os.path.basename(p))):
        m = NAME_RE.fullmatch(os.path.basename(no_t))
        assert m, f'unrecognized no-M2 filename: {no_t}'
        tag, n, start, stop = m.groups()
        base_t = load(no_t)
        assert len(base_t) == int(stop) - int(start), f'incomplete T file: {no_t}'
        ours_t = load_default_map(os.path.join(OUR_COST, tag, f'our_t_cost_{tag}_{n}_*.txt'))

        no_cx = os.path.join(OUR_COST, tag, f'our_cx_cost_{tag}_noM2_{n}_{start}_{stop}.txt')
        base_cx = load(no_cx)
        assert [tt for tt, _ in base_t] == [tt for tt, _ in base_cx], f'T/CX mismatch: {no_t}'
        ours_cx = load_default_map(os.path.join(OUR_COST, tag, f'our_cx_cost_{tag}_{n}_*.txt'))

        lines.append(f'****** {tag} n={n} ({len(base_t)} cases) ******')
        lines.append(f'input interval: [{start}, {stop})')
        lines += compare(f'{tag} n={n} T', base_t, ours_t)
        lines += compare(f'{tag} n={n} CX', base_cx, ours_cx)
        lines.append('')

    text = '\n'.join(lines) + '\n'
    print(text)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
