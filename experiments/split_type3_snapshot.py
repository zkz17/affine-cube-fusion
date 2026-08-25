# -*- coding: utf-8 -*-
"""
P5 (diagnostic): exact algebraic snapshot of SplitLv2Pattern type 3 "borrow a 0 edge".

Flow: build a random small function → locate a type-3 candidate → extract the pre-split branch pool → perform the split →
extract the post-split branch pool (original graph + new pieces) → find the segments that exist only before/after, print the function and costs,
verify: identity, cost change, and the "0 = 1⊕1" borrowing structure.

Output: experiments/phase0_results/split_type3_snapshot.txt
"""
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from limbdd.LimBDD import LimBDD
from limbdd.LimBDDManager import LimBDDManager
from limbdd.Extraction import abs_lim, extract_complete_branches
from limbdd.patterns.SplitLv2Pattern import SplitLv2Pattern
from backend.fusion.BranchFusion import anf_degree, shift_fn
from backend.CompleteBranch import t_cost

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'split_type3_snapshot.txt')


def pool_of(limbdd, n):
    """Whole-graph branch pool (effective function: root-mask shift already applied): {(fn, cost): count}."""
    r = abs_lim(limbdd.root.lim, limbdd.var_order)
    c = Counter()
    for cb in extract_complete_branches(limbdd, n, anomalies=Counter()):
        fn = shift_fn(cb.fn(), r, n)
        c[(fn, cb.t_cost(), cb.nctrl())] += 1
    return c


def pool_of_pieces(pieces, n):
    c = Counter()
    for g in pieces:
        r = abs_lim(g.root.lim, g.var_order)
        for cb in extract_complete_branches(g, n, anomalies=Counter()):
            fn = shift_fn(cb.fn(), r, n)
            c[(fn, cb.t_cost(), cb.nctrl())] += 1
    return c


def find_type3(mgr):
    """Return (node, level) or None."""
    pat = SplitLv2Pattern()
    for level in range(2, mgr.limbdd.root.level + 1):
        cands = pat.identify_candidates(mgr.limbdd, level)
        for cand in cands:
            node, connect_list, succ_l, succ_r, ptype, _ = cand
            if ptype == 3:
                return cand, level
    return None, None


def main():
    out = []
    found = 0
    t0 = time.perf_counter()
    for n in (4, 5):
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            tts = [int(l.rstrip('\n')) for l in f.readlines()[:400]]
        for tt_int in tts:
            tt = TruthTable(tt_int, n)
            mgr = LimBDDManager()
            mgr.initialize(tt, list(range(n)))
            cand, level = find_type3(mgr)
            if cand is None:
                continue

            node, connect_list, succ_l, succ_r, ptype, succ_info = cand
            before = pool_of(mgr.limbdd, n)

            # Perform the split (main graph only; new pieces go into mgr.temp_list)
            mgr.perform_split(mgr.limbdd)
            mgr.append_temp_to_list()
            pieces = [mgr.limbdd]
            pieces += [g for gs in mgr.limbdd_list.values() for g in gs]
            after = pool_of_pieces(pieces, n)

            only_before = before - after
            only_after = after - before
            s_before = sum(k[1] * v for k, v in only_before.items())
            s_after = sum(k[1] * v for k, v in only_after.items())

            x_b = 0
            for (fn, _, _), cnt in before.items():
                x_b ^= fn * (cnt & 1)
            x_a = 0
            for (fn, _, _), cnt in after.items():
                x_a ^= fn * (cnt & 1)

            msg = [f'--- case n={n} tt={tt_int} (deg={anf_degree(tt_int, n)}, trigger level {level}) ---',
                   f'full-pool XOR invariant: {x_b == x_a} ({x_b:0{1 << n}b}); '
                   f'pool segment count {sum(before.values())} → {sum(after.values())}',
                   f'segments only before split ({sum(only_before.values())} items, Σcost={s_before}):']
            for (fn, cost, k), cnt in only_before.items():
                msg.append(f'  fn={fn:0{1 << n}b} cost={cost} nctrl={k} ×{cnt}')
            msg.append(f'segments only after split ({sum(only_after.values())} items, Σcost={s_after}):')
            for (fn, cost, k), cnt in only_after.items():
                msg.append(f'  fn={fn:0{1 << n}b} cost={cost} nctrl={k} ×{cnt}')
            print('\n'.join(msg), flush=True)
            out += msg + ['']
            found += 1
            if found >= 3:
                break
        if found >= 3:
            break
    print(f'\ntime {time.perf_counter()-t0:.1f}s, {found} cases found')
    with open(OUT, 'w') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
