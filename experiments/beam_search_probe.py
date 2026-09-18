# -*- coding: utf-8 -*-
"""
P7 probe: engine-level gain from beam search realizing the "structural promise" (n ≤ 7 only).

Motivation (P3/P4 + P6a explanation correction): BDD's small-n advantage comes from the
structural promise of split mode (δ=2 rewrite during graph construction); the engine-side
substitute = beam search that commits to matching structures — each round keeps width best
pool states (single-step branch moves) instead of a greedy maximal matching in one shot.

Comparison setup:
  pool A = BDD graph-phase product (_bdd_entries, with root mask);
  pool FS = ANF pool of the FPRM-optimal shift (before fusion).
  variants: greedy (production fuse_pool_global), beam1/beam4/beam16 (single-step-move beam);
  pool FS additionally gets GL search cost (measuring beam's recovery rate vs full-coordinate refactorization).

Implemented as a probe script; production code untouched. State dedup key = (ΣT, set of (cb, r) in pool).
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.TruthTable import TruthTable
from backend.CompleteBranch import CompleteBranch, plan_from_affine
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.ClosedForm import canonical_affine, fuse_closed
from backend.fusion.GLSearch import fprm_best, gl_search

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'beam_search_probe.txt')
PROFILE = {5: (30, (1, 4, 16)), 6: (20, (1, 4, 16)), 7: (8, (1, 4))}


def successors(state, n):
    """Single-step-move successors of one pool state: one successor for each cancellation/positive-gain fusion."""
    cost, pool = state
    aff = [canonical_affine(cb, r) for cb, r, _ in pool]
    out = []
    seen = set()
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            res = fuse_closed(aff[i], aff[j], n)
            if not res.ok:
                continue
            if res.cancel:
                new_pool = [e for k, e in enumerate(pool) if k not in (i, j)]
                new_cost = cost - (pool[i][0].t_cost() + pool[j][0].t_cost())
            else:
                fused = CompleteBranch(n, synthetic=plan_from_affine(res.base, res.basis, n),
                                       origin=pool[i][0].origin)
                # Accept only cancellation or non-negative gain (structural-promise basis: neutral moves allowed)
                if pool[i][0].t_cost() + pool[j][0].t_cost() < fused.t_cost():
                    continue
                new_pool = [e for k, e in enumerate(pool) if k not in (i, j)] + [(fused, 0, None)]
                new_cost = cost - (pool[i][0].t_cost() + pool[j][0].t_cost()
                                   - fused.t_cost())
                assert new_cost <= cost
            key = frozenset((cb, r) for cb, r, _ in new_pool)
            if key in seen:
                continue
            seen.add(key)
            out.append((new_cost, new_pool))
    return out


def beam_fuse(entries, n, width, max_rounds=None):
    """Single-step-move beam search; returns (best_cost, best_pool)."""
    cost0 = sum(cb.t_cost() for cb, _, _ in entries)
    states = [(cost0, list(entries))]
    m = len(entries)
    if max_rounds is None:
        max_rounds = m + 2
    for _ in range(max_rounds):
        nxt = {}
        for st in states:
            for s in successors(st, n):
                key = (s[0], frozenset((cb, r) for cb, r, _ in s[1]))
                if key not in nxt or s[0] < nxt[key][0]:
                    nxt[key] = s
        if not nxt:
            break
        states = sorted(nxt.values(), key=lambda s: s[0])[:width]
    return min(states, key=lambda s: s[0])


def run_case(fn, n, widths):
    """Run greedy/beam variants on the BDD pool and the FPRM pool; return a statistics dict."""
    from backend.OracleSynthesizer import OracleSynthesizer
    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    tt = TruthTable(fn, n)
    bdd = synth._bdd_entries(tt, None)
    vF, cF = fprm_best(fn, n)
    frm = anf_pool(shift_fn(fn, cF, n), n)

    out = Counter()
    for label, entries in (('bdd', bdd), ('frm', frm)):
        t0 = time.perf_counter()
        g = pool_t_cost(*fuse_pool_global(entries, n))
        out[f'{label}_greedy'] = g
        out[f'{label}_greedy_wall'] = time.perf_counter() - t0
        for w in widths:
            t0 = time.perf_counter()
            cost, _ = beam_fuse(entries, n, w)
            out[f'{label}_beam{w}'] = cost
            out[f'{label}_beam{w}_wall'] = time.perf_counter() - t0
            out[f'{label}_beam{w}_better'] = 1 if cost < g else 0
    t0 = time.perf_counter()
    kept_g, st_g, _, _ = gl_search(fn, n)
    out['frm_gl'] = pool_t_cost(kept_g, st_g)
    out['frm_gl_wall'] = time.perf_counter() - t0
    return out


def main():
    lines = ['P7 probe: beam search (engine-level structural promise) vs greedy fusion', '']
    for n, (cases, widths) in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = Counter()
        for fn in data:
            r = run_case(fn, n, widths)
            for k, v in r.items():
                acc[k] += v
        msg = [f'====== n = {n}({cases} cases)======']
        for label in ('bdd', 'frm'):
            g = acc[f'{label}_greedy']
            msg.append(f'{label} pool: greedy ΣT={g} ({acc[f"{label}_greedy_wall"]:.1f}s)')
            for w in widths:
                b = acc[f'{label}_beam{w}']
                msg.append(f'  beam{w}: ΣT={b} ({(b/g-1):+.2%}, better {acc[f"{label}_beam{w}_better"]}/{cases}, '
                           f'{acc[f"{label}_beam{w}_wall"]:.1f}s)')
        gl = acc['frm_gl']
        msg.append(f'frm pool comparison: GL ΣT={gl} ({(gl/acc["frm_greedy"]-1):+.2%}, {acc["frm_gl_wall"]:.1f}s)')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nresults written to {OUT}')


if __name__ == '__main__':
    main()
