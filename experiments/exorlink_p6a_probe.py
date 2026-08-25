# -*- coding: utf-8 -*-
"""
P6a probe: bulk verification of exorlink-δ opportunity density and the δ=2 sign prediction (n ≤ 7).

Three snapshots; for each pool, count the classical distance δ of all segment pairs:
  ANF pool     : anf_pool(original f)                          -- all ordinary cubes;
  BDD pool     : graph-stage product (_bdd_entries, with root masks) -- general affine segments;
  fixed-point pool: fused fixed point of the best FPRM shift   -- mixed routing/synthesis.
General affine segments use canonical_affine's RREF constraint rows (mask, const) as "literals":
same mask and same const = agree; same mask but different const = opposite polarity (disagreement); one-sided = disagreement.

Metrics:
  δ distribution histogram; fraction of δ=2 pairs with a positive-gain candidate; positive-gain hit rate for δ≥3 pairs;
  bulk verification of the revised sign prediction (positive gain ⇔ an opposite-polarity variable exists, gain = t(k)−t(k−1))
  (actual gain vs formula, count of mismatches);
  one-shot improvement of the fixed-point pool: global best positive-gain δ=2/≥3 rewrite + full-pool re-fusion ΣT increment.
"""
import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import config
from backend.CompleteBranch import CompleteBranch, plan_from_affine
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.ClosedForm import canonical_affine, fuse_closed
from backend.fusion.GLSearch import fprm_best

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'exorlink_p6a_probe.txt')
PROFILE = {5: 60, 6: 40, 7: 20}
TC = config['t_cost']


def rows_of(cb, r, n):
    """Canonical constraint rows of a general affine segment: ((mask, const), ...), including root-mask shift (base ^= r).

    Note: affine_rows() removes the pivot bit from each row mask (the pivot is implicit); it must be added back as (1 << pivot).
    """
    base, basis = canonical_affine(cb, r)
    plan = plan_from_affine(base, basis, n)
    tmp = CompleteBranch(n, synthetic=plan)
    return tuple(((1 << col) | mask, const) for col, mask, const in tmp.affine_rows())


def pair_info(ra, rb, n):
    """
    Classical distance δ between two constraint-row sets and the candidate h list (all single-row disagreement candidates).
    Returns (δ, agree_rows, opp_indices, oneside_indices).
    Row sets are compared in canonical (mask, const) form.
    """
    da = dict((m, c) for m, c in ra)
    db = dict((m, c) for m, c in rb)
    agree = []
    opp = []
    oneside_a = []
    oneside_b = []
    for m in da:
        if m in db:
            if da[m] == db[m]:
                agree.append((m, da[m]))
            else:
                opp.append(m)
        else:
            oneside_a.append((m, da[m]))
    for m in db:
        if m not in da:
            oneside_b.append((m, db[m]))
    return len(opp) + len(oneside_a) + len(oneside_b), agree, opp, oneside_a, oneside_b


def h_candidates(agree, opp, oneside_a, oneside_b):
    """Constraint row list for a candidate h: one row per diverging variable (opp takes one of each side)."""
    out = []
    for m in opp:
        out.append(agree + [(m, 0)])
        out.append(agree + [(m, 1)])
    for m, c in oneside_a:
        out.append(agree + [(m, c)])
    for m, c in oneside_b:
        out.append(agree + [(m, c)])
    return out


def cb_from_rows(rows, n):
    """Constraint row set → CompleteBranch (synthetic representation). Rows must be linearly independent."""
    pivots = {}
    for m, c in rows:
        mm, cc = m, c
        while mm:
            low = mm & (-mm)
            if low in pivots:
                mm ^= pivots[low][0]
                cc ^= pivots[low][1]
            else:
                pivots[low] = (mm, cc)
                break
        if mm == 0 and cc:
            return None          # unsatisfiable, not a cube
    # Build (base, basis)
    basis = []
    base = 0
    pivot_cols = {v.bit_length() - 1 for v in pivots}
    for f in range(n):
        if f in pivot_cols:
            continue
        v = 1 << f
        for m, c in pivots.values():
            if (m >> f) & 1:
                v |= 1 << (m & (-m)).bit_length() - 1
        basis.append(v)
    for m, c in pivots.values():
        if c:
            base |= 1 << ((m & (-m)).bit_length() - 1)
    return CompleteBranch(n, synthetic=plan_from_affine(base, basis, n))


def move_gain(cb_a, cb_b, rows_h, n):
    """Actual gain of the move {A,B} → {A⊕h, B⊕h}; returns (None, None) when fusion fails.

    Returns (gain, (kind1, kind2)) for structural assertions: positive gain ⇔ one side is kind1 (C2)
    and the other is kind3/4 (C3/C4 nesting, cost unchanged).
    """
    ch = cb_from_rows(rows_h, n)
    if ch is None:
        return None, None
    r1 = fuse_closed(canonical_affine(cb_a), canonical_affine(ch), n)
    r2 = fuse_closed(canonical_affine(cb_b), canonical_affine(ch), n)
    if not r1.ok or not r2.ok or r1.cancel or r2.cancel:
        return None, None
    g = TC(cb_a.nctrl()) + TC(cb_b.nctrl()) - TC(n - r1.dim) - TC(n - r2.dim)
    return g, (r1.kind, r2.kind)


def analyze_pool(pool, n, label, acc):
    """pool: [(cb, r)]. Count δ distribution, positive-gain hits for δ=2/≥3, and symbolic formula verification."""
    m = len(pool)
    hist = Counter()
    pos2 = pos3 = tot2 = tot3 = 0
    formula_bad = 0
    best_move = None
    for i in range(m):
        ri = rows_of(pool[i][0], pool[i][1], n)
        for j in range(i + 1, m):
            rj = rows_of(pool[j][0], pool[j][1], n)
            d, agree, opp, oneside_a, oneside_b = pair_info(ri, rj, n)
            hist[min(d, 5)] += 1
            if d < 2:
                continue
            cands = h_candidates(agree, opp, oneside_a, oneside_b)
            gains = []                       # [(gain, kinds)]
            for rows_h in cands:
                g, kinds = move_gain(pool[i][0], pool[j][0], rows_h, n)
                if g is not None:
                    gains.append((g, kinds))
            pos = max((g for g, _ in gains), default=0)
            has_pos = pos > 0
            if d == 2:
                tot2 += 1
                pos2 += has_pos
                if has_pos:
                    gi = next(k for k, (g, _) in enumerate(gains) if g == pos)
                    if best_move is None or pos > best_move[0]:
                        best_move = (pos, i, j, cands[gi])
            else:
                tot3 += 1
                pos3 += has_pos
                if has_pos:
                    gi = next(k for k, (g, _) in enumerate(gains) if g == pos)
                    if best_move is None or pos > best_move[0]:
                        best_move = (pos, i, j, cands[gi])
            # Symbolic prediction check (structural): for each candidate, positive gain ⇔
            # one side is kind1 (C2) and the other is kind3/4 (C3/C4 nested, cost unchanged).
            for g, kinds in gains:
                is_c2 = 1 in kinds
                is_nest = any(k in (3, 4) for k in kinds)
                if (g > 0) != (is_c2 and is_nest):
                    formula_bad += 1
    acc['hist'] += hist
    acc['pairs'] += m * (m - 1) // 2
    acc['tot2'] += tot2
    acc['pos2'] += pos2
    acc['tot3'] += tot3
    acc['pos3'] += pos3
    acc['formula_bad'] += formula_bad
    return best_move


def apply_move(pool, best_move, n):
    """Apply one best rewrite and re-fuse the whole pool; return (new fixed-point cost, old cost)."""
    pos, i, j, rows_h = best_move
    ch = cb_from_rows(rows_h, n)
    r1 = fuse_closed(canonical_affine(pool[i][0], pool[i][1]), canonical_affine(ch), n)
    r2 = fuse_closed(canonical_affine(pool[j][0], pool[j][1]), canonical_affine(ch), n)
    p1 = CompleteBranch(n, synthetic=plan_from_affine(r1.base, r1.basis, n))
    p2 = CompleteBranch(n, synthetic=plan_from_affine(r2.base, r2.basis, n))
    new_pool = [(cb, r, key) for key, (cb, r) in enumerate(pool)]
    new_pool[i] = (p1, pool[i][1], i)
    new_pool[j] = (p2, pool[j][1], j)
    kept, st = fuse_pool_global(new_pool, n)
    return pool_t_cost(kept, st)


def main():
    lines = ['P6a probe: exorlink-δ opportunity density and δ=2 symbolic-prediction batch verification', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = {k: 0 for k in ('hist', 'pairs', 'tot2', 'pos2', 'tot3', 'pos3', 'formula_bad')}
        acc['hist'] = Counter()
        one_step = Counter()
        t0 = time.perf_counter()
        for fn in data:
            # ANF pool
            pool_anf = [(cb, 0) for cb, _, _ in anf_pool(fn, n)]
            analyze_pool(pool_anf, n, 'anf', acc)
            # BDD pool
            from backend.OracleSynthesizer import OracleSynthesizer
            synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
            pool_bdd = [(cb, r) for cb, r, _ in synth._bdd_entries(__import__('utils.TruthTable', fromlist=['TruthTable']).TruthTable(fn, n), None)]
            analyze_pool(pool_bdd, n, 'bdd', acc)
            # Fixed-point pool
            vF, cF = fprm_best(fn, n)
            kept, st = fuse_pool_global(anf_pool(shift_fn(fn, cF, n), n), n)
            pool_fp = [(cb, r) for cb, r, _ in kept] + [(cb, 0) for cb in st]
            best_move = analyze_pool(pool_fp, n, 'fp', acc)
            one_step['cases'] += 1
            if best_move is not None:
                c0 = pool_t_cost(kept, st)
                c1 = apply_move(pool_fp, best_move, n)
                one_step['gain_sum'] += c0 - c1
                one_step['gain_max'] = max(one_step['gain_max'], c0 - c1)
                one_step['nonzero'] += c1 < c0
        wall = time.perf_counter() - t0
        h = acc['hist']
        msg = [
            f'====== n = {n}({cases} cases,{wall:.1f}s)======',
            f'total segment pairs {acc["pairs"]}; δ distribution: ' +
            ', '.join(f'δ={k}:{h[k]}({h[k]/max(acc["pairs"],1):.0%})' for k in sorted(h)),
            f'δ=2: {acc["tot2"]} pairs total, positive gain {acc["pos2"]} ({acc["pos2"]/max(acc["tot2"],1):.1%})',
            f'δ≥3: {acc["tot3"]} pairs total, positive gain {acc["pos3"]} ({acc["pos3"]/max(acc["tot3"],1):.2%})',
            f'symbolic prediction mismatches: {acc["formula_bad"]}',
            f'one-shot fixed-point improvement: effective {one_step["nonzero"]}/{one_step["cases"]},'
            f' Σ gain {one_step["gain_sum"]}, max {one_step["gain_max"]}',
        ]
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nResults written to {OUT}')


if __name__ == '__main__':
    main()
