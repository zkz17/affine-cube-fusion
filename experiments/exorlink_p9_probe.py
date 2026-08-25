# -*- coding: utf-8 -*-
"""
P9 probe: production-style interleaved scheduling verification of macro M2 (five paths, each measuring its marginal gain).

Scheduling (production-style): pool → fuse_pool_global (maximal matching, to fixed point) → enumerate the best macro M2 →
apply → re-fuse, loop; bounded by the interleave round limit. Control V0 = pure production fusion (no M2).

The **pre-fusion** pools of the five paths:
  A  = BDD graph-stage product (split on, absorb removed);
  C  = anf_pool(f) (c=0);
  F  = anf_pool(f∘τ_cF) (FPRM optimal shift);
  FP = anf_pool(f∘τ_cFp) (fprm_proxy optimal shift);
  G  = gl_search result pool (kept+standalone; the pool body is the decomposition of g=f(Ax⊕c)).

Verification:
  - after every fusion/macro application, assert the pool XOR invariant;
  - the final pools of the A/C/F/FP paths are emitted through the production emission pipeline
    (shift sandwich + root sandwich + B-4 reordering + simplify) and must have compute_tt == f;
    the G path's linear wrapper belongs to the emission layer, so only the pool XOR == g is checked
    (the emission layer does not take part in M2, so G only measures pool-level marginals).
Isolation: read only data/test_cases, write only experiments/phase0_results/.
"""
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import config
from utils.Qubit import QubitManager
from utils.TruthTable import TruthTable
from backend.CompleteBranch import CompleteBranch, _bits, plan_from_affine
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn
from backend.fusion.ClosedForm import canonical_affine, fuse_closed
from backend.fusion.FprmProxy import fprm_proxy_best
from backend.fusion.GLSearch import (apply_linear, compose_transvection_right,
                                    fprm_best, gl_search)
from backend.OracleSynthesizer import OracleSynthesizer
from circuit.MCXCircuit import MCXCircuit, order_segments_greedy
import experiments.exorlink_p6a_probe as P6

OUT = os.path.join(ROOT, 'experiments', 'phase0_results', 'exorlink_p9_probe.txt')
TC = config['t_cost']
PROFILE = {5: 25, 6: 15, 7: 6}
MAX_INTERLEAVE = 30


def seg_from(res, root, n):
    return CompleteBranch(n, synthetic=plan_from_affine(res.base ^ root, res.basis, n))


def pool_xor(pool, n):
    x = 0
    for cb, r, _ in pool:
        x ^= shift_fn(cb.fn(), r, n)
    return x


def best_m2(pool, n):
    """
    Best macro M2 at a fixed point (true total gain = pair move imm + A′+C fusion gain, strictly > 0).

    Returns (total, imm, g_fuse, idxA, idxB, k, rA, rB, rc, side) or None.
    Note: at a fixed point imm ≤ 0 (otherwise the d2 move would already have been taken);
    Π-neutral pairs have imm == 0, while imm < 0 is the "bounded worsening + fusion
    compensation" generalization (total gain is still strictly positive).
    """
    aff = [canonical_affine(cb, r) for cb, r, _ in pool]
    rows = [P6.rows_of(pool[i][0], pool[i][1], n) for i in range(len(pool))]
    m = len(pool)
    best = None
    for i in range(m):
        ti = pool[i][0].t_cost()
        for j in range(i + 1, m):
            d, agree, opp, oa, ob = P6.pair_info(rows[i], rows[j], n)
            if d != 2:
                continue
            for rows_h in P6.h_candidates(agree, opp, oa, ob):
                ch = P6.cb_from_rows(rows_h, n)
                if ch is None:
                    continue
                ah = canonical_affine(ch)
                r1 = fuse_closed(aff[i], ah, n)
                r2 = fuse_closed(aff[j], ah, n)
                if not (r1.ok and r2.ok and not r1.cancel and not r2.cancel):
                    continue
                imm = ti + pool[j][0].t_cost() - TC(n - r1.dim) - TC(n - r2.dim)
                for side, (rA, rB, idxA, idxB) in (('A', (r1, r2, i, j)),
                                                   ('B', (r2, r1, j, i))):
                    tA = TC(n - rA.dim)
                    affA = (rA.base, rA.basis)
                    for k in range(m):
                        if k in (i, j):
                            continue
                        rc = fuse_closed(affA, aff[k], n)
                        if not rc.ok or rc.cancel:
                            continue
                        g_fuse = tA + pool[k][0].t_cost() - TC(n - rc.dim)
                        total = imm + g_fuse
                        if total > 0 and (best is None or total > best[0]):
                            best = (total, imm, g_fuse, idxA, idxB, k, rA, rB, rc, side)
    return best


def interleaved(entries, n, expected_xor):
    """Production-style interleaved scheduling: fuse to fixed point → best M2 → re-fuse, loop. Returns (cost, pool, stats)."""
    pool = [(cb, r, key) for cb, r, key in entries]
    stats = Counter()
    rounds = 0
    while rounds < MAX_INTERLEAVE:
        kept, st = fuse_pool_global(pool, n)
        pool = [(cb, r, key) for cb, r, key in kept] + [(cb, 0, None) for cb in st]
        assert pool_xor(pool, n) == expected_xor, 'pool XOR invariant broken after fusion'
        mv = best_m2(pool, n)
        if mv is None:
            break
        total, imm, g_fuse, idxA, idxB, k, rA, rB, rc, side = mv
        b1 = seg_from(rB, pool[idxB][1], n)
        fused = seg_from(rc, 0, n)
        pool = [e for q, e in enumerate(pool) if q not in (idxA, idxB, k)] + \
               [(b1, pool[idxB][1], pool[idxB][2]), (fused, 0, None)]
        assert pool_xor(pool, n) == expected_xor, 'pool XOR invariant broken after M2'
        stats['m2'] += 1
        stats['m2_gain'] += total
        stats['m2_neutral'] += imm == 0
        rounds += 1
    return sum(cb.t_cost() for cb, _, _ in pool), pool, stats


def emit_and_verify(pool, fn, n, shift_c=0):
    """Emit through the production emission pipeline (shift sandwich + root sandwich + B-4 reordering + simplify) and verify."""
    qm = QubitManager()
    for v in range(n):
        qm.get_input_qubit(v)
    circ = MCXCircuit(qm)
    circ.add_lim_operator([qm.get_input_qubit(v) for v in _bits(shift_c)])
    kept_by_r = {}
    for cb, r, key in pool:
        if key is not None:
            kept_by_r.setdefault(r, []).append(cb)
    for r, segs in sorted(kept_by_r.items()):
        rq = [qm.get_input_qubit(v) for v in _bits(r)]
        circ.add_lim_operator(rq)
        for sc in order_segments_greedy([cb.to_circuit(qm) for cb in segs],
                                        circ if not circ.empty() else None):
            circ.merge(sc)
        circ.add_lim_operator(rq)
    stand = [cb for cb, r, key in pool if key is None]
    for sc in order_segments_greedy([cb.to_circuit(qm) for cb in stand],
                                    circ if not circ.empty() else None):
        circ.merge(sc)
    circ.add_lim_operator([qm.get_input_qubit(v) for v in _bits(shift_c)])
    circ.simplify_terms()
    assert circ.compute_tt() == fn, 'emission-level compute_tt verification failed'
    return circ


def g_of_cnots(fn, shift_c, cnots, n):
    """Rebuild A from gl_search's cnots (generation order); return g = f(Ax ⊕ c)."""
    A = [1 << i for i in range(n)]
    for c, t in cnots:
        A = compose_transvection_right(A, c, t, n)
    return apply_linear(shift_fn(fn, shift_c, n) if shift_c else fn, A, n)


def run_case(fn, n, acc):
    tt = TruthTable(fn, n)
    synth = OracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)

    # ---- Path preparation phase (counted in each path's total-time budget) ----
    t0 = time.perf_counter()
    entries_a = synth._bdd_entries(tt, None)
    acc['A_prep_wall'] += time.perf_counter() - t0

    t0 = time.perf_counter()
    entries_c = anf_pool(fn, n)
    acc['C_prep_wall'] += time.perf_counter() - t0

    t0 = time.perf_counter()
    _, cF = fprm_best(fn, n)
    acc['F_prep_wall'] += time.perf_counter() - t0

    t0 = time.perf_counter()
    _, _, cFp = fprm_proxy_best(fn, n)
    acc['FP_prep_wall'] += time.perf_counter() - t0

    t0 = time.perf_counter()
    kept_g, st_g, c_g, cnots_g = gl_search(fn, n)
    acc['G_prep_wall'] += time.perf_counter() - t0

    g_pool = [(cb, r, key) for cb, r, key in kept_g] + [(cb, 0, None) for cb in st_g]
    pools = {
        'A': (entries_a, fn, 0),
        'C': (entries_c, fn, 0),
        'F': (anf_pool(shift_fn(fn, cF, n), n), shift_fn(fn, cF, n), cF),
        'FP': (anf_pool(shift_fn(fn, cFp, n), n), shift_fn(fn, cFp, n), cFp),
        'G': (g_pool, g_of_cnots(fn, c_g, cnots_g, n), None),
    }
    for pname, (entries, expect, shift_c) in pools.items():
        t0 = time.perf_counter()
        v0 = pool_t_cost(*fuse_pool_global(entries, n))
        acc[f'{pname}_V0_wall'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        v1, pool1, st = interleaved(entries, n, expect)
        acc[f'{pname}_V1_wall'] += time.perf_counter() - t0
        acc[f'{pname}_V0'] += v0
        acc[f'{pname}_V1'] += v1
        acc[f'{pname}_V1_better'] += v1 < v0
        acc[f'{pname}_m2'] += st['m2']
        acc[f'{pname}_m2_gain'] += st['m2_gain']
        acc[f'{pname}_m2_neutral'] += st['m2_neutral']
        if pname != 'G':
            emit_and_verify(pool1, fn, n, shift_c)
            acc[f'{pname}_emit_ok'] += 1
    acc['cases'] += 1


def main():
    lines = ['P9 probe: production-style interleaved scheduling of macro M2 (five-path marginals)', '']
    for n, cases in PROFILE.items():
        data = [int(l.rstrip('\n')) for l in
                open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')).readlines()[:cases]]
        acc = Counter()
        for fn in data:
            run_case(fn, n, acc)
        c = acc['cases']
        msg = [f'====== n = {n}({cases} cases)======']
        for pname in ('A', 'C', 'F', 'FP', 'G'):
            v0, v1 = acc[f'{pname}_V0'], acc[f'{pname}_V1']
            msg.append(f'{pname:>3} pool: V0={v0}  V1={v1}  ({(v1/v0-1):+.2%},better '
                       f'{acc[f"{pname}_V1_better"]}/{c}, M2 {acc[f"{pname}_m2"]} times/Σgain '
                       f'{acc[f"{pname}_m2_gain"]},neutral pairs '
                       f'{acc[f"{pname}_m2_neutral"]})')
        msg.append(f'A/C/F/FP emission-level compute_tt verification: {acc["A_emit_ok"]}/{c} passed')
        for pname in ('A', 'C', 'F', 'FP', 'G'):
            prep = acc[f'{pname}_prep_wall'] / c
            v0 = acc[f'{pname}_V0_wall'] / c
            v1 = acc[f'{pname}_V1_wall'] / c
            msg.append(f'{pname:>3} time (ms/case): prep={prep*1e3:8.2f}  V0 fusion={v0*1e3:7.2f}  '
                       f'V1 fusion+M2={v1*1e3:7.2f}  M2 increment={max(v1-v0,0)*1e3:6.2f}')
        fp_m2 = (acc['FP_prep_wall'] + acc['FP_V1_wall']) / c
        f_orig = (acc['F_prep_wall'] + acc['F_V0_wall']) / c
        msg.append(f'FP+M2 total time (prep+V1)={fp_m2*1e3:.2f} ms/case  vs  original F (prep+V0)='
                   f'{f_orig*1e3:.2f} ms/case  ({(fp_m2/f_orig-1):+.1%})')
        print('\n'.join(msg), flush=True)
        lines += msg + ['']
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nResults written to {OUT}')


if __name__ == '__main__':
    main()
