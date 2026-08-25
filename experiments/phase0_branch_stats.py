# -*- coding: utf-8 -*-
"""
Phase 0: branch-level fusion opportunity statistics (measurement only; pipeline and algorithm unchanged).

Method: use ProbeSynthesizer (an OracleSynthesizer subclass) to override the "circuit generation"
part of limbdd2circ — the graph phase (perform_split / collect_branches / append_temp_to_list) is
identical to the real pipeline, but circuit generation is replaced with:
  1. extract normalized signatures of all branches currently in the subgraph (Lim mapped to the absolute variable bit space);
  2. run signature-level global verification: XOR(all branch segments of every synthesized subgraph) == original truth table;
  3. count rule 1/2/3 candidate pairs and gains (rule 3 is an upper-bound estimate);
  4. optional: sample complete path pairs from the same graph, execute pivot_to + nested synthesis, and measure the fused T.

Output: stdout + experiments/phase0_results/phase0_results.txt (override with --out)
"""
import argparse
import collections
import os
import random
import sys
import time

# Project root (script lives under experiments/, so go up one level); add to sys.path so limbdd/utils is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENTS = os.path.join(ROOT, 'experiments')

from config import config
from utils.TruthTable import TruthTable
from backend.OracleSynthesizer import OracleSynthesizer
from limbdd.LimBDD import SimpleBranch, LimBDD
from limbdd.LimBDDManager import LimBDDManager
from limbdd.Node import LimBDDNode, LimBDDSignal
from circuit.MCXCircuit import MCXCircuit
from utils.to_qiskit import to_qiskit
from qiskit.circuit.library.standard_gates.x import XGate, CXGate, MCXGate, CCXGate


def t_cost(k):
    """T cost of a k-controlled MCX (single source of truth: 't_cost' in config.py, near-linear new model)."""
    return config['t_cost'](k)


def abs_lim(lim, var_order, anomalies):
    """Convert an in-graph Lim (level space) to absolute variable bit space: bit k = variable k."""
    m = 0
    for j in range(lim.bitnum):
        if (lim.bitstr >> j) & 1:
            if j >= len(var_order):
                anomalies['lim_overflow'] += 1
                continue
            m |= 1 << var_order[j]
    return m


# ---------------------------------------------------------------- extraction ----

def extract_branches(limbdd, anomalies):
    """
    Extract branch signatures mirroring the branch2circ walk level by level.
    Returns [(sig, branch_bitstr, branch_bitnum)]:
      sig = (position, ...) top-down; position = (var, ft, bit, w0, w1, lim0, lim1)
      ft=1 means fallthrough (consumes no branch bit); lim values are absolute variable bit-space masks.
    """
    terminator = limbdd.terminator
    results = []

    def walk(node, positions, bits, bitnum):
        if node is terminator:
            results.append((tuple(positions), bits, bitnum))
            return
        if node.level == 0:  # a non-terminal level-0 node should not occur
            anomalies['zero_level'] += 1
            return
        w0, w1 = node.low.weight, node.high.weight
        if w0 == 0 and w1 == 0:
            anomalies['zero_node'] += 1
            return
        if node.low.neg or node.high.neg:
            anomalies['neg_edge'] += 1
        ft = (node.low.node is node.high.node and w0 == 1 and w1 == 1)
        l0 = abs_lim(node.low.lim, limbdd.var_order, anomalies)
        l1 = abs_lim(node.high.lim, limbdd.var_order, anomalies)
        if ft:
            pos = (node.var_index, 1, -1, w0, w1, l0, l1)
            walk(node.low.node, positions + [pos], bits, bitnum)
        else:
            for b, succ in ((0, node.low), (1, node.high)):
                if succ.weight:
                    pos = (node.var_index, 0, b, w0, w1, l0, l1)
                    walk(succ.node, positions + [pos], bits | (b << bitnum), bitnum + 1)

    walk(limbdd.root.node, [], 0, 0)
    return results


# ---------------------------------------------------------------- simulator ----

def sim_sig(sig, x):
    """Simulate one branch segment on input x (absolute variable bit space); return the target flip bit."""
    state = x
    ok = True
    for (var, ft, bit, w0, w1, l0, l1) in sig:
        if ft:
            if ((state >> var) & 1) == 0:
                state ^= l0
            else:
                state ^= l1
        else:
            state ^= (l1 if bit else l0)
            if ((state >> var) & 1) != bit:
                ok = False
    return 1 if ok else 0


def sigs_xor_tt(sigs, root_mask, n):
    """Truth table (bitstr over 2^n) of XOR(all branch segments) on input x^root_mask."""
    tt = 0
    for x in range(1 << n):
        xr = x ^ root_mask
        f = 0
        for sig in sigs:
            f ^= sim_sig(sig, xr)
        if f:
            tt |= 1 << x
    return tt


# ---------------------------------------------------------------- probe synthesizer ----

class ProbeSynthesizer(OracleSynthesizer):
    """
    Override limbdd2circ: the graph phase is identical to the real pipeline; the circuit-generation
    step instead records branch signatures.
    recorded: [(sig, root_mask, origin, branch_bitstr, branch_bitnum)]
    """

    def __init__(self, reorder=True, ancilla_num=0, n=0, pivot_sample=0, rng=None):
        super().__init__(reorder=reorder, ancilla_num=ancilla_num, fused_backend=False)
        self.n = n
        self.pivot_sample = pivot_sample
        self.rng = rng or random.Random()
        self.recorded = []
        self.pivot_records = []
        self.origin_counter = 0

    def limbdd2circ(self, limbdd, anc_num, collect_br=True):
        # ---- same graph phase as the real limbdd2circ ----
        if not limbdd.root.weight:
            return MCXCircuit()
        if collect_br:
            self.limbdd_mgr.collect_branches(limbdd, limbdd.root.level)
        else:
            self.limbdd_mgr.perform_split(limbdd)
        self.limbdd_mgr.append_temp_to_list()
        if not limbdd.root.weight:  # absorption/splitting may zero this piece (same as the real limbdd2circ)
            return MCXCircuit()

        # ---- the original code generates the circuit here; we record branch signatures instead ----
        origin = self.origin_counter
        self.origin_counter += 1
        anomalies = collections.Counter()
        res = extract_branches(limbdd, anomalies)
        root_mask = abs_lim(limbdd.root.lim, limbdd.var_order, anomalies)
        for (sig, bits, bitnum) in res:
            self.recorded.append((sig, root_mask, origin, bits, bitnum))

        if self.pivot_sample > 0:
            self._pivot_sample_inline(limbdd, res, anomalies)

        return MCXCircuit()  # circuit generation omitted (Phase 0 only measures)

    def _pivot_sample_inline(self, limbdd, res, anomalies):
        """For same-graph "function-level fusible pairs" (equal or affine cube), run a sampled pivot_to measurement."""
        if self.pivot_sample <= 0 or len(res) < 2:
            return
        fns = [sigs_xor_tt([sig], 0, self.n) for (sig, bits, bitnum) in res]
        cands = []
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                d = fns[i] ^ fns[j]
                if d == 0:
                    cands.append((i, j, -1))
                    continue
                pc = d.bit_count()
                if pc > (1 << min(self.n, 7)) or (pc & (pc - 1)) != 0:
                    continue
                ok, k = is_affine_subspace(d, self.n)
                if ok:
                    cands.append((i, j, k))
        if not cands:
            return
        if len(cands) > self.pivot_sample:
            cands = self.rng.sample(cands, self.pivot_sample)
        for i, j, k in cands:
            sig_a, bits_a, bitnum_a = res[i]
            sig_b, bits_b, bitnum_b = res[j]
            ka, kb = nctrl(sig_a), nctrl(sig_b)
            est = t_cost(ka) + t_cost(kb) - (t_cost(self.n - k) if k >= 0 else 0)
            try:
                pivot_limbdd, x_map, cx_map = LimBDDManager.pivot_to(
                    limbdd.root.node, SimpleBranch(bits_a, bitnum_a), SimpleBranch(bits_b, bitnum_b), limbdd)
                nested = OracleSynthesizer(ancilla_num=0, fused_backend=False)
                nested.qubit_mgr = self.qubit_mgr
                nested.circ = MCXCircuit(self.qubit_mgr)
                nested.limbdd_mgr.limbdd = pivot_limbdd
                nested.limbdd_mgr.limbdd_list.clear()
                for h in range(pivot_limbdd.root.level + 1):
                    nested.limbdd_mgr.limbdd_list[h] = []
                nested.process()
                fused_t = count_t_cost(to_qiskit(nested.circ))
                ttp = nested.circ.compute_tt()
                ok = True
                for x in range(1 << self.n):
                    y = x
                    for var in x_map:
                        if x_map[var]:
                            y ^= 1 << var
                    for ctrl in cx_map:
                        for tgt in cx_map[ctrl]:
                            if ((x >> ctrl) & 1) == cx_map[ctrl][tgt]:
                                y ^= 1 << tgt
                    if ((ttp >> y) & 1) != (sim_sig(sig_a, x) ^ sim_sig(sig_b, x)):
                        ok = False
                        break
                self.pivot_records.append({'ka': ka, 'kb': kb, 'dim': k, 'est_t': est,
                                           'fused_t': fused_t, 'ok': ok})
            except Exception as e:
                self.pivot_records.append({'ka': ka, 'kb': kb, 'dim': k, 'est_t': est,
                                           'fused_t': None, 'ok': False, 'err': str(e)})


# ---------------------------------------------------------------- statistics ----

def nctrl(sig):
    return sum(1 for p in sig if p[1] == 0)


def is_affine_subspace(d, n):
    """Whether d's support-bit set is an affine subspace of GF(2)^n; returns (True, dimension k)."""
    bits = [i for i in range(1 << n) if (d >> i) & 1]
    if not bits:
        return False, 0
    base = bits[0]
    tset = {e ^ base for e in bits}
    tlist = sorted(tset)
    for a in range(len(tlist)):
        for b in range(a, len(tlist)):
            if tlist[a] ^ tlist[b] not in tset:
                return False, 0
    k = len(tlist).bit_length() - 1
    return True, k


def case_stats(n, tt, label, recorded, anomalies, pivot_records):
    st = {
        'n': n, 'label': label, 'tt': tt,
        'n_sub': 0, 'n_branch': 0,
        'rule1_pairs': 0, 'rule1_cross': 0, 'rule1_gain': 0,
        'rule2_pairs': 0, 'rule2_cross': 0, 'rule2_gain': 0,
        'complete': 0, 'rule3_same_pairs': 0, 'rule3_same_gain_ub': 0,
        'rule3_d1_pairs': 0, 'rule3_d1_gain': 0, 'rule3_dge2_pairs': 0,
        'rule3_agree_hist': collections.Counter(),
        'func1_pairs': 0, 'func1_gain': 0,
        'affine_pairs': 0, 'affine_gain': 0, 'affine_gain_pos': 0,
        'affine_pairs_cross': 0, 'affine_match_gain': 0,
        'affine_dim_hist': collections.Counter(),
        'flat_t': 0, 'fused_t_est': 0, 'actual_t': None,
        'valid': False, 'pivot_ok': 0, 'pivot_fail': 0,
        'pivot_fused_t_sum': 0, 'pivot_est_t_sum': 0, 'pivot_pairs': 0, 'pivot_base_t_sum': 0,
    }

    # ---- global verification: XOR(all branch segments of all synthesized subgraphs) == original truth table ----
    grouped = collections.defaultdict(list)
    for (sig, root_mask, origin, bits, bitnum) in recorded:
        grouped[origin].append((sig, root_mask))
    xor_total = 0
    for origin, items in grouped.items():
        xor_total ^= sigs_xor_tt([s for s, _ in items], items[0][1], n)
    st['valid'] = (xor_total == tt)
    st['n_sub'] = len(grouped)
    if not st['valid']:
        anomalies['validate_fail'] += 1

    segs = []  # (sig, bits, bitnum, origin, k)
    for (sig, root_mask, origin, bits, bitnum) in recorded:
        segs.append((sig, bits, bitnum, origin, nctrl(sig)))
    st['n_branch'] = len(segs)

    # ---- rule 1: identical signatures ----
    cnt = collections.Counter(s[0] for s in segs)
    cnt_origin = collections.Counter((s[0], s[3]) for s in segs)
    for sig, c in cnt.items():
        if c < 2:
            continue
        st['rule1_pairs'] += c * (c - 1) // 2
        st['rule1_gain'] += (c - c % 2) * t_cost(nctrl(sig))
    cross = st['rule1_pairs']
    for (sig, o), c in cnt_origin.items():
        cross -= c * (c - 1) // 2
    st['rule1_cross'] = cross

    # ---- rule 2: single-position 0/1 difference (at position p, w0=w1=1) ----
    buckets = {}
    for sig, bits, bitnum, origin, k in segs:
        for p, pos in enumerate(sig):
            var, ft, bit, w0, w1, l0, l1 = pos
            if ft or w0 != 1 or w1 != 1:
                continue
            key = sig[:p] + ((var, 0, -1, w0, w1, -1, -1),) + sig[p + 1:]
            buckets.setdefault(key, [[], []])[bit].append(origin)
    for key, (L0, L1) in buckets.items():
        if not L0 or not L1:
            continue
        kk = nctrl(key)
        npair = len(L0) * len(L1)
        st['rule2_pairs'] += npair
        st['rule2_gain'] += npair * max(0, t_cost(kk) - t_cost(kk - 1))
        c0, c1 = collections.Counter(L0), collections.Counter(L1)
        st['rule2_cross'] += npair - sum(c0[o] * c1[o] for o in set(c0) & set(c1))

    # ---- rule 3: complete paths (no fallthrough anywhere and w0=w1=1) ----
    complete = [(sig, bits, bitnum, origin, k) for (sig, bits, bitnum, origin, k) in segs
                if all(p[1] == 0 and p[3] == 1 and p[4] == 1 for p in sig)]
    st['complete'] = len(complete)
    by_len = collections.defaultdict(list)
    for e in complete:
        by_len[e[4]].append(e)
    PAIR_CAP = 200000
    for k, entries in by_len.items():
        npair = len(entries) * (len(entries) - 1) // 2
        pairs = [(i, j) for i in range(len(entries)) for j in range(i + 1, len(entries))]
        if npair > PAIR_CAP:
            pairs = [pairs[t] for t in random.Random(20250815 + k).sample(range(npair), PAIR_CAP)]
        for i, j in pairs:
            a, b = entries[i], entries[j]
            d = (a[1] ^ b[1]).bit_count()
            agree = k - d
            st['rule3_same_pairs'] += 1
            st['rule3_same_gain_ub'] += max(0, 2 * t_cost(k) - t_cost(agree))
            st['rule3_agree_hist'][agree] += 1
    for ka in sorted(by_len):
        for kb in sorted(by_len):
            if ka >= kb:
                continue
            pairs = len(by_len[ka]) * len(by_len[kb])
            if ka + 1 == kb:
                st['rule3_d1_pairs'] += pairs
                st['rule3_d1_gain'] += pairs * t_cost(ka)
            else:
                st['rule3_dge2_pairs'] += pairs

    # ---- function-level opportunity: whether the support set of the XOR of two branch functions is an
    #      affine subspace (the exact condition for fusing into one branch segment) ----
    fns = [sigs_xor_tt([sig], 0, n) for (sig, bits, bitnum, origin, k) in segs]
    cand_pairs = []  # (gain, i, j) for greedy matching
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            d = fns[i] ^ fns[j]
            if d == 0:
                gain = t_cost(segs[i][4]) + t_cost(segs[j][4])
                st['func1_pairs'] += 1
                st['func1_gain'] += gain
                cand_pairs.append((gain, i, j))
                continue
            pc = d.bit_count()
            if pc > (1 << min(n, 7)) or (pc & (pc - 1)) != 0:
                continue
            ok, k = is_affine_subspace(d, n)
            if ok:
                gain = t_cost(segs[i][4]) + t_cost(segs[j][4]) - t_cost(n - k)
                st['affine_pairs'] += 1
                st['affine_gain'] += gain
                if gain > 0:
                    st['affine_gain_pos'] += gain
                if segs[i][3] != segs[j][3]:
                    st['affine_pairs_cross'] += 1
                st['affine_dim_hist'][k] += 1
                if gain > 0:
                    cand_pairs.append((gain, i, j))
    cand_pairs.sort(reverse=True)
    used = set()
    for gain, i, j in cand_pairs:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        st['affine_match_gain'] += gain

    # ---- T baseline ----
    st['flat_t'] = sum(t_cost(k) for (sig, bits, bitnum, origin, k) in segs)
    st['fused_t_est'] = fused_t_estimate(segs)

    # ---- pivot measurement summary ----
    for r in pivot_records:
        if r['fused_t'] is None or not r['ok']:  # contract violation/exception: not counted in the gain
            st['pivot_fail'] += 1
            continue
        st['pivot_pairs'] += 1
        st['pivot_fused_t_sum'] += r['fused_t']
        st['pivot_est_t_sum'] += r['est_t']
        st['pivot_base_t_sum'] += t_cost(r['ka']) + t_cost(r['kb'])
        st['pivot_ok'] += 1

    return st


def fused_t_estimate(segs):
    """Conservative estimate: T after rule-1 cancellation + rule-2 fusion (then cancellation). Rule 3 not counted."""
    pool = collections.Counter(s[0] for s in segs)
    pool = collections.Counter({sig: c % 2 for sig, c in pool.items() if c % 2})
    if not pool:
        return 0
    buckets = {}
    for sig in pool.elements():
        for p, pos in enumerate(sig):
            var, ft, bit, w0, w1, l0, l1 = pos
            if ft or w0 != 1 or w1 != 1:
                continue
            key = sig[:p] + ((var, 0, -1, w0, w1, -1, -1),) + sig[p + 1:]
            buckets.setdefault(key, [[], []])[bit].append(sig)
    remaining = pool.copy()
    fused = collections.Counter()
    for key, (L0, L1) in buckets.items():
        m = min(sum(remaining[s] for s in set(L0)), sum(remaining[s] for s in set(L1)))
        i0 = i1 = 0
        while m > 0:
            while i0 < len(L0) and remaining[L0[i0]] <= 0:
                i0 += 1
            while i1 < len(L1) and remaining[L1[i1]] <= 0:
                i1 += 1
            if i0 >= len(L0) or i1 >= len(L1):
                break
            a, b = L0[i0], L1[i1]
            remaining[a] -= 1
            remaining[b] -= 1
            p = next(idx for idx, pos in enumerate(key) if pos[1] == 0 and pos[2] == -1)
            # a walks low (bit=0), b walks high (bit=1): fused into a fallthrough, lim0 = a's low edge, lim1 = b's high edge
            f = a[:p] + ((key[p][0], 1, -1, 1, 1, a[p][5], b[p][6]),) + a[p + 1:]
            fused[f] += 1
            m -= 1
    total = 0
    for sig, c in (remaining + fused).items():
        if c % 2:
            total += t_cost(nctrl(sig))
    return total


def count_t_cost(qc):
    t = 0
    for instruction in qc.data:
        gate = instruction.operation
        if isinstance(gate, MCXGate):
            t += t_cost(gate.num_ctrl_qubits)
        elif isinstance(gate, CCXGate):
            t += 7
        elif isinstance(gate, (CXGate, XGate)):
            pass
    return t


# ---------------------------------------------------------------- cases ----

def load_cases(args):
    cases = []
    if args.config_cases:
        for (num_var, tt, skip) in config['case_pool']:
            if skip and not args.config_all:
                continue
            cases.append((num_var, tt, 'config'))
    for n, k in [(4, args.n4), (5, args.n5), (6, args.n6), (7, args.n7)]:
        if k <= 0:
            continue
        with open(os.path.join(ROOT, f'data/test_cases/test_cases_{n}')) as f:
            lines = f.readlines()
        rng = random.Random(args.seed * 100 + n)
        sample = rng.sample(lines, min(k, len(lines)))
        for line in sample:
            tt = int(line.rstrip('\n'))
            cases.append((n, tt, 'rand'))
    seen, out = set(), []
    for c in cases:
        key = (c[0], c[1])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def fmt_case(st):
    return (f"{st['label']:<7} n={st['n']} tt={st['tt']:<24} sub={st['n_sub']:>3} br={st['n_branch']:>4} "
            f"r1={st['rule1_pairs']:>4}(x{st['rule1_cross']}) r2={st['rule2_pairs']:>4}(x{st['rule2_cross']}) "
            f"f1={st['func1_pairs']:>4} aff={st['affine_pairs']:>4} cpl={st['complete']:>4} "
            f"T: act={st['actual_t'] if st['actual_t'] is not None else '?':>4} flat={st['flat_t']:>4} fused={st['fused_t_est']:>4} "
            f"valid={int(st['valid'])}")


def main():
    ap = argparse.ArgumentParser(description='Phase 0 branch fusion opportunity statistics')
    ap.add_argument('--reorder', type=int, default=1, help='variable reordering on/off (0/1)')
    ap.add_argument('--config-cases', action='store_true', help='include config case_pool cases')
    ap.add_argument('--config-all', action='store_true', help='include cases skipped in config')
    ap.add_argument('--n4', type=int, default=2000)
    ap.add_argument('--n5', type=int, default=1000)
    ap.add_argument('--n6', type=int, default=300)
    ap.add_argument('--n7', type=int, default=150)
    ap.add_argument('--pivot-sample', type=int, default=0, help='pivot measurement pairs per subgraph (0=off)')
    ap.add_argument('--seed', type=int, default=20250815)
    ap.add_argument('--out', default=os.path.join('experiments', 'phase0_results', 'phase0_results.txt'))
    args = ap.parse_args()

    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    rng = random.Random(args.seed)
    cases = load_cases(args)
    emit(f'# Phase 0 branch fusion opportunity statistics  reorder={args.reorder}  cases={len(cases)}  '
         f'n4={args.n4} n5={args.n5} n6={args.n6} n7={args.n7} pivot_sample={args.pivot_sample}')

    anomalies = collections.Counter()
    agg = collections.defaultdict(dict)
    top_cases = []  # (affine_match_gain, n, tt)
    t0 = time.time()
    for ci, (num_var, tt, label) in enumerate(cases):
        try:
            tt_obj = TruthTable(tt, num_var)
            probe = ProbeSynthesizer(reorder=bool(args.reorder), ancilla_num=0,
                                     n=num_var, pivot_sample=args.pivot_sample, rng=rng)
            probe.synthesize(tt_obj)
            st = case_stats(num_var, tt, label, probe.recorded, anomalies, probe.pivot_records)
            st['actual_t'] = actual_pipeline_t(tt_obj, bool(args.reorder))
            top_cases.append((st['affine_match_gain'], num_var, tt))
            a = agg[num_var]
            for k in ('n_sub', 'n_branch', 'rule1_pairs', 'rule1_cross', 'rule1_gain',
                      'rule2_pairs', 'rule2_cross', 'rule2_gain', 'complete',
                      'rule3_same_pairs', 'rule3_same_gain_ub', 'rule3_d1_pairs',
                      'rule3_d1_gain', 'rule3_dge2_pairs',
                      'func1_pairs', 'func1_gain', 'affine_pairs', 'affine_gain',
                      'affine_gain_pos', 'affine_pairs_cross', 'affine_match_gain',
                      'flat_t', 'fused_t_est',
                      'actual_t', 'pivot_ok', 'pivot_fail', 'pivot_pairs',
                      'pivot_fused_t_sum', 'pivot_est_t_sum', 'pivot_base_t_sum'):
                a[k] = a.get(k, 0) + (st[k] or 0)
            a['cases'] = a.get('cases', 0) + 1
            if not st['valid']:
                a['invalid'] = a.get('invalid', 0) + 1
            h = a.setdefault('agree_hist', collections.Counter())
            for kk, v in st['rule3_agree_hist'].items():
                h[kk] += v
            h2 = a.setdefault('affine_dim_hist', collections.Counter())
            for kk, v in st['affine_dim_hist'].items():
                h2[kk] += v
            emit(f'[{ci+1}/{len(cases)}] {fmt_case(st)}')
        except Exception as e:
            anomalies['case_fail'] += 1
            emit(f'[{ci+1}/{len(cases)}] n={num_var} tt={tt} FAILED: {type(e).__name__}: {e}')

    emit('')
    emit('# ===== per-n summary =====')
    for n in sorted(agg):
        a = agg[n]
        nc = a['cases']
        emit(f'-- n={n} ({nc} cases) --')
        emit(f'  subgraphs/branches: {a["n_sub"]/nc:.1f} / {a["n_branch"]/nc:.1f} (mean per case)')
        emit(f'  syntactic rule1 identical pairs: {a["rule1_pairs"]} (cross-subgraph {a["rule1_cross"]}), gain {a["rule1_gain"]}; '
             f'rule2 single-position pairs: {a["rule2_pairs"]} (cross-subgraph {a["rule2_cross"]}), gain {a["rule2_gain"]}')
        emit(f'  function-level equal pairs:    {a["func1_pairs"]} (gain {a["func1_gain"]}); '
             f'affine-cube fusible pairs: {a["affine_pairs"]} (cross-subgraph {a["affine_pairs_cross"]}), '
             f'total gain {a["affine_gain"]} (positive gain {a["affine_gain_pos"]}), '
             f'greedy-match gain {a["affine_match_gain"]}, dim histogram {dict(sorted(a["affine_dim_hist"].items()))}')
        emit(f'  rule3 complete paths:   {a["complete"]}; same-length pairs {a["rule3_same_pairs"]}, '
             f'upper-bound gain {a["rule3_same_gain_ub"]}; length-diff-1 pairs {a["rule3_d1_pairs"]} '
             f'(gain {a["rule3_d1_gain"]}); length-diff>=2 pairs {a["rule3_dge2_pairs"]}')
        emit(f'  rule3 same-length agree-bit histogram: {dict(sorted(a["agree_hist"].items()))}')
        emit(f'  T: current pipeline Σ={a["actual_t"]} (mean {a["actual_t"]/nc:.1f}) | '
             f'flattened Σ={a["flat_t"]} (mean {a["flat_t"]/nc:.1f}) | '
             f'rule1+2 fused Σ={a["fused_t_est"]} (mean {a["fused_t_est"]/nc:.1f})')
        if a.get('pivot_pairs'):
            emit(f'  pivot measurement (same-graph function-level pairs): {a["pivot_pairs"]} pairs, baseline Σ2T={a["pivot_base_t_sum"]}, '
                 f'estimated fused Σ={a["pivot_est_t_sum"]}, measured fused Σ={a["pivot_fused_t_sum"]} '
                 f'(vs baseline {a["pivot_fused_t_sum"]/max(1,a["pivot_base_t_sum"]):.2%}), '
                 f'ok={a["pivot_ok"]}, fail={a["pivot_fail"]}')
        emit(f'  global verification failures: {a.get("invalid", 0)}')
    emit('')
    emit('# Top 10 cases by greedy-match gain:')
    for gain, n, tt in sorted(top_cases, reverse=True)[:10]:
        emit(f'  n={n} tt={tt} gain={gain}')
    emit('')
    emit(f'# anomaly counts: {dict(anomalies)}')
    emit(f'# total time {time.time()-t0:.1f}s')

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n[written to {args.out}]')


def actual_pipeline_t(tt, reorder):
    syn = OracleSynthesizer(reorder=reorder, ancilla_num=0)
    syn.synthesize(tt)
    qc = to_qiskit(syn.circ)
    return count_t_cost(qc)


if __name__ == '__main__':
    main()
