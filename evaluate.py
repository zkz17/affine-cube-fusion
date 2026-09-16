import os
import sys
import time
from qiskit.circuit.library.standard_gates.x import XGate, CXGate, MCXGate, CCXGate

from config import config
from backend.OracleSynthesizer import OracleSynthesizer
from backend.fusion.BranchFusion import anf_degree
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit

T_COST_FN = config['t_cost']
CX_COST_FN = config['cx_cost']

# All supported synthesis routes, selected by a single `path` argument.
# A/C/F are the BDD/ANF/FPRM routes, FP is F with the proxy prescreen, G the GL search,
# MT/PK the engine routes over the minterm / pseudo-Kronecker covers, PKS/PKP their
# shift-search variants, and MT0/PK0 the raw covers emitted without the fusion engine.
ROUTES = ('A', 'C', 'F', 'FP', 'G', 'MT', 'PK', 'PKS', 'PKP', 'MT0', 'PK0')
ENGINE_ROUTES = ('A', 'C', 'F', 'FP', 'G', 'MT', 'PK', 'PKS', 'PKP')
COVER_ROUTES = ('MT', 'PK', 'PKS', 'PKP', 'MT0', 'PK0')


def count_costs(circ):
    """
    Circuit (T, CX) cost. The scope matches data/exorcism_cost.py / sshr data:
      MCX/CCX are charged T_COST_FN(k) and CX_COST_FN(k) by their control count k;
      an explicit CX gate counts as 1 CX; X gates are free.
    """
    t = cx = 0
    for instruction in circ.data:
        gate = instruction.operation

        if isinstance(gate, MCXGate):
            t += T_COST_FN(gate.num_ctrl_qubits)
            cx += CX_COST_FN(gate.num_ctrl_qubits)
        elif isinstance(gate, CCXGate):
            t += T_COST_FN(2)
            cx += CX_COST_FN(2)
        elif isinstance(gate, CXGate):
            cx += 1
        elif isinstance(gate, XGate):
            pass
        else:
            print(f'Unrecognized gate type {type(gate)}.')
    return t, cx


def count_t_cost(circ):
    """T-only scope (compatible with existing callers)."""
    return count_costs(circ)[0]


def count_cx_cost(circ):
    """CX-only scope."""
    return count_costs(circ)[1]


def load_aligned(cases_path, cost_path, start, stop):
    """
    Read test cases and per-case baseline costs (aligned by line number).
    Returns (cases, costs); costs=None when cost_path is missing or empty (e.g. sshr left empty for n≥7).
    """
    with open(cases_path) as f:
        lines = f.readlines()
    stop = stop if stop <= len(lines) else len(lines)
    cases = [int(line.rstrip('\n')) for line in lines[start:stop]]

    try:
        with open(cost_path) as f:
            clines = f.readlines()
    except FileNotFoundError:
        return cases, None
    if stop > len(clines):
        return cases, None
    costs = [int(line.split(':')[1]) for line in clines[start:stop]]
    return cases, costs


def _compare(name, cases, ours, baseline):
    """Single-scope comparison; skipped when baseline is None. Returns a list of report lines."""
    if baseline is None:
        return [f"===== {name}:baseline missing, skipped ====="]
    better, worse, same = 0, 0, 0
    worse_cases = []
    s_base = s_ours = 0
    for i, tt in enumerate(cases):
        t_ours = ours[i]
        t_base = baseline[i]
        s_base += t_base
        s_ours += t_ours
        if t_ours > t_base:
            worse += 1
            worse_cases.append(tt)
        elif t_ours == t_base:
            same += 1
        else:
            better += 1
    return [
        f"===== {name} =====",
        f"lower  cost in {better}({better / len(cases) * 100}%) cases",
        f"higher cost in {worse}({worse / len(cases) * 100}%) cases",
        f"same   cost in {same}({same / len(cases) * 100}%) cases",
        f"cases with higher cost: {worse_cases[:min(len(worse_cases), 20)]}",
        f"total cost: baseline {s_base}, ours {s_ours} "
        f"(reduction {1 - s_ours / max(1, s_base):.2%})",
    ]


def evaluate_all(num_var=3, start=0, stop=None, path='F', m2_post=True, reorder=True):
    """
    Comparison of the fused backend against Exorcism / SSHR (T and CX dual scopes).

    Usage: evaluate.py <n> <start> <stop> <path> [m2] [reorder]

    path    : the synthesis route, one of A / C / F / FP / G / MT / PK / PKS / PKP / MT0 / PK0.
              A = BDD decomposition, C = plain ANF, F = FPRM (shift search), FP = F with the
              proxy prescreen, G = deterministic GL(n,2) search, MT/PK = minterm /
              pseudo-Kronecker covers fused by the engine, PKS/PKP = their shift-search
              variants, MT0/PK0 = the raw covers emitted without the fusion engine (baselines).
    m2      : macro M2 interleaved post-processing (1 = on, default; 0 = off, tagged `_noM2`).
              Ignored for MT0/PK0, which skip the engine entirely.
    reorder : variable reordering for the BDD route A only (1 = on, default).

    Outputs are written to `data/our_cost/{tag}/our_{t,cx}_cost_{tag}_{n}_{a}_{b}.txt` and
    `data/our_cost/{tag}/our_results_{tag}_{n}_{a}_{b}.txt`. The G-3 degree lower bound is
    reported inside the results file but no longer written as a separate file (it depends
    only on the test set, not on the route).
    """
    assert path in ROUTES, f'illegal route: {path}'
    cases_path = f'data/test_cases/test_cases_{num_var}'
    with open(cases_path) as f:
        n_total = len(f.readlines())
    stop = stop if stop is not None and stop <= n_total else n_total
    total = stop - start

    cases, exor_t = load_aligned(cases_path, f'data/exorcism_cost/exorcism_t_cost_{num_var}', start, stop)
    _, exor_cx = load_aligned(cases_path, f'data/exorcism_cost/exorcism_cx_cost_{num_var}', start, stop)
    _, sshr_t = load_aligned(cases_path, f'data/sshr_cost/sshr_t_cost_{num_var}', start, stop)
    _, sshr_cx = load_aligned(cases_path, f'data/sshr_cost/sshr_cx_cost_{num_var}', start, stop)

    tag = path if path in ('MT0', 'PK0') else path + ('' if m2_post else '_noM2')
    outdir = f'data/our_cost/{tag}'
    os.makedirs(outdir, exist_ok=True)
    out_t = open(f'{outdir}/our_t_cost_{tag}_{num_var}_{start}_{stop}.txt', 'w')
    out_cx = open(f'{outdir}/our_cx_cost_{tag}_{num_var}_{start}_{stop}.txt', 'w')

    if path == 'G':
        synthesizer = OracleSynthesizer(fused_backend=True, gl_search=True, m2_post=m2_post)
    else:
        kwargs = dict(reorder=reorder, fused_backend=True, m2_post=m2_post)
        if path in ('A', 'C', 'F', 'FP'):
            kwargs['paths'] = ('A',) if path == 'A' else ('C',) if path == 'C' else ('F',)
            if path == 'FP':
                kwargs['fprm_proxy'] = True
        synthesizer = OracleSynthesizer(**kwargs)
    cover = path if path in COVER_ROUTES else None

    ours_t, ours_cx, bounds = [], [], []
    wall_ns = 0
    for i, tt in enumerate(cases):
        if i % 1000 == 0:
            print(i, flush=True)
        truth_table = TruthTable(tt, num_var)
        t0 = time.perf_counter_ns()
        if cover:
            synthesizer.clear()
            synthesizer.allocate_input_qubit(num_var)
            synthesizer.process_cover_route(truth_table, cover)
        else:
            synthesizer.synthesize(truth_table)
        wall_ns += time.perf_counter_ns() - t0
        t, cx = count_costs(to_qiskit(synthesizer.circ))
        bound = T_COST_FN(anf_degree(truth_table.bitstr, num_var))

        out_t.write(f'{tt}:{t}\n')
        out_cx.write(f'{tt}:{cx}\n')
        ours_t.append(t)
        ours_cx.append(cx)
        bounds.append(bound)

    out_t.close()
    out_cx.close()

    avg_wall_ms = wall_ns / max(total, 1) / 1e6
    report = [f"****** Comparison of TDD-based Method Alg. 1 (fused backend) ******",
              f"number of input variables: {num_var} ({total} cases); path={tag}",
              f"avg synthesize wall: {avg_wall_ms:.2f} ms/case"]
    report += _compare("Exorcism T", cases, ours_t, exor_t)
    report += _compare("Exorcism CX", cases, ours_cx, exor_cx)
    report += _compare("SSHR T", cases, ours_t, sshr_t)
    report += _compare("SSHR CX", cases, ours_cx, sshr_cx)
    report.append(f"G-3 lower bound: Σ t(deg f) = {sum(bounds)} "
                  f"({sum(bounds) / max(1, sum(ours_t)):.2%} of ours ΣT)")
    report.append("****** ****** ******")

    text = '\n'.join(report) + '\n'
    with open(f'{outdir}/our_results_{tag}_{num_var}_{start}_{stop}.txt', 'w') as f:
        f.write(text)
    print(text, end='')


def main():
    num_var = int(sys.argv[1]) if len(sys.argv) >= 2 else 3
    start   = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    stop    = int(sys.argv[3]) if len(sys.argv) >= 4 else None
    path    = sys.argv[4] if len(sys.argv) >= 5 else 'F'
    m2      = int(sys.argv[5]) if len(sys.argv) >= 6 else 1
    reorder = int(sys.argv[6]) if len(sys.argv) >= 7 else 1
    assert path in ROUTES, f'illegal route: {path}'
    assert m2 in (0, 1), f'illegal m2 switch (0/1): {m2}'

    evaluate_all(num_var, start, stop, path, m2_post=m2 > 0, reorder=reorder > 0)


if __name__ == "__main__":
    main()
