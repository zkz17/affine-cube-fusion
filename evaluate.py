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


def paths_tag(paths, fprm_proxy, gl_search):
    """Path label (used in output file names): gl → 'G'; single path A/C/F; fprm_proxy with F → 'FP'."""
    if gl_search:
        return 'G'
    p = paths[0] if isinstance(paths, (tuple, list)) else paths
    assert p in ('A', 'C', 'F'), f'illegal paths (only the single paths A/C/F are allowed at this stage): {paths}'
    return 'FP' if (fprm_proxy and p == 'F') else p


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


def evaluate_all(num_var=3, reorder=False, num_anc=0, start=0, stop=None, gl_search=False,
                 fprm_proxy=False, paths=('F',), m2_post=True):
    """
    Comparison of the fused backend against Exorcism / SSHR (T and CX dual scopes).

    gl_search=True   : enable the G-2.2 deterministic GL search (time for quality, about +6s/case at n=7).
    fprm_proxy=True  : Plan A — the F path is **replaced** by the fusion-free proxy-prescreened version
                       (a choice relationship, not a min; for n≥9 the intent is to upgrade to shift climb2).
    paths            : single-path choice (exploration-phase discipline, default 'F'; only 'A'/'C'/'F'
                       are allowed, multi-path combinations are disabled at this stage; FP/G are selected via flags).
    m2_post=True     : macro M2 interleaved post-processing (**enabled by default**; zero regression on the T side
                       case by case, 0 CX degradations across all n≤7 test cases).
    Output files are named by path combination: `data/our_cost/our_{t,cx}_cost_{tag}_{n}_{a}_{b}.txt`,
    tag = paths_tag(paths, fprm_proxy, gl_search). The default (m2_post=True) keeps the plain path tag;
    m2_post=False (the no-macro-M2 comparison runs) appends the `_noM2` suffix to the tag so its outputs
    never overwrite the default runs.

    Data layout (updated 2025-08-22):
      test cases  data/test_cases/test_cases_{n} (one truth-table integer per line)
      baselines   data/exorcism_cost/exorcism_{t,cx}_cost_{n}, data/sshr_cost/sshr_{t,cx}_cost_{n}
      (aligned with test cases by line number; sshr n≥7 is temporarily empty → that comparison is skipped automatically)
    Outputs:
      data/our_cost/our_{t,cx}_cost_{n}_{start}_{stop}.txt,
      data/our_cost/our_deg_bound_{n}_{start}_{stop}.txt (G-3 lower bound)
    """
    cases_path = f'data/test_cases/test_cases_{num_var}'
    with open(cases_path) as f:
        n_total = len(f.readlines())
    stop = stop if stop is not None and stop <= n_total else n_total
    total = stop - start

    cases, exor_t = load_aligned(cases_path, f'data/exorcism_cost/exorcism_t_cost_{num_var}', start, stop)
    _, exor_cx = load_aligned(cases_path, f'data/exorcism_cost/exorcism_cx_cost_{num_var}', start, stop)
    _, sshr_t = load_aligned(cases_path, f'data/sshr_cost/sshr_t_cost_{num_var}', start, stop)
    _, sshr_cx = load_aligned(cases_path, f'data/sshr_cost/sshr_cx_cost_{num_var}', start, stop)

    synthesizer = OracleSynthesizer(reorder=reorder, ancilla_num=num_anc,
                                          fused_backend=True, gl_search=gl_search,
                                          fprm_proxy=fprm_proxy, paths=paths,
                                          m2_post=m2_post)
    tag = paths_tag(paths, fprm_proxy, gl_search) + ('' if m2_post else '_noM2')

    out_t = open(f'data/our_cost/our_t_cost_{tag}_{num_var}_{start}_{stop}.txt', 'w')
    out_cx = open(f'data/our_cost/our_cx_cost_{tag}_{num_var}_{start}_{stop}.txt', 'w')
    out_bound = open(f'data/our_cost/our_deg_bound_{tag}_{num_var}_{start}_{stop}.txt', 'w')

    ours_t, ours_cx, bounds = [], [], []
    wall_ns = 0
    for i, tt in enumerate(cases):
        if i % 1000 == 0:
            print(i, flush=True)
        truth_table = TruthTable(tt, num_var)
        t0 = time.perf_counter_ns()
        synthesizer.synthesize(truth_table)
        wall_ns += time.perf_counter_ns() - t0
        t, cx = count_costs(to_qiskit(synthesizer.circ))
        bound = T_COST_FN(anf_degree(truth_table.bitstr, num_var))

        out_t.write(f'{tt}:{t}\n')
        out_cx.write(f'{tt}:{cx}\n')
        out_bound.write(f'{tt}:{bound}\n')
        ours_t.append(t)
        ours_cx.append(cx)
        bounds.append(bound)

    out_t.close()
    out_cx.close()
    out_bound.close()

    avg_wall_ms = wall_ns / max(total, 1) / 1e6
    report = [f"****** Comparison of TDD-based Method Alg. 1 (fused backend) ******",
              f"number of input variables: {num_var} ({total} cases); paths={tag}",
              f"avg synthesize wall: {avg_wall_ms:.2f} ms/case"]
    report += _compare("Exorcism T", cases, ours_t, exor_t)
    report += _compare("Exorcism CX", cases, ours_cx, exor_cx)
    report += _compare("SSHR T", cases, ours_t, sshr_t)
    report += _compare("SSHR CX", cases, ours_cx, sshr_cx)
    report.append(f"G-3 lower bound: Σ t(deg f) = {sum(bounds)} "
                  f"({sum(bounds) / max(1, sum(ours_t)):.2%} of ours ΣT)")
    report.append("****** ****** ******")

    text = '\n'.join(report) + '\n'
    with open(f'data/our_cost/our_results_{tag}_{num_var}_{start}_{stop}.txt', 'w') as f:
        f.write(text)
    print(text, end='')


def main():
    num_var = int(sys.argv[1]) if len(sys.argv) >= 2 else 3
    reorder = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
    num_anc = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    start   = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    stop    = int(sys.argv[5]) if len(sys.argv) >= 6 else None
    gl      = int(sys.argv[6]) if len(sys.argv) >= 7 else 0
    proxy   = int(sys.argv[7]) if len(sys.argv) >= 8 else 0
    paths   = sys.argv[8] if len(sys.argv) >= 9 else 'F'
    m2      = int(sys.argv[9]) if len(sys.argv) >= 10 else 1
    assert paths in ('A', 'C', 'F'), f'illegal paths (only single paths A/C/F allowed): {paths}'
    assert m2 in (0, 1), f'illegal m2 switch (0/1): {m2}'
    # m2=1 (default) writes the plain path tag; m2=0 writes the `_noM2` suffix so the
    # no-macro-M2 comparison runs never overwrite the default runs.

    if num_anc != 0:
        print('Warning: fused_backend only takes effect when ancilla_num=0 (currently anc>0, the old path will be used).')

    evaluate_all(num_var, reorder > 0, num_anc, start, stop, gl_search=gl > 0,
                 fprm_proxy=proxy > 0, paths=paths, m2_post=m2 > 0)


if __name__ == "__main__":
    main()
