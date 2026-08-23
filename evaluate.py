import sys
import time
from qiskit.circuit.library.standard_gates.x import XGate, CXGate, MCXGate, CCXGate

from config import config
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.fusion.BranchFusion import anf_degree
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit

T_COST_FN = config['t_cost']
CX_COST_FN = config['cx_cost']


def count_costs(circ):
    """
    电路 (T, CX) 代价。口径与 data/exorcism_cost.py / sshr 数据一致:
      MCX/CCX 按其控制数 k 计 T_COST_FN(k) 与 CX_COST_FN(k);
      显式 CX 门计 1 个 CX;X 门免费。
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
    """仅 T 口径（兼容既有调用方）。"""
    return count_costs(circ)[0]


def count_cx_cost(circ):
    """仅 CX 口径。"""
    return count_costs(circ)[1]


def paths_tag(paths, fprm_proxy, gl_search):
    """路径标签(输出文件名用):gl → 'G';单路径 A/C/F;fprm_proxy 且 F → 'FP'。"""
    if gl_search:
        return 'G'
    p = paths[0] if isinstance(paths, (tuple, list)) else paths
    assert p in ('A', 'C', 'F'), f'非法 paths(当前阶段仅允许单路径 A/C/F): {paths}'
    return 'FP' if (fprm_proxy and p == 'F') else p


def load_aligned(cases_path, cost_path, start, stop):
    """
    读测试用例与逐例基线代价（按行号对齐）。
    返回 (cases, costs);cost_path 不存在或为空时 costs=None（如 sshr n≥7 留空）。
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
    """单口径对比;baseline 为 None 时跳过。返回报告行列表。"""
    if baseline is None:
        return [f"===== {name}:基线缺失,跳过 ====="]
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
        f"(缩减 {1 - s_ours / max(1, s_base):.2%})",
    ]


def evaluate_all(num_var=3, reorder=False, num_anc=0, start=0, stop=None, gl_search=False,
                 fprm_proxy=False, paths=('F',)):
    """
    融合后端与 Exorcism / SSHR 的对比（T 与 CX 双口径）。

    gl_search=True   : 启用 G-2.2 确定性 GL 搜索（时间换质量,n=7 约 +6s/例）。
    fprm_proxy=True  : 方案 A——F 路径由免融合代理预筛版**取代**(选择关系,
                       非 min;n≥9 意图升级平移 climb2)。
    paths            : 单路径选择(探索期纪律,默认 'F';仅允许 'A'/'C'/'F',
                       多路径组合在本阶段禁用;FP/G 经开关选择)。
    输出文件按路径组合分目录名:`data/our_cost/our_{t,cx}_cost_{tag}_{n}_{a}_{b}.txt`,
    tag = paths_tag(paths, fprm_proxy, gl_search)。

    数据布局（2025-08-22 更新）：
      测例     data/test_cases/test_cases_{n}（每行一个真值表整数）
      基线     data/exorcism_cost/exorcism_{t,cx}_cost_{n}、data/sshr_cost/sshr_{t,cx}_cost_{n}
      （按行号与测例对齐;sshr n≥7 暂空 → 自动跳过该对比）
    输出:
      data/our_cost/our_{t,cx}_cost_{n}_{start}_{stop}.txt、
      data/our_cost/our_deg_bound_{n}_{start}_{stop}.txt（G-3 下界）
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

    synthesizer = LimBDDOracleSynthesizer(reorder=reorder, ancilla_num=num_anc,
                                          fused_backend=True, gl_search=gl_search,
                                          fprm_proxy=fprm_proxy, paths=paths)
    tag = paths_tag(paths, fprm_proxy, gl_search)

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
              f"avg synthesize wall: {avg_wall_ms:.2f} ms/例"]
    report += _compare("Exorcism T", cases, ours_t, exor_t)
    report += _compare("Exorcism CX", cases, ours_cx, exor_cx)
    report += _compare("SSHR T", cases, ours_t, sshr_t)
    report += _compare("SSHR CX", cases, ours_cx, sshr_cx)
    report.append(f"G-3 下界: Σ t(deg f) = {sum(bounds)} "
                  f"(为 ours ΣT 的 {sum(bounds) / max(1, sum(ours_t)):.2%})")
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
    assert paths in ('A', 'C', 'F'), f'非法 paths(仅允许 A/C/F 单路径): {paths}'

    if num_anc != 0:
        print('Warning: fused_backend 仅在 ancilla_num=0 时生效（当前 anc>0，将走旧路径）。')

    evaluate_all(num_var, reorder > 0, num_anc, start, stop, gl_search=gl > 0,
                 fprm_proxy=proxy > 0, paths=paths)


if __name__ == "__main__":
    main()
