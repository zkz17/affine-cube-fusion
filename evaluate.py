import sys
from qiskit.circuit.library.standard_gates.x import XGate, CXGate, MCXGate, CCXGate

from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit

t_cost_list =[0,0,7,16,24,62,80,200,216,272,392,448,504,560,616,672,728]
#t_cost_list =[0,0,7,16,24,32,40,48,56,64,72,80,88,96,104,112,120]

def count_t_cost(circ):
    t_cost = 0
    for instruction in circ.data:
        gate = instruction.operation

        if isinstance(gate, MCXGate):
            t_cost += t_cost_list[gate.num_ctrl_qubits]
        elif isinstance(gate, CCXGate):
            t_cost += 7
        elif isinstance(gate, CXGate):
            pass
        elif isinstance(gate, XGate):
            pass
        else:
            print(f'Unrecognized gate type {type(gate)}.')
    
    return t_cost

def evaluate_all(num_var=3, reorder=False, num_anc=0, start=0, stop=80000):
    """
    融合后端（默认 min(BDD, ANF)）与 Exorcism 的对比：
      - 逐例 better/worse/same 统计；
      - 总 T 门数量（ΣT）对比。
    逐例融合后端 T 写入 data/our_rand_tcost_fused_{num_var}_{start}_{stop}.txt。
    """
    lines = []
    with open(f'data/exorcism_rand_tcost_{num_var}.txt', 'r') as f:
        lines = f.readlines()

    synthesizer = LimBDDOracleSynthesizer(reorder=reorder, ancilla_num=num_anc, fused_backend=True)

    better, worse, eqv = 0, 0, 0
    worse_cases = []
    t_exor_sum, t_fused_sum = 0, 0

    stop = stop if stop <= len(lines) else len(lines)
    total = stop - start

    out = open(f'data/our_rand_tcost_fused_{num_var}_{start}_{stop}.txt', 'w')

    for counter in range(start, stop):
        if counter % 1000 == 0:
            print(counter, flush=True)
        tt = int(lines[counter].split(":")[0])
        tcost_exor = int(lines[counter].split(":")[1])

        truth_table = TruthTable(tt, num_var)
        synthesizer.synthesize(truth_table)
        t_cost = count_t_cost(to_qiskit(synthesizer.circ))
        out.write(f'{tt}:{t_cost}\n')

        t_exor_sum += tcost_exor
        t_fused_sum += t_cost
        if t_cost > tcost_exor: 
            worse += 1
            worse_cases.append(tt)
        elif t_cost == tcost_exor: eqv += 1
        else: better += 1

    out.close()
    print("****** Comparison of TDD-based Method Alg. 1 (fused backend, min(BDD,ANF)) to Exorcism  ******")
    print(f"number of input variables: {num_var}")
    print(f"lower  t-cost in {better}({better / total * 100}%) cases")
    print(f"higher t-cost in {worse}({worse / total * 100}%) cases")
    print(f"same   t-cost in {eqv}({eqv / total * 100}%) cases")
    print(f"cases with higher t-cost: {worse_cases[:min(len(worse_cases), 20)]}")
    print(f"total t-cost: exorcism {t_exor_sum}, fused {t_fused_sum} "
          f"(ΣT 缩减 {1 - t_fused_sum / max(1, t_exor_sum):.2%})")
    print("****** ****** ******")

def main():
    num_var = int(sys.argv[1]) if len(sys.argv) >= 2 else 3
    reorder = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
    num_anc = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    start   = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    stop    = int(sys.argv[5]) if len(sys.argv) >= 6 else 80000

    if num_anc != 0:
        print('Warning: fused_backend 仅在 ancilla_num=0 时生效（当前 anc>0，将走旧路径）。')

    evaluate_all(num_var, reorder > 0, num_anc, start, stop)

if __name__ == "__main__":
    main()
