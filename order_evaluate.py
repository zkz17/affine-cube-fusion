from qiskit.circuit.library.standard_gates.x import XGate, CXGate, MCXGate, CCXGate

from config import config
from backend.OracleSynthesizer import OracleSynthesizer
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit

T_COST_FN = config['t_cost']

def count_t_cost(circ):
    t_cost = 0
    for instruction in circ.data:
        gate = instruction.operation

        if isinstance(gate, MCXGate):
            t_cost += T_COST_FN(gate.num_ctrl_qubits)
        elif isinstance(gate, CCXGate):
            t_cost += T_COST_FN(2)
        elif isinstance(gate, CXGate):
            pass
        elif isinstance(gate, XGate):
            pass
        else:
            print(f'Unrecognized gate type {type(gate)}.')
    
    return t_cost

def evaluate_order(num_var=3):
    lines = []
    with open(f'data/test_cases/test_cases_{num_var}', 'r') as f:
        lines = f.readlines()

    # Preserve this historical one-ancilla DD experiment explicitly; the public
    # constructor now defaults to zero-ancilla FS, where DD reordering has no effect.
    synthesizer_reorder = OracleSynthesizer(reorder=True, ancilla_num=1)
    synthesizer_noorder = OracleSynthesizer(reorder=False, ancilla_num=1)
    better, worse, eqv = 0, 0, 0
    worse_cases = []
    for counter in range(len(lines)):
        print(counter)
        tt = int(lines[counter].rstrip('\n'))

        truth_table = TruthTable(tt, num_var)
        synthesizer_reorder.synthesize(truth_table)
        synthesizer_noorder.synthesize(truth_table)
        qiskit_circ_reorder = to_qiskit(synthesizer_reorder.circ)
        qiskit_circ_noorder = to_qiskit(synthesizer_noorder.circ)
        t_cost_reorder = count_t_cost(qiskit_circ_reorder)
        t_cost_noorder = count_t_cost(qiskit_circ_noorder)

        if t_cost_reorder > t_cost_noorder: 
            worse += 1
            worse_cases.append(tt)
        elif t_cost_reorder == t_cost_noorder: eqv += 1
        else: better += 1

    print("****** Legacy DD variable reordering comparison (one ancilla) ******")
    print(f"number of input variables: {num_var}")
    print(f"lower  t-cost in {better}({better / len(lines) * 100}%) cases")
    print(f"higher t-cost in {worse}({worse / len(lines) * 100}%) cases")
    print(f"same   t-cost in {eqv}({eqv / len(lines) * 100}%) cases")
    print(f"cases with higher t-cost: {worse_cases[:min(len(worse_cases), 20)]}")
    print("****** ******  ******")

def main():
    #evaluate_order(2)
    #evaluate_order(3)
    evaluate_order(4)
    #evaluate_order(5)
    #evaluate_order(6)

if __name__ == "__main__":
    main()
