import sys

from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit

def test_synthesizer(num_var = 3, reorder = False, num_anc = 0):
    fail_list = []
    synthesizer = LimBDDOracleSynthesizer(reorder=reorder, ancilla_num=num_anc)
    max_tt = min(1 << (1 << num_var), 65536)
    for tt in range(max_tt):
        print(tt)
        truth_table = TruthTable(tt, num_var)
        synthesizer.synthesize(truth_table)
        if not synthesizer.circ.verify(truth_table):
            fail_list.append(tt)
            if len(fail_list) >= 1: break
    if len(fail_list): print(f'Failed to pass synthesizer test (see cases {fail_list}).')
    else: print('Synthesizer test successfully passed!!!')

def test_rand_input(num_var = 5, reorder = False, num_anc = 0, start=0, stop=80000):
    fail_list = []

    lines = []
    with open(f'data/test_cases/test_cases_{num_var}', 'r') as f:
        lines = f.readlines()
    
    stop = stop if stop <= len(lines) else len(lines)
    for counter in range(start, stop):
        tt = int(lines[counter].rstrip('\n'))
        print(counter, tt)
        
        truth_table = TruthTable(tt, num_var)
        synthesizer = LimBDDOracleSynthesizer(reorder=reorder, ancilla_num=num_anc)
        synthesizer.synthesize(truth_table)
        if not synthesizer.circ.verify(truth_table):
            fail_list.append(tt)
            print(to_qiskit(synthesizer.circ))
            print(f'Expect {str(truth_table)}, got {bin(synthesizer.circ.tt)[2:].zfill(1 << truth_table.num_var)}')
            if len(fail_list) >= 1: break
    if len(fail_list): print(f'Failed to pass synthesizer test (see cases {fail_list}).')
    else: print('Synthesizer test successfully passed!!!')

def main():
    num_var = int(sys.argv[1]) if len(sys.argv) >= 2 else 3
    reorder = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
    num_anc = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    start   = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    stop    = int(sys.argv[5]) if len(sys.argv) >= 6 else 80000

    if num_var <= 4: test_synthesizer(num_var, reorder > 0, num_anc)
    else: test_rand_input(num_var, reorder > 0, num_anc, start, stop)

if __name__ == "__main__":
    main()