# -*- coding: utf-8 -*-

from config import config
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer

def main():
    case_pool = config["case_pool"]
    synthesizer = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0)
    for num_var, tt, skip in case_pool:
        if skip: continue
        truth_table = TruthTable(tt, num_var)

        synthesizer.synthesize(truth_table, path=config['figure_path_prefix'] + f'{truth_table.num_var}_{truth_table.bitstr}')
        circ = to_qiskit(synthesizer.circ)
        print(circ)
        if not synthesizer.circ.verify(truth_table):
            print(f'Verification failed for {str(truth_table)}: got {bin(synthesizer.circ.tt)[2:].zfill(1 << truth_table.num_var)}')
            print(f'Difference: {bin(synthesizer.circ.tt ^ truth_table.bitstr)[2:].zfill(1 << truth_table.num_var)}')

if __name__ == "__main__":
    main()