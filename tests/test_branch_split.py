import random

from utils.TruthTable import TruthTable
from limbdd.LimBDD import LimBDD
from limbdd.LimBDDManager import LimBDDManager

def test_branch_split(num_vars=6, rounds=1000, verbose=False):
    print(f'Testing split of given branches from a LimBDD with {rounds} randomly generated {num_vars}-input functions...')

    max_tt = (1 << (1 << num_vars)) - 1

    for i in range(rounds):
        bitstr = random.randint(0, max_tt)
        tt = TruthTable(bitstr, num_vars)

        limbdd = LimBDD()
        limbdd.initialize(tt, enable_neg=False)

        branch_list = LimBDDManager.branch_process(limbdd.root.node, [limbdd.terminator])[limbdd.terminator]
        if not len(branch_list): continue

        split_list = random.sample(branch_list, k=random.randint(1, len(branch_list) - 1)) if len(branch_list) > 1 else branch_list
        if verbose: print('  ', i, str(tt), [str(branch) for branch in split_list])

        split_limbdd = LimBDDManager.split_branches(limbdd, split_list)

        bitstr0 = limbdd.get_truth_table()
        bitstr1 = split_limbdd.get_truth_table()
        if not tt.bitstr == (bitstr0 ^ bitstr1):
            raise Exception(f'Error encountered when spliting {[str(branch) for branch in split_list]} from {str(tt)}')
        
    print(f'Branch split test passed!!!')
        