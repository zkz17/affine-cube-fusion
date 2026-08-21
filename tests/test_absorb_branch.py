import random

from utils.TruthTable import TruthTable
from limbdd.LimBDD import LimBDD
from limbdd.LimBDDManager import LimBDDManager

def test_absorb_branch(num_vars=6, rounds=1000, verbose=False):
    print(f'Testing absorbing another LimBDD with {rounds} randomly generated {num_vars}-input functions...')

    max_tt = (1 << (1 << num_vars)) - 1

    for i in range(rounds):
        bitstr0, bitstr1 = random.randint(0, max_tt), random.randint(0, max_tt)
        tt0, tt1 = TruthTable(bitstr0, num_vars), TruthTable(bitstr1, num_vars)
        if verbose: print(f'{str(tt0), str(tt1)}')

        limbdd0 = LimBDD()
        limbdd0.initialize(tt0, enable_neg=False)
        limbdd1 = LimBDD()
        limbdd1.initialize(tt1, enable_neg=False)

        tt_xor_old = limbdd0.get_truth_table() ^ limbdd1.get_truth_table()
        LimBDDManager.try_absorb_branch(limbdd0, limbdd1)
        tt_xor_new = limbdd0.get_truth_table() ^ limbdd1.get_truth_table()

        if tt_xor_old != tt_xor_new:
            raise Exception(f'Error encountered when absorbing {str(tt1)} into {str(tt0)}, expect: {bin(tt_xor_old)[2:].zfill(1 << num_vars)}, got {str(bin(tt_xor_new)[2:].zfill(1 << num_vars))}')
        
    print(f'Branch absorb test passed!!!')
        