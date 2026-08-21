from random import randint

from utils.TruthTable import TruthTable
from limbdd.LimBDD import LimBDD

def test_fuse(num_vars=6, rounds=1000, verbose=False):
    print(f'Testing LimBDD fusion with {rounds} pairs of randomly generated {num_vars}-input functions...')

    max_tt = (1 << (1 << num_vars))

    for i in range(rounds):
        bitstr0, bitstr1 = randint(0, max_tt), randint(0, max_tt)
        tt0, tt1 = TruthTable(bitstr0, num_vars), TruthTable(bitstr1, num_vars)
        if verbose: print('  ', i, str(tt0), str(tt1))

        limbdd0 = LimBDD()
        limbdd0.initialize(tt0, enable_neg=False)

        limbdd1 = LimBDD()
        limbdd1.initialize(tt1, enable_neg=False)

        fuse_limbdd = limbdd0.fuse(limbdd1)

        bitstr = fuse_limbdd.get_truth_table()
        tt = TruthTable(bitstr, num_vars)
        if not tt.bitstr == (tt0.bitstr ^ tt1.bitstr):
            raise Exception(f'Error encountered when fusing {str(tt0)} and {str(tt1)}: expect {bin(tt0.bitstr ^ tt1.bitstr)[2:].zfill(1 << num_vars)}, got {str(tt)}')
        
    print(f'Fusion test passed!!!')
        