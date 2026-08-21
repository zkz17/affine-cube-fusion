class TruthTable:
    __slots__ = ('bitstr', 'num_var', 'mask')

    # variable: xn-1 | ... | x2 | x1 | x0 | result | bitstr
    #           0    | ... | 0  | 0  | 0  |   1    |  low
    #           0    | ... | 0  | 0  | 1  |   0    |   A
    #           0    | ... | 0  | 1  | 0  |   1    |   |
    #                | ... |    |    |    |        |   |
    #           1    | ... | 1  | 1  | 1  |   0    |  high
    def __init__(self, bitstr, num_var):
        self.bitstr = bitstr
        self.num_var = num_var
        self.mask = (1 << (1 << self.num_var)) - 1
        if not self.num_var: self.mask = 1
        self.bitstr &= self.mask

    def copy(self):
        return TruthTable(self.bitstr, self.num_var)

    # Split function f into f(x{index}=0) and f(x{index}=1)
    def split(self, index):
        if not self.num_var: raise Exception(f'Cannot split {str(self)} with 0 variable. ')
        assert(index >= 0 and index < self.num_var)
        
        tt0, tt1, new_num_var = 0, 0, self.num_var - 1
        window_length = (1 << index)
        window_mask   = (1 << window_length) - 1
        block_length  = (1 << (index + 1))

        tt, pos = self.bitstr, 0
        while tt:
            tt0 |= (tt & window_mask) << pos
            tt1 |= ((tt >> window_length) & window_mask) << pos

            tt >>= block_length
            pos += window_length

        return TruthTable(tt0, new_num_var), TruthTable(tt1, new_num_var)
    
    def try_neg(self):
        if self.bitstr.bit_count() > (1 << (self.num_var - 1)) and not self.is_true():
            self.neg()
            return True
        return False
    
    def neg(self):
        self.bitstr ^= self.mask
    
    def is_false(self):
        return self.bitstr == 0
    
    def is_true(self):
        return self.bitstr == self.mask
    
    def is_equal_to(self, tt_num):
        return self.bitstr == tt_num
    
    def is_complemental_to(self, tt_num):
        return self.bitstr + tt_num == self.mask
    
    def __str__(self):
        return f'TruthTable({bin(self.bitstr)[2:].zfill(1 << self.num_var)})'
    
    @staticmethod
    def distence(tt0, tt1):
        assert(isinstance(tt0, TruthTable) and isinstance(tt1, TruthTable))

        return (tt0.bitstr ^ tt1.bitstr).bit_count()