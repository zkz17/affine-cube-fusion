class Lim:
    __slots__ = ('bitstr', 'bitnum')

    # LimBDDNode.level: high Xn-1 ... X2 I  X0 low
    #    Lim.bitstr   : high 1    ... 1  0  1  low
    #    Lim.bitnum   : n
    def __init__(self, bitstr, bitnum):
        self.bitstr = bitstr
        self.bitnum = bitnum

    def is_identity(self):
        return self.bitstr == 0

    def act_on_direct_child(self):
        return self.bitnum >= 1 and (self.bitstr >> (self.bitnum - 1)) == 1
    
    def swap_children(self):
        self.bitstr ^= (1 << (self.bitnum - 1))
    
    def swap_levels(self, level1, level2):
        assert(level1 != level2 and level1 <= self.bitnum and level2 <= self.bitnum)

        if self.is_identity(): return

        pivot1, pivot2 = (1 << (level1 - 1)), (1 << (level2 - 1))
        bit1, bit2 = (pivot1 & self.bitstr) != 0, (pivot2 & self.bitstr) != 0
        self.bitstr -= (pivot1 * int(bit1) + pivot2 * int(bit2))
        self.bitstr += (pivot2 * int(bit1) + pivot1 * int(bit2))

    def copy(self):
        return Lim(self.bitstr, self.bitnum)
    
    def __eq__(self, other):
        assert(isinstance(other, Lim))

        return self.bitstr == other.bitstr and self.bitnum == other.bitnum

    def __str__(self):
        result = ''

        index = 0
        bitstr = self.bitstr
        while index < self.bitnum and bitstr >= 0:
            if bitstr & 1: result = f'X{index}' + result
            index += 1
            bitstr >>= 1

        if not len(result): result = 'I'
        return result

    @staticmethod
    def sum(lim1, lim2):
        assert(isinstance(lim1, Lim) and isinstance(lim2, Lim))
        if lim1.bitnum != lim2.bitnum: raise Exception(f'Cannot sum Lim operators with different masks ({lim1.bitnum} and {lim2.bitnum}).')
        return Lim(lim1.bitstr ^ lim2.bitstr, lim1.bitnum)
    
    @staticmethod
    def intersect(lim1, lim2):
        assert(isinstance(lim1, Lim) and isinstance(lim2, Lim))
        if lim1.bitnum != lim2.bitnum: raise Exception(f'Cannot intersect Lim operators with different masks ({lim1.bitnum} and {lim2.bitnum}).')
        return Lim(lim1.bitstr & lim2.bitstr, lim1.bitnum)
    
    @staticmethod
    def extend(lim, tail=False):
        assert(isinstance(lim, Lim))
        return Lim(lim.bitstr + (int(tail) << lim.bitnum), lim.bitnum + 1)

    @staticmethod
    def truncate(lim, bitnum):
        assert(isinstance(lim, Lim) and bitnum <= lim.bitnum)
        return Lim(lim.bitstr >> (lim.bitnum - bitnum), bitnum)

    @staticmethod
    def downgrade(lim):
        assert(isinstance(lim, Lim) and 1 <= lim.bitnum)

        bitnum = lim.bitnum - 1
        bitstr = lim.bitstr
        pivot = 1 << bitnum
        if bitstr & pivot: bitstr -= pivot
        return Lim(bitstr, bitnum)