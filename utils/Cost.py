class Cost:
    __slots__ = ('length', 'num')

    def __init__(self, length, num):
        self.length = length
        self.num = num

    def increment(self, delta=1):
        self.length += delta

    def copy(self):
        return Cost(self.length, self.num)
    
    def __eq__(self, other):
        assert(isinstance(other, Cost))

        return self.length == other.length and self.num == other.num
    
    def __lt__(self, other):
        assert(isinstance(other, Cost))

        return self.length < other.length or (self.length == other.length and self.num < other.num)
    
    def __str__(self):
        return f'Cost(length={self.length}, num={self.num})'

    @staticmethod
    def meet(cost0, cost1):
        assert((isinstance(cost0, Cost) or cost0 is None) and (isinstance(cost1, Cost) or cost1 is None))

        if not cost0 and not cost1: return None
        elif not cost0: return cost1.copy()
        elif not cost1: return cost0.copy()

        if cost0.length > cost1.length: return cost0.copy()
        elif cost0.length < cost1.length: return cost1.copy()
        else: return Cost(cost0.length + 1, cost0.num + cost1.num)