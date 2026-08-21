from limbdd.Lim import Lim

class LimBDDNode:
    __slots__ = ('idx', 'level', 'var_index', 'parents', 'bit_count', 'node_type', 'successors', '_topo')

    def __init__(self, idx, level=0, low=None, high=None, var_index=-1):
        self.idx = idx
        self.level = level
        self.var_index = var_index
        self.parents   = {}    # use dict to guarantee iteration order, good for debugging
        self._topo = None      # cached topo signature, invalidated on structural changes

        if not low: self.bit_count = 1
        else: self.bit_count = low.weight * low.bit_count + high.weight * high.bit_count

        if not low:self.node_type = 0
        elif not low.node is high.node and low.weight and high.weight: self.node_type = 1
        elif low.node is high.node and low.weight and high.weight: self.node_type = 2
        else: self.node_type = 3
        
        self.successors = [low, high]

    @property
    def references(self):
        ref_set = {}
        for parent in self.parents: # Note: The root node's reference set is empty but is referenced by the root signal.
            for parent_successor in parent.successors:
                    if parent_successor.node is self and parent_successor.weight: ref_set[parent_successor] = 0
        return ref_set

    @property
    def low(self):
        return self.successors[0]

    @property
    def high(self):
        return self.successors[1]
    
    @property
    def is_branch_node(self):
        return self.node_type == 1
    
    @property
    def is_fallthrough_node(self):
        return self.node_type == 2
    
    @property
    def is_onechild_node(self):
        return self.node_type == 3

    @property
    def low_key(self):
        return self.low.weight * self.low.node.idx
    
    @property
    def high_key(self):
        return self.high.weight * self.high.node.idx

    @property
    def topo(self):
        topo = self._topo
        if topo is None:
            low_key, high_key = -1, -1
            low_neg, high_neg = 1, 1
            lim_sum = 0
            if self.low:
                low_key  = self.low_key
                high_key = self.high_key
                low_neg  = -1 if self.low.neg else 1
                high_neg = -1 if self.high.neg else 1
                if low_key > high_key:
                    low_key, high_key = high_key, low_key
                lim_sum = (self.low.lim.bitstr ^ self.high.lim.bitstr) * self.low.weight * self.high.weight
            topo = self._topo = (low_key, high_key, low_neg * high_neg, lim_sum, self.level)
        return topo

    def invalidate_topo(self):
        self._topo = None
    
    @property
    def is_symmetric(self):
        return self.is_fallthrough_node and self.low.lim.bitstr == self.high.lim.bitstr
    
    def swap_children(self):
        self.successors = [self.high, self.low]
        self.invalidate_topo()

    def copy(self):
        # TODO
        pass

    def remove_parent(self, parent):
        assert(isinstance(parent, LimBDDNode))
        
        if parent in self.parents: self.parents.pop(parent)

    def add_parent(self, parent):
        assert(isinstance(parent, LimBDDNode))

        self.parents[parent] = 0

    def update_parents(self):
        if self.low:
            if self.low.weight:
                self.low.node.add_parent(self)
                self.low.node.update_parents()
            if self.high.weight:
                self.high.node.add_parent(self)
                self.high.node.update_parents()

    def recompute_bitcount(self):
        old_bit_count = self.bit_count

        if not self.low: self.bit_count = 1
        else: self.bit_count = self.low.weight * self.low.bit_count + self.high.weight * self.high.bit_count

        if old_bit_count != self.bit_count:
            for parent in self.parents: parent.recompute_bitcount()

    def recompute_nodetype(self):
        if not self.low:self.node_type = 0
        elif not self.low.node is self.high.node and self.low.weight and self.high.weight: self.node_type = 1
        elif self.low.node is self.high.node and self.low.weight and self.high.weight: self.node_type = 2
        else: self.node_type = 3

    def extract_lim(self):
        if not self.low or (self.low.level == 0 and self.high.level == 0): return Lim(0, self.level)

        lim = None
        if not self.low.weight:
            lim = self.high.lim.copy()
        elif not self.high.weight:
            lim = self.low.lim.copy()
        else:
            lim = Lim.intersect(self.low.lim, self.high.lim)
        return lim

    def shift_lim(self, lim, signal=None):
        assert(not signal or isinstance(signal, LimBDDSignal))
        if not self.low or (self.low.level == 0 and self.high.level == 0): return

        if self.low.weight : self.low.lim  = Lim.sum(self.low.lim , lim)
        if self.high.weight: self.high.lim = Lim.sum(self.high.lim, lim)

        both_neg = self.low.neg and self.high.neg
        self.low.neg  ^= both_neg
        self.high.neg ^= both_neg

        for reference in self.references:
            reference.lim = Lim.sum(reference.lim, Lim.extend(lim))
            reference.neg ^= both_neg

        if signal:
            signal.lim = Lim.sum(signal.lim, Lim.extend(lim))
            signal.neg ^= both_neg

        # Reference updates above are parent successor edges: invalidate topo caches.
        self.invalidate_topo()
        for parent in self.parents: parent.invalidate_topo()
    
class LimBDDSignal:
    __slots__ = ('node', 'weight', 'neg', 'lim')

    def __init__(self, node, weight=1, neg=False, lim=None):
        assert(isinstance(node, LimBDDNode) and (weight == 0 or weight == 1) and (not lim or isinstance(lim, Lim)))
        self.node   = node
        self.weight = weight
        self.neg    = neg

        if lim: self.lim = lim.copy()
        else: self.lim = Lim(0, node.level)

    @property
    def level(self):
        return self.node.level

    @property
    def bit_count(self):
        return self.node.bit_count
    
    def update_parents(self):
        self.node.update_parents()

    def extract_lim(self):
        return self.node.extract_lim()
    
    def shift_lim(self, lim):
        assert(isinstance(lim, Lim))

        if len(self.node.parents): self.node.shift_lim(lim)
        else: self.node.shift_lim(lim, self)
    
    def edge_label(self):
        result = f'{self.weight},' + str(self.lim)
        return result
    
    def copy(self):
        result = LimBDDSignal(self.node, self.weight, self.neg)
        result.lim = self.lim.copy()
        return result