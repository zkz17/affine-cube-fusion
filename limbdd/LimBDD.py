from graphviz import Digraph
from sortedcontainers import SortedList

from utils.TruthTable import TruthTable
from utils.Cost import Cost
from limbdd.Node import LimBDDNode, LimBDDSignal
from limbdd.Lim import Lim

class SimpleBranch:
    __slots__ = ('bitstr', 'bitnum')

    # level : low 1 | 2 | ... | n high
    # bitstr:     1 | - | ... | 0 
    def __init__(self, bitstr = 0, bitnum = 0):
        self.bitstr = bitstr
        self.bitnum = bitnum

    @property
    def is_low_branch(self):
        return self.bitstr & 1 == 0

    @property
    def is_high_branch(self):
        return self.bitstr & 1 == 1

    def branch(self):
        return SimpleBranch(self.bitstr, self.bitnum + 1), SimpleBranch(self.bitstr + (1 << self.bitnum), self.bitnum + 1)
    
    def cut(self, pos):
        top_bitnum = pos
        bot_bitnum = self.bitnum - top_bitnum

        top_bitstr = self.bitstr & ((1 << top_bitnum) - 1)
        bot_bitstr = self.bitstr >> top_bitnum

        return SimpleBranch(top_bitstr, top_bitnum), SimpleBranch(bot_bitstr, bot_bitnum)
    
    def rshift(self, num=1):
        self.bitstr >>= num
        self.bitnum -= num
        return self
    
    def copy(self):
        return SimpleBranch(self.bitstr, self.bitnum)

    def __eq__(self, value):
        assert(isinstance(value, SimpleBranch))

        return value.bitnum == self.bitnum and value.bitstr == self.bitstr

    def __hash__(self):
        return hash((self.bitstr, self.bitnum))
    
    def __str__(self):
        return f'SimpleBranch({bin(self.bitstr)[2:].zfill(self.bitnum)})'


# Backward-compatibility alias (transitional: old notebooks etc. still import this as Branch; remove in a later phase)
Branch = SimpleBranch


counter = 0

class LimBDD:
    def __init__(self, tt=None, var_order=None, enable_neg=True):
        self.node_table = {}     # { LimBDDNode.topo : LimBDDNode }
        self.level2nodes = {}    # { LimBDDNode.level : { id(LimBDDNode) : LimBDDNode } }
        self.root = None         # LimBDDSignal
        self.terminator = None   # LimBDDNode
        # Note: when tt is None this list is held long-term as an instance attribute, so it must be created fresh each time
        # (the previous mutable default var_order=[] made all LimBDD instances share one list,
        #  and swap_neighbor_levels rewrites self.var_order **in place**).
        self.var_order = [] if var_order is None else var_order # reverse order
        self.split_order = []
        self.node_visited = 0
        self.node_idx_counter = 1
        self.reserved_node_index = []
        self.tt_bitstr = 0
        self.split_performed = False

        if tt: self.initialize(tt, var_order, enable_neg)

    def clear(self):
        self.node_table = {}
        self.level2nodes = {}
        self.root = None
        self.terminator = None
        self.var_order = []
        self.split_order = []
        self.node_visited = 0
        self.node_idx_counter = 1
        self.reserved_node_index = []
        self.tt_bitstr = 0
        self.split_performed = False
    
    def refresh(self):
        self.node_table.clear()
        for nodes in self.level2nodes.values():
            nodes.clear()

        node_list = [self.root.node]
        while len(node_list):
            node = node_list.pop()
            topo = node.topo
            if topo in self.node_table: continue

            self.node_table[topo] = node
            self.level2nodes[node.level][id(node)] = node
            node.parents.clear()

            if node.level:
                if node.low.weight:
                    node_list.append(node.low.node)
                if node.high.weight:
                    node_list.append(node.high.node)

        self.root.update_parents()

    def visit_node(self, node):
        assert(isinstance(node, LimBDDNode))

        self.node_visited |= (1 << node.idx)

    def visited(self, node):
        assert(isinstance(node, LimBDDNode))

        return ((self.node_visited >> node.idx) & 1) == 1
    
    def clear_visited(self):
        self.node_visited = 0

    def get_node_idx(self):
        if len(self.reserved_node_index): return self.reserved_node_index.pop()
        else:
            self.node_idx_counter += 1
            return self.node_idx_counter
        
    def pop_node_idx(self, idx):
        self.reserved_node_index.append(idx)

    def update_parents(self):
        for node in self.node_table.values():
            node.parents.clear()

        self.root.update_parents()

    def initialize(self, tt, var_order=None, enable_neg=True):
        assert(isinstance(tt, TruthTable))

        # Determine var order.
        if var_order:
            self.var_order = var_order.copy()
            
            self.split_order = []
            sorted_list = SortedList()
            for var_index in self.var_order:
                pos = sorted_list.bisect_left(var_index)
                self.split_order.append(var_index - pos)
                sorted_list.add(var_index)
            
            self.var_order.reverse()
            self.split_order.reverse()
        else: self.split_order = self.var_order = list(range(tt.num_var))

        # Initialize levels and terminator.
        for level in range(1, tt.num_var + 1):
            self.level2nodes[level] = {}

        self.terminator = LimBDDNode(1)
        self.node_table[self.terminator.topo] = self.terminator
        self.level2nodes[0] = {id(self.terminator): self.terminator}

        # Recursively construct LimBDD.
        self.root = self.construct(tt.copy(), enable_neg=enable_neg)
        self.update_parents()

        # Verify.
        #self.get_truth_table()
        #assert(self.tt_bitstr == tt.bitstr)
        #print(self.tt_bitstr)

    def construct(self, tt, enable_neg=True):
        assert(isinstance(tt, TruthTable))

        # Deal with True and False. 
        if tt.is_true():
            return self.insert_true_node_at_level(tt.num_var)
        if tt.is_false():
            return LimBDDSignal(self.terminator, weight=0)
        
        neg_flag = tt.try_neg() if enable_neg else False

        this_var_index = self.var_order[tt.num_var - 1]

        if tt.num_var == 1:
            # Construct new node. 
            low  = LimBDDSignal(self.terminator, tt.bitstr & 1)
            high = LimBDDSignal(self.terminator, (tt.bitstr & 2) >> 1)
            new_node = LimBDDNode(self.get_node_idx(), 1, low, high, var_index=this_var_index)

            # Insert to node table (if unique). 
            signal = self.insert_node_by_topo(new_node)
            signal.neg ^= neg_flag
            return signal
        elif tt.num_var < 1:
            raise Exception(f'Cannot construct LimBDD for {str(tt)}')
        else:
            # Split and construct new node. 
            tt0, tt1 = tt.split(self.split_order[tt.num_var - 1])
            if tt1.is_equal_to(tt0.bitstr):
                low  = self.construct(tt0, enable_neg)
                high = low.copy()
            else:
                low  = self.construct(tt0, enable_neg)
                high = self.construct(tt1, enable_neg)
            
            new_node = LimBDDNode(self.get_node_idx(), tt.num_var, low, high, var_index=this_var_index)

            # Insert to node table (if unique).
            signal = self.insert_node_by_topo(new_node)
            signal.neg ^= neg_flag
            return signal

    def insert_node_by_topo(self, node):
        assert(isinstance(node, LimBDDNode))

        if not node.low.weight and not node.high.weight:
            self.pop_node_idx(node.idx)
            return LimBDDSignal(self.terminator, weight=0)

        topo = node.topo
        if topo in self.node_table:
            topo_node = self.node_table[topo]
            signal = LimBDDSignal(topo_node)
            self.pop_node_idx(node.idx)

            
            swap = topo_node.low_key != node.low_key or (node.low.node is node.high.node \
                        and ((node.low.neg ^ topo_node.low.neg and node.low.neg ^ node.high.neg) \
                        or (topo_node.low.lim.bitstr ^ node.low.lim.bitstr).bit_count() > (topo_node.low.lim.bitstr ^ node.high.lim.bitstr).bit_count()))
            if swap: node.swap_children()

            if node.low.weight:
                signal.lim = Lim.sum(node.low.lim, topo_node.low.lim)
                signal.neg ^= (node.low.neg ^ topo_node.low.neg)
            else:
                signal.lim = Lim.sum(node.high.lim, topo_node.high.lim)
                signal.neg ^= (node.high.neg ^ topo_node.high.neg)
            while signal.lim.bitnum < topo_node.level - 1: signal.lim = Lim.extend(signal.lim)
            signal.lim = Lim.extend(signal.lim, swap)
            return signal
        else:
            signal = LimBDDSignal(node)
            signal.shift_lim(signal.extract_lim())
            self.node_table[topo] = node
            self.level2nodes[node.level][id(node)] = node
            return signal
    
    def insert_true_node_at_level(self, level):
        assert(level >= 0)

        if level == 0:
            return LimBDDSignal(self.terminator)
        else:
            low  = self.insert_true_node_at_level(level - 1)
            high = low.copy()
        new_node = LimBDDNode(self.get_node_idx(), level, low, high, var_index=self.var_order[level - 1])
        signal = self.insert_node_by_topo(new_node)

        low.node.add_parent(signal.node)
        high.node.add_parent(signal.node)
        return signal
    
    def remove_node(self, node):
        assert(isinstance(node, LimBDDNode))

        topo = node.topo
        if topo in self.node_table: self.node_table.pop(topo)

        try: del self.level2nodes[node.level][id(node)]
        except KeyError: pass

        for successor in node.successors:
            successor.node.remove_parent(node)
    
    def eliminate_redundant_lim(self):
        '''
        from config import config
        self.show(path=f'figure/debug/fig_{config['debug_figure_counter']}_before_eli')
        old_bitstr = self.get_truth_table()
        print(f'round {config['debug_figure_counter']}')
        '''
        if self.root.level < 2: return
        symmetry_map = {}
        for node in self.level2nodes[1].values():
            if node.is_fallthrough_node: symmetry_map[node.idx] = 1
            else: symmetry_map[node.idx] = 0

        for level in range(2, self.root.level + 1):
            update_list = {}
            for node in self.level2nodes[level].values():
                old_topo = None
                for i in range(2):
                    successor = node.successors[i]
                    if successor.weight:
                        update_bitstr = successor.lim.bitstr & symmetry_map[successor.node.idx]
                        if update_bitstr:
                            if not old_topo: old_topo = node.topo
                            successor.lim.bitstr ^= update_bitstr
                            node.invalidate_topo()
                if old_topo: update_list[node] = old_topo
            
            self.update_topo_list(update_list)
            for node in self.level2nodes[level].values():
                symmetry_bitstr = (1 << (level - 1)) - 1
                if node.low.weight : symmetry_bitstr &= symmetry_map[node.low.node.idx]
                if node.high.weight: symmetry_bitstr &= symmetry_map[node.high.node.idx]

                if node.is_symmetric: symmetry_bitstr += (1 << (level - 1))
                symmetry_map[node.idx] = symmetry_bitstr
        '''
        self.show(path=f'figure/debug/fig_{config['debug_figure_counter']}_after_eli')
        new_bitstr = self.get_truth_table()
        if new_bitstr != old_bitstr:
            print(f'Error in round {config['debug_figure_counter']}!')
            for index, bitstr in symmetry_map.items():
                print(f'index: {index}, bitstr: {bin(bitstr)[2:]}')
        config['debug_figure_counter'] += 1
        '''

    def insert_branch_from_another(self, other_pivot, branch, lim_accumulate):
        assert(isinstance(other_pivot, LimBDDNode) and isinstance(branch, SimpleBranch) and isinstance(lim_accumulate, Lim))

        swap_children = lim_accumulate.act_on_direct_child()
        if other_pivot.level == 1:
            low  = LimBDDSignal(self.terminator, weight=other_pivot.low.weight)
            high = LimBDDSignal(self.terminator, weight=other_pivot.high.weight)
            if swap_children: low, high = high, low
        else:
            if other_pivot.is_fallthrough_node:
                low  = self.insert_branch_from_another(other_pivot.low.node, branch, Lim.downgrade(lim_accumulate))
                high = low.copy()
                low.lim  = Lim.sum(low.lim , other_pivot.low.lim )
                high.lim = Lim.sum(high.lim, other_pivot.high.lim)
                if swap_children: low, high = high, low
            else:
                successor_index = branch.bitstr & 1
                new_successors = [LimBDDSignal(self.terminator, weight=0), LimBDDSignal(self.terminator, weight=0)]
                new_successors[successor_index ^ swap_children] = \
                    self.insert_branch_from_another(other_pivot.successors[successor_index].node, branch.copy().rshift(), Lim.downgrade(lim_accumulate))
                
                if new_successors[successor_index ^ swap_children].weight:
                    new_successors[successor_index ^ swap_children].lim = Lim.sum(new_successors[successor_index ^ swap_children].lim, other_pivot.successors[successor_index].lim)
                low, high = new_successors

        new_node = LimBDDNode(self.get_node_idx(), other_pivot.level, low, high, var_index=other_pivot.var_index)
        new_signal = self.insert_node_by_topo(new_node)
        return new_signal

    # Fuse self and another LimBDD (assumed to have same variable order) into one LimBDD.
    # Note: Does not support negation edges for now.
    def fuse(self, other):
        assert(isinstance(other, LimBDD) and self.var_order == other.var_order)

        target_limbdd = LimBDD()

        for level in range(1, self.root.level + 1):
            target_limbdd.level2nodes[level] = {}
        target_limbdd.var_order = self.var_order.copy()

        target_limbdd.terminator = LimBDDNode(1)
        target_limbdd.node_table[target_limbdd.terminator.topo] = target_limbdd.terminator
        target_limbdd.level2nodes[0] = {id(target_limbdd.terminator): target_limbdd.terminator}

        target_limbdd.root = target_limbdd.fuse_signal(self.root, other.root, Lim(0, self.root.level))

        target_limbdd.update_parents()
        # TODO: refresh?
        return target_limbdd

    def fuse_signal(self, signal, other_signal, lim_accumulate):
        assert(isinstance(signal, LimBDDSignal) and isinstance(other_signal, LimBDDSignal) and isinstance(lim_accumulate, Lim))

        if not other_signal.weight:
            return self.absorb_all(signal, Lim(0, signal.level))
        if not signal.weight:
            return self.absorb_all(other_signal, lim_accumulate)
        
        total_lim = Lim.sum(Lim.sum(signal.lim, other_signal.lim), lim_accumulate)
        next_lim_accumulate = Lim.downgrade(total_lim)

        swap_children = total_lim.act_on_direct_child()
        if signal.level == 1:
            if swap_children:
                low  = LimBDDSignal(self.terminator, weight=signal.node.low.weight ^ other_signal.node.high.weight)
                high = LimBDDSignal(self.terminator, weight=signal.node.high.weight ^ other_signal.node.low.weight)
            else:
                low  = LimBDDSignal(self.terminator, weight=signal.node.low.weight ^ other_signal.node.low.weight)
                high = LimBDDSignal(self.terminator, weight=signal.node.high.weight ^ other_signal.node.high.weight)
        else:
            if swap_children:
                low  = self.fuse_signal(signal.node.low, other_signal.node.high, next_lim_accumulate)
                high = self.fuse_signal(signal.node.high, other_signal.node.low, next_lim_accumulate)
            else:
                low  = self.fuse_signal(signal.node.low , other_signal.node.low , next_lim_accumulate)
                high = self.fuse_signal(signal.node.high, other_signal.node.high, next_lim_accumulate)
        new_node = LimBDDNode(self.get_node_idx(), signal.level, low, high, var_index=self.var_order[signal.level - 1])

        new_signal = self.insert_node_by_topo(new_node)
        if new_signal.weight: new_signal.lim = Lim.sum(new_signal.lim, signal.lim)
        return new_signal

    def absorb_all(self, signal, lim_accumulate):
        assert(isinstance(signal, LimBDDSignal) and isinstance(lim_accumulate, Lim))

        if not signal.weight: return LimBDDSignal(self.terminator, weight=0)

        next_lim_accumulate = Lim.downgrade(lim_accumulate)

        swap_children = lim_accumulate.act_on_direct_child()
        if signal.level == 1:
            low  = LimBDDSignal(self.terminator, weight=signal.node.low.weight)
            high = LimBDDSignal(self.terminator, weight=signal.node.high.weight)
        else:
            low  = self.absorb_all(signal.node.low , next_lim_accumulate)
            high = self.absorb_all(signal.node.high, next_lim_accumulate)
        if swap_children: low, high = high, low
        new_node = LimBDDNode(self.get_node_idx(), signal.level, low, high, var_index=self.var_order[signal.level - 1])

        new_signal = self.insert_node_by_topo(new_node)
        if new_signal.weight: new_signal.lim = Lim.sum(new_signal.lim, signal.lim)
        return new_signal

    # Try to update topo after replacing low and high branches.
    # Note: If old_topo == None, any change is assumed to be reflected in 'low' or 'high' and 'node' is assumed to be unchanged.
    def update_topo(self, node, low, high, old_topo=None):
        assert(isinstance(node, LimBDDNode) and isinstance(low, LimBDDSignal) and isinstance(high, LimBDDSignal))

        # Replace branches.
        if not old_topo: old_topo = node.topo
        node.successors = [low, high]
        node.invalidate_topo()
        node.recompute_bitcount()
        node.recompute_nodetype()

        # Remove outdated topo information.
        assert(old_topo in self.node_table)
        self.node_table.pop(old_topo)
        new_topo = node.topo
        if old_topo == new_topo: return

        # Update 'node_table'.
        if new_topo in self.node_table: # Fuse with an existing node.
            topo_node = self.node_table[new_topo]

            swap = topo_node.low_key != node.low_key or (node.low.node is node.high.node \
                        and ((node.low.neg ^ topo_node.low.neg and node.low.neg ^ node.high.neg) \
                        or (topo_node.low.lim.bitstr ^ node.low.lim.bitstr).bit_count() > (topo_node.low.lim.bitstr ^ node.high.lim.bitstr).bit_count()))
            if swap: node.swap_children()

            if node.low.weight:
                lim = Lim.sum(node.low.lim, topo_node.low.lim)
                neg = node.low.neg ^ topo_node.low.neg
            else:
                lim = Lim.sum(node.high.lim, topo_node.high.lim)
                neg = node.high.neg ^ topo_node.high.neg
            while lim.bitnum < topo_node.level - 1: lim = Lim.extend(lim)
            lim = Lim.extend(lim, swap)

            # Update parent topo.
            update_list = {}
            for parent in node.parents:
                update_list[parent] = parent.topo

                new_successors = []
                for parent_successor in parent.successors:
                    if parent_successor.node is node:
                        new_successor = LimBDDSignal(topo_node)
                        new_successor.lim = Lim.sum(parent_successor.lim, lim)
                        new_successor.neg = parent_successor.neg ^ neg
                        new_successors.append(new_successor)
                    else:
                        new_successors.append(parent_successor)
                
                topo_node.add_parent(parent)
                parent.successors = new_successors
                parent.invalidate_topo()

            self.update_topo_list(update_list)

            # Update successor topo.
            for successor in node.successors:
                successor.node.remove_parent(node)

            # Update 'level2nodes'.
            del self.level2nodes[node.level][id(node)]
        else: # Create a new node.
            update_list = {}
            for parent in node.parents:
                update_list[parent] = parent.topo

            node.shift_lim(node.extract_lim(), self.root if self.root.node is node else None)
            self.update_topo_list(update_list)

            self.node_table[new_topo] = node

    def update_topo_list(self, update_list):
        if not len(update_list): return

        next_update_list = {}
        for node, old_topo in update_list.copy().items():
            if node.topo == old_topo:
                update_list.pop(node)
                continue
            self.node_table.pop(old_topo)

            for parent in node.parents:
                if parent in next_update_list: continue
                next_update_list[parent] = parent.topo

        for node, old_topo in update_list.items():
            if not node.low.weight and not node.high.weight:
                # Update parent topo.
                for parent in node.parents:
                    new_successors = []
                    for parent_successor in parent.successors:
                        if parent_successor.node is node:
                            new_successors.append(LimBDDSignal(self.terminator, weight=0))
                        else:
                            new_successors.append(parent_successor)
                    parent.successors = new_successors
                    parent.invalidate_topo()

                # Update 'level2nodes'.
                del self.level2nodes[node.level][id(node)]
                continue

            new_topo = node.topo
            node.recompute_bitcount()
            node.recompute_nodetype()
            # Update 'node_table'.
            if new_topo in self.node_table: # Fuse with an existing node.
                topo_node = self.node_table[new_topo]

                swap = topo_node.low_key != node.low_key or (node.low.node is node.high.node \
                            and ((node.low.neg ^ topo_node.low.neg and node.low.neg ^ node.high.neg) \
                            or (topo_node.low.lim.bitstr ^ node.low.lim.bitstr).bit_count() > (topo_node.low.lim.bitstr ^ node.high.lim.bitstr).bit_count()))
                if swap: node.swap_children()

                if node.low.weight:
                    lim = Lim.sum(node.low.lim, topo_node.low.lim)
                    neg = node.low.neg ^ topo_node.low.neg
                else:
                    lim = Lim.sum(node.high.lim, topo_node.high.lim)
                    neg = node.high.neg ^ topo_node.high.neg
                while lim.bitnum < topo_node.level - 1: lim = Lim.extend(lim)
                lim = Lim.extend(lim, swap)

                # Update parent topo.
                for parent in node.parents:
                    new_successors = []
                    for parent_successor in parent.successors:
                        if parent_successor.node is node:
                            new_successor = LimBDDSignal(topo_node)
                            new_successor.lim = Lim.sum(parent_successor.lim, lim)
                            new_successor.neg = parent_successor.neg ^ neg
                            new_successors.append(new_successor)
                        else:
                            new_successors.append(parent_successor)
                    parent.successors = new_successors
                    parent.invalidate_topo()
                    
                    topo_node.add_parent(parent)

                # Update successor topo.
                for successor in node.successors:
                    successor.node.remove_parent(node)

                # Update 'level2nodes'.
                del self.level2nodes[node.level][id(node)]
            else: # Create a new node.
                node.shift_lim(node.extract_lim(), self.root if self.root.node is node else None)

                self.node_table[new_topo] = node

        self.update_topo_list(next_update_list)

    def branch_count(self, node, end_level=0, cache=None):
        assert(isinstance(node, LimBDDNode) and node.level >= end_level and end_level >= 0)

        if node.level == end_level: return 1

        key = (node.idx, end_level)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None: return cached

        count = 0
        if node.low.weight and node.high.weight:
            count += self.branch_count(node.low.node, end_level, cache)
            if not node.low.node is node.high.node: count += self.branch_count(node.high.node, end_level, cache)
        elif node.low.weight : count += self.branch_count(node.low.node , end_level, cache)
        elif node.high.weight: count += self.branch_count(node.high.node, end_level, cache)

        if cache is not None: cache[key] = count
        return count

    def swap_to_level(self, current_level, target_level):
        assert(current_level <= self.root.level and current_level > 0 and target_level <= self.root.level and target_level > 0)

        #print(f'before swap: current level = {current_level}, target_level = {target_level}')

        current_node_num, min_node_num, min_level = 0, 0, current_level
        min_branch_count = self.branch_count(self.root.node, cache={})

        def track(level):
            """Minimum node count (branch count as tie-breaker) criterion."""
            nonlocal min_node_num, min_level, min_branch_count
            new_branch_count = self.branch_count(self.root.node, cache={})
            if current_node_num < min_node_num:
                min_node_num = current_node_num
                min_level = level
                min_branch_count = new_branch_count
            elif current_node_num == min_node_num:
                if new_branch_count <= min_branch_count:
                    min_node_num = current_node_num
                    min_level = level
                    min_branch_count = new_branch_count

        while current_level < target_level:
            successful, delta_node_num = self.swap_up(current_level)
            if not successful: break

            current_level += 1
            current_node_num += delta_node_num
            track(current_level)

        while current_level > target_level:
            successful, delta_node_num = self.swap_down(current_level)
            if not successful: break

            current_level -= 1
            current_node_num += delta_node_num
            track(current_level)
        '''
        while current_level < target_level:
            successful, delta_node_num = self.swap_up(current_level)
            if not successful: break

            current_level += 1
            current_node_num += delta_node_num
            new_branch_count = self.branch_count(self.root.node)
            if new_branch_count < min_branch_count:
                min_node_num = current_node_num
                min_level = current_level
                min_branch_count = new_branch_count
            elif new_branch_count == min_branch_count:
                if current_node_num <= min_node_num:
                    min_node_num = current_node_num
                    min_level = current_level
                    min_branch_count = new_branch_count

        while current_level > target_level:
            successful, delta_node_num = self.swap_down(current_level)
            if not successful: break

            current_level -= 1
            current_node_num += delta_node_num
            new_branch_count = self.branch_count(self.root.node)
            if new_branch_count < min_branch_count:
                min_node_num = current_node_num
                min_level = current_level
                min_branch_count = new_branch_count
            elif new_branch_count == min_branch_count:
                if current_node_num <= min_node_num:
                    min_node_num = current_node_num
                    min_level = current_level
                    min_branch_count = new_branch_count
        '''
        #print(f'after swap: current level = {current_level}')

        return current_level, min_level

    def swap_down(self, level):
        return self.swap_neighbor_levels(level - 1, level)

    def swap_up(self, level):
        return self.swap_neighbor_levels(level, level + 1)

    def swap_neighbor_levels(self, low_level, high_level):
        assert(low_level + 1 == high_level and low_level > 0 and high_level <= self.root.level)
        '''
        old_bitstr = self.get_truth_table()
        '''
        # Record update data.
        new_low_level_nodes, new_high_level_nodes = [], []
        old_node_num = len(self.node_table)

        # Remove nodes from node_table.
        for level in range(low_level, high_level + 1):
            for node in self.level2nodes[level].values():
                self.node_table.pop(node.topo)
                self.pop_node_idx(node.idx)

        # Update upstream Lim.
        update_list = {}
        for level in range(high_level + 1, self.root.level + 1):
            for node in self.level2nodes[level].values():
                self.node_table.pop(node.topo)
                for successor in node.successors:
                    if successor.weight: successor.lim.swap_levels(low_level, high_level)
                node.invalidate_topo()

            for node in self.level2nodes[level].values():
                topo = node.topo
                self.node_table[topo] = node
                if level == high_level + 1: update_list[node] = topo
        self.root.lim.swap_levels(low_level, high_level)
        '''
        from config import config
        print(f'****** {config['debug_figure_counter'] - 1} ******')
        print(f'swapping {low_level} and {high_level}')
        print(f'    level2nodes before transform: {[(level, [node.idx for node in nodes]) for level, nodes in self.level2nodes.items()]}')
        '''
        # Update structure.
        #print(f'    high_level: {high_level}')
        #print(f'    high_nodes: {[node.idx for node in self.level2nodes[high_level]]}')
        for node in list(self.level2nodes[high_level].values()):

            # Compute corresponding Lim.
            successor_0_0 = node.successors[0].node.successors[0].copy() if node.successors[0].weight else LimBDDSignal(self.terminator, weight=0)
            successor_0_1 = node.successors[1].node.successors[0].copy() if node.successors[1].weight else LimBDDSignal(self.terminator, weight=0)
            successor_1_0 = node.successors[0].node.successors[1].copy() if node.successors[0].weight else LimBDDSignal(self.terminator, weight=0)
            successor_1_1 = node.successors[1].node.successors[1].copy() if node.successors[1].weight else LimBDDSignal(self.terminator, weight=0)

            # Swap children if needed.
            if node.successors[0].lim.act_on_direct_child(): successor_0_0, successor_1_0 = successor_1_0, successor_0_0
            if node.successors[1].lim.act_on_direct_child(): successor_0_1, successor_1_1 = successor_1_1, successor_0_1

            # Transfer negation.
            if node.successors[0].neg:
                if node.level == 2:
                    successor_0_0.weight ^= 1
                    successor_1_0.weight ^= 1
                else:
                    if successor_0_0.weight: successor_0_0.neg = not successor_0_0.neg
                    else: successor_0_0 = self.insert_true_node_at_level(node.level - 2)
                    if successor_1_0.weight: successor_1_0.neg = not successor_1_0.neg
                    else: successor_1_0 = self.insert_true_node_at_level(node.level - 2)

            if node.successors[1].neg:
                if node.level == 2:
                    successor_0_1.weight ^= 1
                    successor_1_1.weight ^= 1
                else:
                    if successor_0_1.weight: successor_0_1.neg = not successor_0_1.neg
                    else: successor_0_1 = self.insert_true_node_at_level(node.level - 2)
                    if successor_1_1.weight: successor_1_1.neg = not successor_1_1.neg
                    else: successor_1_1 = self.insert_true_node_at_level(node.level - 2)

            # Transfer Lim.
            successor_0_lim = Lim.downgrade(node.successors[0].lim) if node.successors[0].weight else None
            successor_1_lim = Lim.downgrade(node.successors[1].lim) if node.successors[1].weight else None

            if successor_0_0.weight and successor_0_lim: successor_0_0.lim = Lim.sum(successor_0_0.lim, successor_0_lim)
            if successor_0_1.weight and successor_1_lim: successor_0_1.lim = Lim.sum(successor_0_1.lim, successor_1_lim)
            if successor_1_0.weight and successor_0_lim: successor_1_0.lim = Lim.sum(successor_1_0.lim, successor_0_lim)
            if successor_1_1.weight and successor_1_lim: successor_1_1.lim = Lim.sum(successor_1_1.lim, successor_1_lim)

            # Insert newly generated nodes.
            new_low_node = LimBDDNode(self.get_node_idx(), low_level, successor_0_0, successor_0_1, var_index=node.var_index)
            new_low = self.insert_node_by_topo(new_low_node)
            successor_0_0.node.add_parent(new_low.node)
            successor_0_1.node.add_parent(new_low.node)

            new_high_node = LimBDDNode(self.get_node_idx(), low_level, successor_1_0, successor_1_1, var_index=node.var_index)
            new_high = self.insert_node_by_topo(new_high_node)
            successor_1_0.node.add_parent(new_high.node)
            successor_1_1.node.add_parent(new_high.node)

            if new_low.node  is new_low_node : new_low_level_nodes.append(new_low_node)
            if new_high.node is new_high_node: new_low_level_nodes.append(new_high_node)

            new_node = LimBDDNode(self.get_node_idx(), high_level, new_low, new_high, var_index=node.low.node.var_index if node.low.weight else node.high.node.var_index)
            new_signal = self.insert_node_by_topo(new_node)
            new_low.node.add_parent(new_signal.node)
            new_high.node.add_parent(new_signal.node)

            if new_signal.node is new_node: new_high_level_nodes.append(new_node)

            # Update reference signals.
            for reference in node.references:
                reference.node = new_signal.node
                if new_signal.weight:
                    reference.lim = Lim.sum(reference.lim, new_signal.lim)
                    reference.neg ^= new_signal.neg
                else:
                    reference.weight = 0
                    reference.lim = Lim(0, 0)
            for parent in node.parents: parent.invalidate_topo()
            if node.level == self.root.level:
                self.root.node = new_signal.node
                self.root.lim = Lim.sum(self.root.lim, new_signal.lim)
                self.root.neg ^= new_signal.neg
            '''
            print(f'  node    : {node.idx}, topo = {node.topo}')
            print(f'    parents   : {[parent.idx for parent in node.parents]}')
            print(f'  new_node: {new_node.idx}, topo = {new_node.topo}')
            print(f'    parents: {[parent.idx for parent in new_node.parents]}')
            print(str(new_high.lim), new_signal.node.idx, new_node.idx, new_signal.node.topo)
            '''
        # Update structure.
        self.update_parents()
        self.update_topo_list(update_list)
        self.level2nodes[low_level]  = {id(node): node for node in new_low_level_nodes}
        self.level2nodes[high_level] = {id(node): node for node in new_high_level_nodes}
        self.var_order[low_level - 1], self.var_order[high_level - 1] = self.var_order[high_level - 1], self.var_order[low_level - 1]
        
        for debug_level in range(1, self.root.level):
            for node in self.level2nodes[debug_level].values():
                assert(node.topo in self.node_table)
        '''
        self.show(path=f'figure/debug/fig_{config['debug_figure_counter'] - 1}_reorder')
        
        new_bitstr = self.get_truth_table()
        if old_bitstr != new_bitstr:
            raise Exception(f'Reordering gets wrong at {config['debug_figure_counter'] - 1} round!!! Expect: {bin(old_bitstr)[2:].zfill(1 << self.root.level)}, get {bin(new_bitstr)[2:].zfill(1 << self.root.level)}.')
        
        for nodes in self.level2nodes.values():
            for node in nodes:
                print(f'  node: {node.idx}, topo = {node.topo}')
                print(f'    parents: {[parent.idx for parent in node.parents]}')
                if node.idx == 1:
                    for parent in node.parents:
                        flag = True
                        for successor in parent.successors:
                            if successor.weight and successor.node is node: flag = False
                        if flag: raise Exception(f'Wrong parent {parent.idx} for terminator!!!')
                    
                    for parent in node.parents:
                        if not parent in self.level2nodes[parent.level]:
                            raise Exception(f'Unremoved parent {parent.idx}!!!')
        '''
        flag1 = self.find_and_try_negate(high_level)
        flag2 = self.find_and_try_negate(low_level)
        flag3 = self.find_and_try_negate(low_level - 1) if low_level >= 3 else False
        '''
        if flag1 or flag2 or flag3:
            new_bitstr = self.get_truth_table()
            self.show(path=f'figure/debug/fig_{config['debug_figure_counter'] - 1}_negate')
        
            if old_bitstr != new_bitstr:
                raise Exception(f'Negation gets wrong at {config['debug_figure_counter'] - 1} round!!! Expect: {bin(old_bitstr)[2:].zfill(1 << self.root.level)}, get {bin(new_bitstr)[2:].zfill(1 << self.root.level)}.')
        
            print(f'level2nodes after negation: {[(level, [node.idx for node in nodes]) for level, nodes in self.level2nodes.items()]}')
            for nodes in self.level2nodes.values():
                for node in nodes:
                    print(f'  node: {node.idx}, topo = {node.topo}')
                    print(f'    parents: {[parent.idx for parent in node.parents]}')
                    if node.idx == 1:
                        for parent in node.parents:
                            flag = True
                            for successor in parent.successors:
                                if successor.weight and successor.node is node: flag = False
                            if flag: raise Exception(f'Wrong parent {parent.idx} for terminator!!!')
                        
                        for parent in node.parents:
                            if not parent in self.level2nodes[parent.level]:
                                raise Exception(f'Unremoved parent {parent.idx}!!!')
        
        config['debug_figure_counter'] += 1
        '''
        return True, len(self.node_table) - old_node_num
    
    def find_and_try_negate(self, level):
        assert(level > 0 and level <= self.root.level)

        for debug_level in range(1, self.root.level):
            for node in self.level2nodes[debug_level].values():
                assert(node.topo in self.node_table)

        flag = False
        if level == 1: return flag
        nodes = list(self.level2nodes[level].values())
        node_index = 0
        while node_index < len(nodes):
            node = nodes[node_index]
            node_index += 1
            if node.bit_count > (1 << (level - 1)) and node.bit_count < (1 << level):
                flag = True
                '''
                global counter
                self.show(path=f'figure/debug/fig_{counter}')
                counter += 1
                '''
                update_list, references = {}, set()
                for parent in node.parents:
                    update_list[parent] = parent.topo
                    for successor in parent.successors:
                        if successor.node is node: references.add(successor)
                
                if node is self.root.node: references.add(self.root)

                #print(f'negate starting from node {node.idx}, topo = {node.topo}')
                #print(f'    update_list: {[(parent.idx, topo) for parent, topo in update_list.items()]}')
                neg_signal = self.negate(node, neg_map={})
                for reference in references:
                    reference.lim = Lim.sum(reference.lim, neg_signal.lim)
                    reference.neg ^= neg_signal.neg ^ True
                    reference.node = neg_signal.node

                for parent in update_list: parent.invalidate_topo()
                
                if not node is self.root.node:
                    for parent in update_list:
                        neg_signal.node.add_parent(parent)
                    self.update_topo_list(update_list)
                #print('node_table after negation: ', [(topo, node.idx) for topo, node in self.node_table.items()])

                self.refresh()
                nodes = list(self.level2nodes[level].values())

        nodes = list(self.level2nodes[level].values())
        node_index = 0
        while node_index < len(nodes):
            node = nodes[node_index]
            node_index += 1
            if node.bit_count != (1 << (level - 1)): continue
            big_gain, small_gain = 0, 0
            big_loss, small_loss = 0, 0

            for parent in node.parents:
                this, other = None, None
                if parent.low.node is node: this, other = parent.low, parent.high
                else: this, other = parent.high, parent.low

                assert(this.node is node)
                if other.weight:
                    if other.node is node:
                        if this.neg and other.neg: big_gain += 1
                        elif not this.neg and not other.neg: big_loss += 1
                    else:
                        if this.neg and other.neg: small_loss += 1
                        elif this.neg and not other.neg: big_gain += 1
                        elif not this.neg and other.neg: small_gain += 1
                        else: big_loss += 1
                else:
                    onechile_uncle = 0
                    for grand_parent in parent.parents:
                        for uncle in grand_parent.successors:
                            if uncle.node.is_onechild_node and uncle.node.low.neg ^ uncle.node.high.neg:
                                onechile_uncle += 1
                    
                    if onechile_uncle & 1: small_gain += 1
                    elif this.neg: big_gain += 1
                    else: big_loss += 1
            if big_gain < big_loss or (big_gain == big_loss and small_gain <= small_loss): continue

            flag = True
            update_list, references = {}, set()
            for parent in node.parents:
                update_list[parent] = parent.topo
                for successor in parent.successors:
                    if successor.node is node: references.add(successor)
                
            if node is self.root.node: references.add(self.root)

            #print(f'negate starting from node {node.idx}, topo = {node.topo}')
            #print(f'    update_list: {[(parent.idx, topo) for parent, topo in update_list.items()]}')
            neg_signal = self.negate(node, neg_map={})
            for reference in references:
                reference.lim = Lim.sum(reference.lim, neg_signal.lim)
                reference.neg ^= neg_signal.neg ^ True
                reference.node = neg_signal.node

            for parent in update_list: parent.invalidate_topo()
                
            if not node is self.root.node:
                for parent in update_list:
                    neg_signal.node.add_parent(parent)
                self.update_topo_list(update_list)
            #print('node_table after negation: ', [(topo, node.idx) for topo, node in self.node_table.items()])

            self.refresh()
            nodes = list(self.level2nodes[level].values())

        return flag
        
    def negate(self, start_node, neg_map=None):
        assert(isinstance(start_node, LimBDDNode))

        # neg_map is a recursive memo table (it gets written), so it must be created fresh each time: a shared default dict
        # would carry over the previous graph's LimBDDNode keys and signals into this call, returning a LimBDDSignal pointing into a destroyed subgraph.
        if neg_map is None: neg_map = {}

        if start_node in neg_map: return neg_map[start_node].copy()
        #print(f'negating node {start_node.idx}')

        if start_node.level == 1:
            low  = LimBDDSignal(self.terminator, weight=1 ^ start_node.low.weight)
            high = LimBDDSignal(self.terminator, weight=1 ^ start_node.high.weight)
            new_node = LimBDDNode(self.get_node_idx(), 1, low, high, var_index=start_node.var_index)
        else:
            no_discard_flag = start_node.low.node is start_node.high.node and (start_node.low.neg or start_node.high.neg)

            new_successors = []
            for successor in start_node.successors:
                if not successor.weight:
                    new_successors.append(self.insert_true_node_at_level(start_node.level - 1))
                elif successor.neg:
                    new_successor = successor.copy()
                    new_successor.neg = False
                    new_successors.append(new_successor)
                else:
                    new_successor = self.negate(successor.node, neg_map)
                    if new_successor.weight:
                        new_successor.lim = Lim.sum(new_successor.lim, successor.lim)
                    new_successors.append(new_successor)

            new_node = LimBDDNode(self.get_node_idx(), start_node.level, new_successors[0], new_successors[1], var_index=start_node.var_index)
            #print(f'mapping node {start_node.idx} to new_node {new_node.idx}')

        signal = self.insert_node_by_topo(new_node)
        neg_map[start_node] = signal.copy()
        #print(f'level2nodes : {[(level, [node.idx for node in nodes]) for level, nodes in self.level2nodes.items()]}')
        #print(f'neg_map     : {[(node.idx, signal.node.idx, signal.node.topo, str(signal.lim)) for node, signal in neg_map.items()]}')

        return signal
    
    def cost_estimate(self, top, bottom, cache=None):
        assert(isinstance(top, LimBDDNode) and isinstance(bottom, LimBDDNode) and top.level >= bottom.level)

        key = (top.idx, bottom.idx)
        if cache and key in cache: return cache[key].copy()

        if top.level == bottom.level:
            return Cost(0, 1) if top is bottom else None
        
        cost_low  = self.cost_estimate(top.low.node , bottom, cache) if top.low.weight  else None
        cost_high = self.cost_estimate(top.high.node, bottom, cache) if top.high.weight else None

        if top.low.node is top.high.node and cost_low and cost_high:
            cost = cost_low
        else:
            cost = Cost.meet(cost_low, cost_high)

        if cache: cache[key] = cost
        return cost
    
    def get_truth_table(self):
        self.tt_bitstr = 0
        if not self.root.weight: return 0

        self.get_tt_bits(self.root, Lim(0, self.root.level))
        return self.tt_bitstr

    def get_tt_bits(self, signal, lim, neg=False, pos=0):
        assert(isinstance(signal, LimBDDSignal) and isinstance(lim, Lim))

        if not signal.weight:
            if neg: self.get_all_one_bits(lim.bitnum, pos)
            return

        total_lim, total_neg = Lim.sum(signal.lim, lim), neg ^ signal.neg
        offset = 1 << signal.node.var_index
        if signal.level == 1:
            pivot0 = (1 << pos)
            pivot1 = (1 << (pos + offset))
            low_weight  = signal.node.low.weight  ^ int(total_neg)
            high_weight = signal.node.high.weight ^ int(total_neg)
            assert((self.tt_bitstr & pivot0 == 0) and (self.tt_bitstr & pivot1 == 0))

            if total_lim.act_on_direct_child():
                self.tt_bitstr += pivot1 * low_weight + pivot0 * high_weight
            else:
                self.tt_bitstr += pivot0 * low_weight + pivot1 * high_weight
        else:
            if total_lim.act_on_direct_child():
                self.get_tt_bits(signal.node.high, Lim.downgrade(total_lim), total_neg, pos)
                self.get_tt_bits(signal.node.low , Lim.downgrade(total_lim), total_neg, pos + offset)
            else:
                self.get_tt_bits(signal.node.low , Lim.downgrade(total_lim), total_neg, pos)
                self.get_tt_bits(signal.node.high, Lim.downgrade(total_lim), total_neg, pos + offset)

    def get_all_one_bits(self, level, pos):
        offset = 1 << next(iter(self.level2nodes[level].values())).var_index
        if level == 1:
            pivot0 = (1 << pos)
            pivot1 = (1 << (pos + offset))
            assert((self.tt_bitstr & pivot0 == 0) and (self.tt_bitstr & pivot1 == 0))

            self.tt_bitstr += pivot0 + pivot1
        else:
            self.get_all_one_bits(level - 1, pos)
            self.get_all_one_bits(level - 1, pos + offset)

    def show(self, path="figure/debug/fig", skip_zero=True, title='LimBDD'):
        dot = Digraph(name="LimBDD")

        # Insert invisible root node.
        src_name = "0"
        dot.attr(rankdir='TB')
        dot.node(src_name, '', style='invisible')

        # Recursively draw all signals.
        self.clear_visited()
        self.draw_signal(dot, src_name, self.root, skip_zero=skip_zero)
        self.clear_visited()

        # Line up nodes in the same level.
        for level, node_list in self.level2nodes.items():
            with dot.subgraph() as sub:
                sub.attr(rank='same')
                for node in node_list.values():
                    sub.node(str(node.idx))

        # Set image title.
        dot.attr(label=title, fontsize='20', labelloc='t', labeljust='c')

        # Save as PNG format.
        return dot.render(path, format='png')

    def draw_signal(self, dot, src_name, signal, dashed=False, skip_zero=True):
        assert(isinstance(dot, Digraph) and isinstance(src_name, str) and isinstance(signal, LimBDDSignal))

        # Skip 0 signal.
        if skip_zero and signal.weight == 0: return

        # Draw node.
        tgt_name = str(signal.node.idx)
        tgt_label = f'x{signal.node.var_index}' if signal.node.var_index >= 0 else '1'
        dot.node(tgt_name, tgt_label, shape='circle')

        # Draw edge.
        edge_label = signal.edge_label()
        arrowhead = 'odot' if signal.neg else 'normal'
        if skip_zero: edge_label = edge_label[2:]
        if dashed: dot.edge(src_name, tgt_name, edge_label, style='dashed', arrowhead=arrowhead)
        else: dot.edge(src_name, tgt_name, edge_label, arrowhead=arrowhead)

        # Draw children (if haven't).
        if signal.node.low and not self.visited(signal.node):
            self.draw_signal(dot, tgt_name, signal.node.low, dashed=True, skip_zero=skip_zero)
            self.draw_signal(dot, tgt_name, signal.node.high, skip_zero=skip_zero)
            
        # Mark as visited.
        self.visit_node(signal.node)