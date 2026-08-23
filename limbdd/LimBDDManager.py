
from config import config
from limbdd.Lim import Lim
from limbdd.Node import LimBDDNode, LimBDDSignal
from limbdd.LimBDD import LimBDD, SimpleBranch

# 向后兼容别名（过渡期：旧 notebook 以 Branch 从本模块导入；后续阶段移除）
Branch = SimpleBranch
from limbdd.patterns.SplitPattern import SplitPattern

class LimBDDManager:
    def __init__(self):
        self.limbdd = None
        self.limbdd_list = {}  # { max_branch_length : [ LimBDD ] }
        self.var_order = []
        self.temp_list = []
        self.pattern_pool = []
        self.init_patterns()

    def init_patterns(self):
        from limbdd.patterns.SplitInterceptorPattern import SplitInterceptorPattern
        pattern = SplitInterceptorPattern()
        self.pattern_pool.append(pattern)

        from limbdd.patterns.SplitLv2Pattern import SplitLv2Pattern
        pattern = SplitLv2Pattern()
        self.pattern_pool.append(pattern)

    def initialize(self, tt, var_order=None):
        self.limbdd = LimBDD(tt, var_order)

        self.limbdd_list.clear()
        for height in range(self.limbdd.root.level + 1):
            self.limbdd_list[height] = []

    def append_temp_to_list(self):
        if len(self.temp_list):
            self.limbdd_list[self.temp_list[0].root.level].extend(self.temp_list)
            self.temp_list = []

    def fuse_level(self, level):
        assert(level in self.limbdd_list)

        while len(self.limbdd_list[level]) and not self.limbdd_list[level][-1].root.weight: self.limbdd_list[level].pop()
        if not len(self.limbdd_list[level]): return None
        else:
            limbdd = self.limbdd_list[level].pop()
            while len(self.limbdd_list[level]):
                next_limbdd = self.limbdd_list[level].pop()
                if not next_limbdd.root.weight: continue
                limbdd = limbdd.fuse(next_limbdd)

            return limbdd
        
    def collect_branches(self, limbdd, level):
        assert(isinstance(limbdd, LimBDD))

        self.perform_split(limbdd)
        for other_limbdd in self.limbdd_list[level]:
            if not other_limbdd.root.weight: continue
            self.perform_split(other_limbdd)
            # 已定结论(2025-08-22,bdd_path_conclusions.md (1)):try_absorb_branch 已移除——
            # 它只做 C1 抵消,而全局融合引擎 fuse_pool_global 无条件覆盖 C1;
            # P4 实测其单独效应严格为零(±8 噪声级)。函数本体保留于本模块供研究。
            if not limbdd.root.weight: break
    
    def try_neg(self, limbdd):
        assert(isinstance(limbdd, LimBDD))

        flag = False
        for level in range(1, limbdd.root.level):
            flag = limbdd.find_and_try_negate(level) or flag
        return flag
    
    def split_neg(self, limbdd, stop=1):
        assert(isinstance(limbdd, LimBDD))

        for level in range(limbdd.root.level + 1, stop, -1):
            split_limbdd = self.split_neg_at_level(limbdd, level)
            if not split_limbdd: continue
            #split_limbdd.show(path=f'figure/debug/fig_{config['debug_figure_counter']}_split')
            #limbdd.show(path=f'figure/debug/fig_{config['debug_figure_counter']}')

            self.limbdd_list[split_limbdd.root.level].append(split_limbdd)

    def split_neg_at_level(self, limbdd, level):
        assert(isinstance(limbdd, LimBDD) and level <= limbdd.root.level + 1) # Negation edges point from level to level - 1.

        if level < 2: return None

        target_nodes = []
        if level == limbdd.root.level + 1:
            if not limbdd.root.neg: return None
        else:
            for node in limbdd.level2nodes[level].values():
                if node.low.neg or node.high.neg: target_nodes.append(node)
            if not len(target_nodes): return None
        
        split_limbdd = LimBDDManager.prepare_empty_limbdd(limbdd.root.level + 2 - level)
        termination_constructor = lambda _, successor: LimBDDSignal(split_limbdd.terminator, int(successor.neg))

        if level == limbdd.root.level + 1:
            split_limbdd.root = LimBDDSignal(split_limbdd.terminator)
            limbdd.root.neg = False
        elif level == 2:
            update_list = {}
            for node in target_nodes:
                update_list[node] = node.topo
                
                new_successors = []
                for successor in node.successors:
                    if successor.weight and successor.neg:
                        if successor.node.low.weight and successor.node.high.weight: new_successors.append(LimBDDSignal(limbdd.terminator, weight=0))
                        else:
                            low  = LimBDDSignal(limbdd.terminator, weight=successor.node.low.weight  ^ 1)
                            high = LimBDDSignal(limbdd.terminator, weight=successor.node.high.weight ^ 1)
                            new_node = LimBDDNode(limbdd.get_node_idx(), 1, low, high, var_index=successor.node.var_index)

                            new_successor = limbdd.insert_node_by_topo(new_node)
                            new_successor.lim = Lim.sum(new_successor.lim, successor.lim)
                            new_successors.append(new_successor)
                    else:
                        new_successors.append(successor)
                node.successors = new_successors
                node.invalidate_topo()
            
            limbdd.update_topo_list(update_list)
            limbdd.refresh()
            return None
        else:
            limbdd.clear_visited()
            split_limbdd.root = SplitPattern.copy_nodes_between(limbdd.root, target_nodes, split_limbdd, termination_constructor, level_drop=target_nodes[0].level - 1, copy_map={})
            split_limbdd.update_parents()
            
            update_list = {}
            for node in target_nodes:
                update_list[node] = node.topo

                new_successors = []
                for successor in node.successors:
                    if successor.weight and successor.neg:
                        new_successor = successor.copy()
                        new_successor.neg = False
                        new_successors.append(new_successor)
                    else:
                        new_successors.append(successor)
                node.successors = new_successors
                node.invalidate_topo()
            limbdd.update_topo_list(update_list)
        split_limbdd.var_order = limbdd.var_order[(level - 1):]
        return split_limbdd
    
    def perform_split(self, limbdd):
        assert(isinstance(limbdd, LimBDD))

        if limbdd.split_performed: return

        while True:
            limbdd.eliminate_redundant_lim()
            changed = False
            for pattern in self.pattern_pool:
                changed = self.split_with_pattern(limbdd, pattern) or changed
            if not changed: break

        limbdd.split_performed = True

    def split_with_pattern(self, limbdd, pattern):
        assert(isinstance(limbdd, LimBDD) and isinstance(pattern, SplitPattern))

        changed = False
        for level in range(2, limbdd.root.level + 1):
            #bitstr_old = limbdd.get_truth_table()
            split_limbdd = self.split_at_level_with_pattern(limbdd, level, pattern)
            if not split_limbdd: continue
            '''
            split_limbdd.show(path=f'figure/debug/fig_{config['debug_figure_counter']}_split')
            limbdd.show(path=f'figure/debug/fig_{config['debug_figure_counter']}')
            bitstr_new = limbdd.get_truth_table() ^ split_limbdd.get_truth_table()
            if bitstr_new != bitstr_old: print(f'Error occurred in round {config['debug_figure_counter']} with pattern {type(pattern)}! Expect {bin(bitstr_old)[2:].zfill(1 << limbdd.root.level)}, got {bin(bitstr_new)[2:].zfill(1 << limbdd.root.level)}.')
            config['debug_figure_counter'] += 1
            '''
            if split_limbdd.root.weight:
                self.temp_list.append(split_limbdd)
                changed = True

        return changed

    def split_at_level_with_pattern(self, limbdd, level, pattern):
        assert(isinstance(limbdd, LimBDD) and isinstance(pattern, SplitPattern) and level <= limbdd.root.level)

        candidates = pattern.identify_candidates(limbdd, level)
        if not len(candidates): return None

        split_limbdd = LimBDDManager.prepare_empty_limbdd(limbdd.root.level + 1)
        split_limbdd.var_order = limbdd.var_order.copy()
        split_limbdd, split_targets, split_info = pattern.process_split(limbdd, split_limbdd, candidates)

        # Update original limbdd.
        update_list = {}
        for i in range(len(split_targets)):
            node = split_targets[i]
            update_list[node] = node.topo
            new_successors = split_info[i][0]
            node.successors = new_successors
            node.invalidate_topo()
        limbdd.update_topo_list(update_list)
        limbdd.refresh()
        
        if not split_limbdd.root: return None
        split_limbdd.refresh()
        split_limbdd.var_order = limbdd.var_order.copy()
        return split_limbdd

    def sifting(self, limbdd, low_level, high_level):
        assert(isinstance(limbdd, LimBDD))
        if not low_level < high_level: return
        if low_level + 1 == high_level: return

        flag = True
        for level in range(low_level + 1, high_level + 1):
            if len(limbdd.level2nodes[level]) == 1: continue
            flag = False
            break
        if flag: return

        mid_level = (low_level + high_level) >> 1

        candidate_levels = sorted(range(low_level + 1, high_level + 1), key=lambda level: len(limbdd.level2nodes[level]))
        while len(candidate_levels):
            start_level = candidate_levels.pop()

            final_level = 0
            if start_level <= mid_level:
                current_level_0, min_level_0 = limbdd.swap_to_level(start_level, low_level + 1)
                current_level_1, min_level_1 = limbdd.swap_to_level(current_level_0, high_level)
                final_level, _ = limbdd.swap_to_level(current_level_1, min_level_1)
            else:
                current_level_0, min_level_0 = limbdd.swap_to_level(start_level, high_level)
                current_level_1, min_level_1 = limbdd.swap_to_level(current_level_0, low_level + 1)
                final_level, _ = limbdd.swap_to_level(current_level_1, min_level_1)

            if final_level > start_level:
                for i in range(len(candidate_levels)):
                    if candidate_levels[i] <= final_level and candidate_levels[i] > start_level: candidate_levels[i] -= 1
            elif final_level < start_level:
                for i in range(len(candidate_levels)):
                    if candidate_levels[i] >= final_level and candidate_levels[i] < start_level: candidate_levels[i] += 1

            #self.split_neg(limbdd, limbdd.root.level - 1)
            self.split_neg(limbdd)

    @staticmethod
    def prepare_empty_limbdd(height):
        assert(height > 0)

        empty_limbdd = LimBDD()
        for level in range(1, height):
            empty_limbdd.level2nodes[level] = {}

        empty_limbdd.terminator = LimBDDNode(1)
        empty_limbdd.node_table[empty_limbdd.terminator.topo] = empty_limbdd.terminator
        empty_limbdd.level2nodes[0] = {id(empty_limbdd.terminator): empty_limbdd.terminator}

        return empty_limbdd

    @staticmethod
    def branch_process(start, targets, prefix=None):
        if prefix is None: prefix = SimpleBranch()
        assert(isinstance(start, LimBDDNode) and isinstance(prefix, SimpleBranch) and len(targets))

        if start in targets:
            return {start: [prefix]} if prefix.bitnum else {}
        elif start.level == 0: return {}
        else:
            if start.low.node is start.high.node and start.low.weight and start.high.weight:
                return LimBDDManager.branch_process(start.low.node, targets, prefix)
            else:
                branch_0, branch_1 = prefix.branch()
                branch_0_map = LimBDDManager.branch_process(start.low.node , targets, branch_0) if start.low.weight  else {}
                branch_1_map = LimBDDManager.branch_process(start.high.node, targets, branch_1) if start.high.weight else {}
                for end, branch_list in branch_1_map.items():
                    if end in branch_0_map: branch_0_map[end] += branch_list
                    else: branch_0_map[end] = branch_list
                return branch_0_map

    @staticmethod  
    def locate_branch_end(start, branch, length=0):
        assert(isinstance(start, LimBDDNode) and isinstance(branch, SimpleBranch))
               
        end = start
        bitstr = branch.bitstr
        bitnum = length if length else branch.bitnum
        while bitnum:
            while end.is_fallthrough_node: end = end.low.node
            end = end.successors[bitstr & 1].node
            bitnum -= 1
            bitstr >>= 1
        return end

    @staticmethod
    def pivot_to(start, branch0, branch1, limbdd, target_limbdd=None):
        assert(isinstance(start, LimBDDNode) and isinstance(branch0, SimpleBranch) and isinstance(branch1, SimpleBranch) and isinstance(limbdd, LimBDD))

        end0, end1 = LimBDDManager.locate_branch_end(start, branch0), LimBDDManager.locate_branch_end(start, branch1)
        bitstr0, bitstr1 = branch0.bitstr, branch1.bitstr
        bitnum0, bitnum1 = branch0.bitnum, branch1.bitnum
        if end0.level < end1.level:
            end0, end1 = end1, end0
            bitstr0, bitstr1 = bitstr1, bitstr0
            bitnum0, bitnum1 = bitnum1, bitnum0

        if not target_limbdd:
            target_limbdd = LimBDDManager.prepare_empty_limbdd(limbdd.root.level + 1)
            target_limbdd.var_order = limbdd.var_order[end1.level:]

        bitstr_xor = bitstr0 ^ bitstr1
        branch_node = start
        pivot, pos = 1, 0
        while (bitstr_xor & pivot) == 0 and not branch_node is end0:
            while branch_node.is_fallthrough_node:
                branch_node = branch_node.low.node
            branch_node = branch_node.successors[bitstr0 & 1].node
            bitstr0 >>= 1
            bitstr1 >>= 1
            pivot <<= 1
            pos += 1
        while branch_node.is_fallthrough_node and not branch_node is end0:
            branch_node = branch_node.low.node

        bitstr_common = branch0.bitstr & (pivot - 1)
        level_drop, margin = end1.level, end0.level - end1.level
        #print(f'branch_node: {branch_node.idx}, end0: {end0.idx}, end1: {end1.idx}')
        #print(f'bitstr_common: {bin(bitstr_common)[2:]}, bitstr0: {bin(bitstr0)[2:]}, bitstr1: {bin(bitstr1)[2:]}')

        if branch_node is end0:
            if branch_node.is_fallthrough_node:
                low  = LimBDDManager.copy_branch(branch_node.low , bitstr1, end1, LimBDDSignal(target_limbdd.terminator), limbdd, target_limbdd, level_drop)
                high = LimBDDManager.copy_branch(branch_node.high, bitstr1, end1, LimBDDSignal(target_limbdd.terminator), limbdd, target_limbdd, level_drop)
            else:
                if bitstr1 & 1:
                    low  = LimBDDSignal(target_limbdd.terminator, weight=0)
                    high = LimBDDManager.copy_branch(branch_node.high, bitstr1 >> 1, end1, LimBDDSignal(target_limbdd.terminator), limbdd, target_limbdd, level_drop)
                else:
                    low  = LimBDDManager.copy_branch(branch_node.low , bitstr1 >> 1, end1, LimBDDSignal(target_limbdd.terminator), limbdd, target_limbdd, level_drop)
                    high = LimBDDSignal(target_limbdd.terminator, weight=0)

            new_node = LimBDDNode(target_limbdd.get_node_idx(), branch_node.level - level_drop, low, high, var_index=branch_node.var_index)
            signal = target_limbdd.insert_node_by_topo(new_node)
            end_signal = target_limbdd.negate(signal.node, {})
            end_signal.lim = Lim.sum(signal.lim, end_signal.lim)
        else:
            end_signal_0 = target_limbdd.insert_true_node_at_level(end0.level - end1.level)
            end_signal_1 = target_limbdd.insert_true_node_at_level(0)

            if bitstr0 & 1:
                low  = LimBDDManager.copy_branch(branch_node.low , bitstr1 >> 1, end1, end_signal_1, limbdd, target_limbdd, level_drop)
                high = LimBDDManager.copy_branch(branch_node.high, bitstr0 >> 1, end0, end_signal_0, limbdd, target_limbdd, level_drop, margin)
            else:
                low  = LimBDDManager.copy_branch(branch_node.low , bitstr0 >> 1, end0, end_signal_0, limbdd, target_limbdd, level_drop, margin)
                high = LimBDDManager.copy_branch(branch_node.high, bitstr1 >> 1, end1, end_signal_1, limbdd, target_limbdd, level_drop)

            new_node = LimBDDNode(target_limbdd.get_node_idx(), branch_node.level - level_drop, low, high, var_index=branch_node.var_index)
            end_signal = target_limbdd.insert_node_by_topo(new_node)
            
        target_limbdd.root = LimBDDManager.copy_branch(limbdd.root, bitstr_common, branch_node, end_signal, limbdd, target_limbdd, level_drop)
        target_limbdd.refresh()

        def update_x_cx_map_on_signal(signal, x_map, cx_map, level_cap, ctrl=-1, polar=0):
            for level_minus_1 in range(level_cap):
                pivot = 1 << level_minus_1
                if pivot & signal.lim.bitstr:
                    target_var_index = limbdd.var_order[level_minus_1]
                    if ctrl >= 0:
                        if target_var_index in cx_map[ctrl]:
                            if cx_map[ctrl][target_var_index] != polar:
                                x_map[target_var_index] ^= 1
                            cx_map[ctrl].pop(target_var_index)
                        else:
                            cx_map[ctrl][target_var_index] = polar
                    else:
                        x_map[target_var_index] ^= 1

        def update_x_cx_map(start, end, bitstr, x_map, cx_map, level_cap):
            pivot_node = start
            while not pivot_node is end:
                if pivot_node.is_fallthrough_node:
                    update_x_cx_map_on_signal(pivot_node.low , x_map, cx_map, level_cap, ctrl=pivot_node.var_index, polar=0)
                    update_x_cx_map_on_signal(pivot_node.high, x_map, cx_map, level_cap, ctrl=pivot_node.var_index, polar=1)
                    pivot_node = pivot_node.low.node
                else:
                    if bitstr & 1:
                        update_x_cx_map_on_signal(pivot_node.high, x_map, cx_map, level_cap)
                        pivot_node = pivot_node.high.node
                    else:
                        update_x_cx_map_on_signal(pivot_node.low , x_map, cx_map, level_cap)
                        pivot_node = pivot_node.low.node
                    bitstr >>= 1

        x_map, cx_map = {}, {}
        for var_index in limbdd.var_order[:end0.level]:
            x_map[var_index] = 0
        for var_index in limbdd.var_order:
            cx_map[var_index] = {}

        update_x_cx_map(branch_node, end0, bitstr0, x_map, cx_map, end0.level)
        update_x_cx_map(branch_node, end1, bitstr1, x_map, cx_map, end1.level)

        return target_limbdd, x_map, cx_map
    
    @staticmethod
    def copy_branch(signal, bitstr, end, new_end_signal, limbdd, target_limbdd, level_drop, margin=0):
        assert(isinstance(signal, LimBDDSignal) and isinstance(end, LimBDDNode) and isinstance(new_end_signal, LimBDDSignal) \
               and isinstance(limbdd, LimBDD) and isinstance(target_limbdd, LimBDD))

        #print(f'copy start: {signal.node.idx}, end: {end.idx}, bitstr: {bitstr}')
        if signal.node is end:
            new_signal = new_end_signal.copy()
        elif signal.node.is_fallthrough_node:
            low = LimBDDManager.copy_branch(signal.node.low, bitstr, end, new_end_signal, limbdd, target_limbdd, level_drop)
            high = low.copy()
            high.lim = Lim.sum(high.lim, Lim.truncate(Lim.sum(signal.node.low.lim, signal.node.high.lim), signal.level - 1 - level_drop))
            for pos in range(margin):
                pivot = 1 << pos
                if high.lim.bitstr & pivot: high.lim.bitstr -= pivot

            new_node = LimBDDNode(target_limbdd.get_node_idx(), signal.level - level_drop, low, high, signal.node.var_index)
            new_signal = target_limbdd.insert_node_by_topo(new_node)
        else:
            low, high = LimBDDSignal(target_limbdd.terminator, weight=0), LimBDDSignal(target_limbdd.terminator, weight=0)
            if bitstr & 1: high = LimBDDManager.copy_branch(signal.node.high, bitstr >> 1, end, new_end_signal, limbdd, target_limbdd, level_drop)
            else: low = LimBDDManager.copy_branch(signal.node.low, bitstr >> 1, end, new_end_signal, limbdd, target_limbdd, level_drop)

            new_node = LimBDDNode(target_limbdd.get_node_idx(), signal.level - level_drop, low, high, signal.node.var_index)
            new_signal = target_limbdd.insert_node_by_topo(new_node)

        new_signal.lim = Lim.sum(new_signal.lim, Lim.truncate(signal.lim, signal.level - level_drop))
        for pos in range(margin):
            pivot = 1 << pos
            if new_signal.lim.bitstr & pivot: new_signal.lim.bitstr -= pivot
        return new_signal

    @staticmethod
    def move_branches(signal, branch_list, limbdd, target_limbdd=None, prefix=None):
        if prefix is None: prefix = SimpleBranch()
        assert(isinstance(signal, LimBDDSignal) and isinstance(limbdd, LimBDD) and (not target_limbdd or isinstance(target_limbdd, LimBDD)))

        new_signal, downstream_diverge_node, update_map = None, False, {}
        if signal.level == 0:
            if target_limbdd: new_signal = LimBDDSignal(target_limbdd.terminator, weight=signal.weight)
        elif signal.node.is_branch_node:
            low_branch_list, high_branch_list = [], []
            for branch in branch_list:
                assert(isinstance(branch, SimpleBranch))

                remaining_branch = branch.copy().rshift()
                if branch.is_low_branch: low_branch_list.append(remaining_branch)
                else: high_branch_list.append(remaining_branch)

            prefix0, prefix1 = prefix.branch()
            low , lowstream_diverge_node , low_update_map  = \
                LimBDDManager.move_branches(signal.node.low , low_branch_list , limbdd, target_limbdd, prefix0) if len(low_branch_list)  else (None, True, {})
            high, highstream_diverge_node, high_update_map = \
                LimBDDManager.move_branches(signal.node.high, high_branch_list, limbdd, target_limbdd, prefix1) if len(high_branch_list) else (None, True, {})
            
            downstream_diverge_node = lowstream_diverge_node or highstream_diverge_node
            update_map.update(low_update_map)
            update_map.update(high_update_map)

            if not lowstream_diverge_node or not highstream_diverge_node:
                key = (id(signal), prefix.bitstr, prefix.bitnum)
                val = 0
                if not lowstream_diverge_node : val ^= 1
                if not highstream_diverge_node: val ^= 2
                update_map[key] = val
        elif signal.node.is_fallthrough_node:
            low, high = None, None
            low, downstream_diverge_node, update_map = LimBDDManager.move_branches(signal.node.low, branch_list, limbdd, target_limbdd, prefix)
            if target_limbdd:
                high = low.copy()
                if high.weight: high.lim = Lim.sum(high.lim, Lim.sum(signal.node.low.lim, signal.node.high.lim))
        else: # is_onechild_node
            for branch in branch_list: branch.rshift()

            prefix0, prefix1 = prefix.branch()
            low, high = None, None
            if signal.node.low.weight:
                low , downstream_diverge_node, update_map = LimBDDManager.move_branches(signal.node.low , branch_list, limbdd, target_limbdd, prefix0)
            else:
                high, downstream_diverge_node, update_map = LimBDDManager.move_branches(signal.node.high, branch_list, limbdd, target_limbdd, prefix1)

        if target_limbdd and signal.level:
            if not low : low  = LimBDDSignal(target_limbdd.terminator, weight=0)
            if not high: high = LimBDDSignal(target_limbdd.terminator, weight=0)

            new_node = LimBDDNode(target_limbdd.get_node_idx(), signal.level, low, high, var_index=signal.node.var_index)
            new_signal = target_limbdd.insert_node_by_topo(new_node)
            if new_signal.weight: new_signal.lim = Lim.sum(new_signal.lim, signal.lim)

        return new_signal, downstream_diverge_node, update_map

    @staticmethod
    def remove_branches_with_signal_map(limbdd, signal, signal_map, prefix=None):
        if prefix is None: prefix = SimpleBranch()
        assert(isinstance(limbdd, LimBDD) and isinstance(signal, LimBDDSignal))

        if signal.level == 0: return signal

        if signal.node.low.node is signal.node.high.node and signal.level > 1:
            low = LimBDDManager.remove_branches_with_signal_map(limbdd, signal.node.low, signal_map, prefix)
            high = low.copy()
            high.lim = Lim.sum(high.lim, Lim.sum(signal.node.low.lim, signal.node.high.lim))
        else:
            remove_low, remove_high = False, False
            key = (id(signal), prefix.bitstr, prefix.bitnum)
            if key in signal_map:
                val = signal_map[key]
                remove_low, remove_high = (val & 1 != 0), (val & 2 != 0)

            prefix0, prefix1 = prefix.branch()
            low  = LimBDDSignal(limbdd.terminator, weight=0) if remove_low  \
                else LimBDDManager.remove_branches_with_signal_map(limbdd, signal.node.low , signal_map, prefix0)
            high = LimBDDSignal(limbdd.terminator, weight=0) if remove_high \
                else LimBDDManager.remove_branches_with_signal_map(limbdd, signal.node.high, signal_map, prefix1)

        new_node = LimBDDNode(limbdd.get_node_idx(), signal.level, low, high, var_index=signal.node.var_index)
        new_signal = limbdd.insert_node_by_topo(new_node)
        if new_signal.weight:
            new_signal.lim = Lim.sum(new_signal.lim, signal.lim)

        if signal.level == limbdd.root.level:
            limbdd.root = new_signal
            limbdd.refresh()
        return new_signal
    
    @staticmethod
    def split_branches(limbdd, branch_list, move=True):
        assert(isinstance(limbdd, LimBDD))

        if not len(branch_list): return None
        split_limbdd = LimBDDManager.prepare_empty_limbdd(limbdd.root.level + 1) if move else None
        signal_tmp, diverge, update_map = LimBDDManager.move_branches(limbdd.root, branch_list, limbdd, target_limbdd=split_limbdd)
        if split_limbdd: split_limbdd.root = signal_tmp

        if not diverge:
            limbdd.root = LimBDDSignal(limbdd.terminator, weight=0)
            limbdd.refresh()
            return split_limbdd

        LimBDDManager.remove_branches_with_signal_map(limbdd, limbdd.root, update_map)
        return split_limbdd

    @staticmethod
    def find_corresponding_branch(src_branch, pivot_node, lim_accumulated):
        assert(isinstance(src_branch, SimpleBranch) and isinstance(pivot_node, LimBDDNode) and isinstance(lim_accumulated, Lim))

        tgt_branch = SimpleBranch(src_branch.bitstr, src_branch.bitnum)
        pivot = 1
        while pivot_node.level:
            if not pivot_node.is_fallthrough_node:
                if lim_accumulated.act_on_direct_child():
                    tgt_branch.bitstr ^= pivot
                pivot <<= 1
            pivot_node = pivot_node.low.node
            lim_accumulated = Lim.downgrade(lim_accumulated)
        return tgt_branch

    @staticmethod
    def contain_branch(src_node, tgt_node, successor_index, swap_chileren):
        assert(isinstance(src_node, LimBDDNode) and isinstance(tgt_node, LimBDDNode))

        if not src_node.level == tgt_node.level: return False
        if src_node.is_fallthrough_node:
            if tgt_node.is_fallthrough_node and (\
            (    swap_chileren and src_node.low.lim == tgt_node.high.lim and src_node.high.lim == tgt_node.low.lim) or \
            (not swap_chileren and src_node.low.lim == tgt_node.low.lim  and src_node.high.lim == tgt_node.high.lim)):
                return True
        else:
            if not tgt_node.is_fallthrough_node:
                tgt_successor = tgt_node.successors[successor_index ^ swap_chileren]
                if tgt_successor.weight:
                    return True
        return False

    @staticmethod
    def locate_diverge_node(branch, src_pivot, tgt_pivot, lim_accumulated):
        assert(isinstance(branch, SimpleBranch) and isinstance(src_pivot, LimBDDNode) and isinstance(tgt_pivot, LimBDDNode) and isinstance(lim_accumulated, Lim))

        if not src_pivot.level: return src_pivot, tgt_pivot, branch, lim_accumulated

        successor_index, swap_children = branch.bitstr & 1, lim_accumulated.act_on_direct_child()
        if LimBDDManager.contain_branch(src_pivot, tgt_pivot, successor_index, swap_children):
            if src_pivot.is_fallthrough_node:
                return LimBDDManager.locate_diverge_node(branch, src_pivot.low.node, tgt_pivot.low.node, Lim.downgrade(lim_accumulated))
            else:
                src_successor = src_pivot.successors[successor_index]
                tgt_successor = tgt_pivot.successors[successor_index ^ swap_children]
                lim_remaining = Lim.sum(Lim.downgrade(lim_accumulated), Lim.sum(src_successor.lim, tgt_successor.lim))
                return LimBDDManager.locate_diverge_node(branch.copy().rshift(), src_successor.node, tgt_successor.node, lim_remaining)
        return src_pivot, tgt_pivot, branch, lim_accumulated

    @staticmethod
    def locate_merge_node(branch, src_pivot, tgt_pivot, lim_accumulated):
        assert(isinstance(branch, SimpleBranch) and isinstance(src_pivot, LimBDDNode) and isinstance(tgt_pivot, LimBDDNode) and isinstance(lim_accumulated, Lim))

        assert(isinstance(branch, SimpleBranch) and isinstance(src_pivot, LimBDDNode) and isinstance(tgt_pivot, LimBDDNode) and isinstance(lim_accumulated, Lim))

        if not src_pivot.level: return True, src_pivot, tgt_pivot

        successor_index, swap_children = branch.bitstr & 1, lim_accumulated.act_on_direct_child()
        branch_remaining = branch if src_pivot.is_fallthrough_node else branch.copy().rshift()
        next_src_pivot = src_pivot.successors[successor_index].node
        if tgt_pivot.is_branch_node:
            next_lim_accumulate = Lim.downgrade(lim_accumulated)
            child_merged, src_merge_node, tgt_merge_node = LimBDDManager.locate_merge_node(branch_remaining, next_src_pivot, tgt_pivot.successors[0].node, next_lim_accumulate)
            if not child_merged:
                child_merged1, src_merge_node1, tgt_merge_node1 = LimBDDManager.locate_merge_node(branch_remaining, next_src_pivot, tgt_pivot.successors[1].node, next_lim_accumulate)
                if src_merge_node1.level > src_merge_node.level: 
                    child_merged, src_merge_node, tgt_merge_node = child_merged1, src_merge_node1, tgt_merge_node1
            child_merged = False
        else:
            next_tgt_pivot = tgt_pivot.successors[0].node if tgt_pivot.successors[0].weight else tgt_pivot.successors[1].node
            child_merged, src_merge_node, tgt_merge_node = LimBDDManager.locate_merge_node(branch_remaining, next_src_pivot, next_tgt_pivot, Lim.downgrade(lim_accumulated))
        
        if child_merged:
            if (src_pivot.is_fallthrough_node and tgt_pivot.is_fallthrough_node and \
                src_pivot.low.lim.bitstr ^ src_pivot.high.lim.bitstr ^ tgt_pivot.low.lim.bitstr ^ tgt_pivot.high.lim.bitstr == 0) or (\
                not src_pivot.is_fallthrough_node and not tgt_pivot.is_fallthrough_node):
                return True, src_pivot, tgt_pivot
        return False, src_merge_node, tgt_merge_node

    @staticmethod
    def locate_branch_in_new_limbdd(branch, src_limbdd, tgt_limbdd, lim_start):
        assert(isinstance(branch, SimpleBranch) and isinstance(src_limbdd, LimBDD) and isinstance(tgt_limbdd, LimBDD) and isinstance(lim_start, Lim))

        src_diverge_node, tgt_diverge_node, branch_remaining, lim_remainging = \
            LimBDDManager.locate_diverge_node(branch, src_limbdd.root.node, tgt_limbdd.root.node, lim_start)

        if src_diverge_node.level <= 1:
            return True
        elif not src_diverge_node.is_fallthrough_node and tgt_diverge_node.is_onechild_node:
            return True
        
        _, src_merge_node, tgt_merge_node = LimBDDManager.locate_merge_node(branch_remaining, src_diverge_node, tgt_diverge_node, lim_remainging)
        level_cap = src_diverge_node.level - src_merge_node.level
        if level_cap <= 1: return True

        return False
    
    @staticmethod
    def relocate_branch_to(tgt_limbdd, tgt_pivot, src_pivot, branch, lim_accumulate, diverged=False):
        assert(isinstance(tgt_pivot, LimBDDNode) and (src_pivot, LimBDDNode) and isinstance(branch, SimpleBranch) and isinstance(lim_accumulate, Lim) and isinstance(tgt_limbdd, LimBDD))

        successor_index, swap_children = branch.bitstr & 1, lim_accumulate.act_on_direct_child()
        if tgt_pivot.level == 1:
            new_low  = LimBDDSignal(tgt_limbdd.terminator, weight=tgt_pivot.low.weight  ^ src_pivot.successors[0 ^ swap_children].weight)
            new_high = LimBDDSignal(tgt_limbdd.terminator, weight=tgt_pivot.high.weight ^ src_pivot.successors[1 ^ swap_children].weight)
        elif not diverged and LimBDDManager.contain_branch(src_pivot, tgt_pivot, successor_index, swap_children):
            if src_pivot.is_fallthrough_node:
                new_low  = LimBDDManager.relocate_branch_to(tgt_limbdd, tgt_pivot.low.node, src_pivot.low.node, branch, Lim.downgrade(lim_accumulate))
                new_high = new_low.copy()
                if new_low.weight : new_low.lim  = Lim.sum(new_low.lim , tgt_pivot.low.lim )
                if new_high.weight: new_high.lim = Lim.sum(new_high.lim, tgt_pivot.high.lim)
            else:
                src_successor = src_pivot.successors[successor_index]
                tgt_successor = tgt_pivot.successors[successor_index ^ swap_children]
                next_lim_accumulate = Lim.sum(Lim.downgrade(lim_accumulate), Lim.sum(src_successor.lim, tgt_successor.lim))

                new_successors = [tgt_pivot.low.copy(), tgt_pivot.high.copy()]
                new_successors[successor_index ^ swap_children] = \
                    LimBDDManager.relocate_branch_to(tgt_limbdd, tgt_successor.node, src_successor.node, branch.copy().rshift(), next_lim_accumulate)
                if new_successors[successor_index ^ swap_children].weight:
                    new_successors[successor_index ^ swap_children].lim = Lim.sum(new_successors[successor_index ^ swap_children].lim, tgt_successor.lim)
                new_low, new_high = new_successors[0], new_successors[1]
        else:
            if src_pivot.is_fallthrough_node:
                new_successors = [None, None]
                for i in range(2):
                    tgt_successor_index = i ^ swap_children
                    src_successor = src_pivot.successors[i]
                    tgt_successor = tgt_pivot.successors[tgt_successor_index]
                        
                    if tgt_successor.weight:
                        next_lim_accumulate = Lim.sum(Lim.downgrade(lim_accumulate), Lim.sum(src_successor.lim, tgt_successor.lim))
                        new_successors[tgt_successor_index] = LimBDDManager.relocate_branch_to(tgt_limbdd, tgt_successor.node, src_successor.node, branch, next_lim_accumulate, True)
                        if new_successors[tgt_successor_index].weight:
                            new_successors[tgt_successor_index].lim = Lim.sum(new_successors[tgt_successor_index].lim, tgt_successor.lim)
                    else:
                        next_lim_accumulate = Lim.sum(Lim.downgrade(lim_accumulate), src_successor.lim)
                        new_successors[tgt_successor_index] = tgt_limbdd.insert_branch_from_another(src_successor.node, branch, next_lim_accumulate)
            else:
                new_successors = [None, None]
                tgt_branch_index = successor_index ^ swap_children
                new_successors[tgt_branch_index ^ 1] = tgt_pivot.successors[tgt_branch_index ^ 1].copy()

                src_successor = src_pivot.successors[successor_index]
                tgt_successor = tgt_pivot.successors[tgt_branch_index]

                if tgt_successor.weight:
                    next_lim_accumulate = Lim.sum(Lim.downgrade(lim_accumulate), Lim.sum(src_successor.lim, tgt_successor.lim))
                    new_successors[tgt_branch_index] = LimBDDManager.relocate_branch_to(tgt_limbdd, tgt_successor.node, src_successor.node, branch.copy().rshift(), next_lim_accumulate, True)
                    if new_successors[tgt_branch_index].weight:
                        new_successors[tgt_branch_index].lim = Lim.sum(new_successors[tgt_branch_index].lim, tgt_successor.lim)
                else:
                    next_lim_accumulate = Lim.sum(Lim.downgrade(lim_accumulate), src_successor.lim)
                    new_successors[tgt_branch_index] = tgt_limbdd.insert_branch_from_another(src_successor.node, branch.copy().rshift(), next_lim_accumulate)
            new_low, new_high = new_successors[0], new_successors[1]

        new_node = LimBDDNode(tgt_limbdd.get_node_idx(), tgt_pivot.level, new_low, new_high, var_index=tgt_pivot.var_index)
        new_signal = tgt_limbdd.insert_node_by_topo(new_node)
        return new_signal

    @staticmethod
    def try_absorb_branch(tgt_limbdd, src_limbdd):
        assert(isinstance(tgt_limbdd, LimBDD) and isinstance(src_limbdd, LimBDD))
        '''
        tgt_limbdd.show(f"figure/debug/fig{config['debug_figure_counter']}_tgt_before")
        src_limbdd.show(f"figure/debug/fig{config['debug_figure_counter']}_src_before")
        print(bin(tgt_limbdd.get_truth_table())[2:].zfill(1 << tgt_limbdd.root.level))
        print(bin(src_limbdd.get_truth_table())[2:].zfill(1 << src_limbdd.root.level))
        '''
        if not tgt_limbdd.root.weight or not src_limbdd.root.weight: return
        try:
            src_branches = LimBDDManager.branch_process(src_limbdd.root.node, [src_limbdd.terminator])[src_limbdd.terminator]
            tgt_branches = LimBDDManager.branch_process(tgt_limbdd.root.node, [tgt_limbdd.terminator])[tgt_limbdd.terminator]
        except:
            return
        
        swap = len(src_branches) > len(tgt_branches)
        if swap:
            tgt_limbdd  , src_limbdd   = src_limbdd  , tgt_limbdd
            tgt_branches, src_branches = src_branches, tgt_branches

        lim_start = Lim.sum(tgt_limbdd.root.lim, src_limbdd.root.lim)
        remove_branches = []
        for branch in src_branches:
            success = LimBDDManager.locate_branch_in_new_limbdd(branch, src_limbdd, tgt_limbdd, lim_start)
            if success:
                lim_accumulate = Lim.sum(tgt_limbdd.root.lim, src_limbdd.root.lim)
                new_root = LimBDDManager.relocate_branch_to(tgt_limbdd, tgt_limbdd.root.node, src_limbdd.root.node, branch, lim_accumulate)
                if new_root.weight: new_root.lim = Lim.sum(new_root.lim, tgt_limbdd.root.lim)
                tgt_limbdd.root = new_root
                remove_branches.append(branch)

        tgt_limbdd.refresh()
        if len(remove_branches): tgt_limbdd.split_performed = False

        if len(remove_branches) == len(src_branches):
            src_limbdd.root = LimBDDSignal(src_limbdd.terminator, weight=0)
            src_limbdd.refresh()
        elif len(remove_branches):
            LimBDDManager.split_branches(src_limbdd, remove_branches, move=False)

        if swap:
            tgt_limbdd  , src_limbdd   = src_limbdd  , tgt_limbdd
            tgt_branches, src_branches = src_branches, tgt_branches
        '''
        tgt_limbdd.show(f"figure/debug/fig{config['debug_figure_counter']}_tgt_after")
        src_limbdd.show(f"figure/debug/fig{config['debug_figure_counter']}_src_after")
        print(bin(tgt_limbdd.get_truth_table())[2:].zfill(1 << tgt_limbdd.root.level))
        print(bin(src_limbdd.get_truth_table())[2:].zfill(1 << src_limbdd.root.level))
        config['debug_figure_counter'] += 1
        '''
        return