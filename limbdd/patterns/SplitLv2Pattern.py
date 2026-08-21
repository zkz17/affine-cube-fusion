from limbdd.Lim import Lim
from limbdd.Node import LimBDDNode, LimBDDSignal
from limbdd.LimBDD import LimBDD
from limbdd.patterns.SplitPattern import SplitPattern

class SplitLv2Pattern(SplitPattern):
    def __init__(self):
        super().__init__()

    def same_topo(self, node1, node2):
        assert(isinstance(node1, LimBDDNode) and isinstance(node2, LimBDDNode))

        if not node1.level == node2.level: return False
        if node1.is_onechild_node and node2.is_onechild_node: return True
        if node1.is_fallthrough_node and node2.is_fallthrough_node:
            lim_sum_1 = Lim.sum(node1.low.lim, node1.high.lim)
            lim_sum_2 = Lim.sum(node2.low.lim, node2.high.lim)
            if lim_sum_1.bitstr == lim_sum_2.bitstr: return True
        return False
    
    def create_connect_nodes(self, connect_list, lim_sum_l, lim_sum_r, signal_l, signal_r, limbdd):
        last_level_signal_l, last_level_signal_r = signal_l, signal_r
        for connect_level_info in reversed(connect_list):
            succ_l_level, succ_r_level, is_oc, l_i, r_i, lim_sum_level = connect_level_info

            if last_level_signal_l:
                connect_successors_level = [LimBDDSignal(limbdd.terminator, weight=0), LimBDDSignal(limbdd.terminator, weight=0)]
                if is_oc:
                    connect_successors_level[l_i] = last_level_signal_l
                    connect_successors_level[l_i].lim = Lim.sum(connect_successors_level[l_i].lim, succ_l_level.successors[l_i].lim)
                else:
                    connect_successors_level[0] = last_level_signal_l
                    connect_successors_level[1] = last_level_signal_l.copy()
                    connect_successors_level[0].lim = Lim.sum(connect_successors_level[0].lim, succ_l_level.successors[0].lim)
                    connect_successors_level[1].lim = Lim.sum(connect_successors_level[1].lim, succ_l_level.successors[1].lim)

                connect_node_level = LimBDDNode(limbdd.get_node_idx(), succ_l_level.level, connect_successors_level[0], connect_successors_level[1], var_index=succ_l_level.var_index)
                last_level_signal_l = limbdd.insert_node_by_topo(connect_node_level)

                if lim_sum_l:
                    lim_sum_l = Lim.sum(lim_sum_l, lim_sum_level)
                    lim_sum_l = Lim.extend(lim_sum_l, tail=(l_i ^ r_i))

            if last_level_signal_r:
                connect_successors_level = [LimBDDSignal(limbdd.terminator, weight=0), LimBDDSignal(limbdd.terminator, weight=0)]
                if is_oc:
                    connect_successors_level[r_i] = last_level_signal_r
                    connect_successors_level[r_i].lim = Lim.sum(connect_successors_level[r_i].lim, succ_r_level.successors[r_i].lim)
                else:
                    connect_successors_level[0] = last_level_signal_r
                    connect_successors_level[1] = last_level_signal_r.copy()
                    connect_successors_level[0].lim = Lim.sum(connect_successors_level[0].lim, succ_r_level.successors[0].lim)
                    connect_successors_level[1].lim = Lim.sum(connect_successors_level[1].lim, succ_r_level.successors[1].lim)

                connect_node_level = LimBDDNode(limbdd.get_node_idx(), succ_r_level.level, connect_successors_level[0], connect_successors_level[1], var_index=succ_r_level.var_index)
                last_level_signal_r = limbdd.insert_node_by_topo(connect_node_level)

                if lim_sum_r:
                    lim_sum_r = Lim.sum(lim_sum_r, lim_sum_level)
                    lim_sum_r = Lim.extend(lim_sum_r, tail=(l_i ^ r_i))

        return last_level_signal_l, last_level_signal_r, lim_sum_l, lim_sum_r

    def identify_candidates(self, limbdd, level):
        assert(isinstance(limbdd, LimBDD) and level <= limbdd.root.level)

        if level < 2: return []

        candidates, cache = [], {}
        for node in limbdd.level2nodes[level].values():
            if not node.is_branch_node: continue

            succ_l, succ_r = node.successors[0].node, node.successors[1].node

            key = (succ_l.idx, succ_r.idx) if succ_l.idx < succ_r.idx else (succ_r.idx, succ_l.idx)
            if key in cache:
                connect_list, succ_l, succ_r, pattern_type, succ_info = cache[key]
                candidates.append((node, connect_list, succ_l, succ_r, pattern_type, succ_info))
                continue

            connect_list = []
            while self.same_topo(succ_l, succ_r):
                if succ_l.is_onechild_node:
                    is_oc = 1
                    l_i = 0 if succ_l.low.weight else 1
                    r_i = 0 if succ_r.low.weight else 1
                else:
                    is_oc, l_i, r_i = 0, 0, 0
                lim_sum = Lim.sum(succ_l.successors[l_i].lim, succ_r.successors[r_i].lim)
                connect_list.append((succ_l, succ_r, is_oc, l_i, r_i, lim_sum))
                succ_l, succ_r = succ_l.successors[l_i].node, succ_r.successors[r_i].node

            if succ_l.is_branch_node and succ_r.is_branch_node:
                pattern_type = 1

                succ_l_i, succ_r_i = -1, -1
                for i in range(2):
                    for j in range(2):
                        if succ_l.successors[i].node is succ_r.successors[j].node:
                            succ_l_i, succ_r_i = i, j
                            break
                if succ_l_i == -1: continue
                if succ_l.successors[succ_l_i ^ 1].node is succ_r.successors[succ_r_i ^ 1].node: pattern_type = 4

                succ_info = (succ_l_i, succ_r_i)
            elif (succ_l.is_onechild_node and succ_r.is_branch_node) or (succ_l.is_branch_node and succ_r.is_onechild_node):
                pattern_type = 2

                succ_l_is_oc, succ_r_is_oc = int(succ_l.is_onechild_node), int(succ_r.is_onechild_node)
                succ_l_i, succ_r_i = -1, -1
                for i in range(2):
                    for j in range(2):
                        if succ_l.successors[i].node is succ_r.successors[j].node:
                            succ_l_i, succ_r_i = i, j
                            break
                if succ_l_i == -1: continue

                succ_info = (succ_l_is_oc, succ_r_is_oc, succ_l_i, succ_r_i)
            elif (succ_l.is_onechild_node and succ_r.is_fallthrough_node) or (succ_l.is_fallthrough_node and succ_r.is_onechild_node):
                pattern_type = 3

                succ_l_is_oc, succ_r_is_oc = int(succ_l.is_onechild_node), int(succ_r.is_onechild_node)
                succ_l_succ_i = 0 if succ_l.low.weight else 1
                succ_r_succ_i = 0 if succ_r.low.weight else 1
                if not succ_l.successors[succ_l_succ_i].node is succ_r.successors[succ_r_succ_i].node: continue

                succ_info = (succ_l_is_oc, succ_r_is_oc, succ_l_succ_i, succ_r_succ_i)
            elif succ_l.is_fallthrough_node and succ_r.is_fallthrough_node:
                if not succ_l.low.node is succ_r.low.node: continue

                lim_sum_l = Lim.sum(succ_l.low.lim, succ_l.high.lim)
                lim_sum_r = Lim.sum(succ_r.low.lim, succ_r.high.lim)
                lim_sum = Lim.sum(lim_sum_l, lim_sum_r)
                if lim_sum.bitstr.bit_count() > 2: continue

                pattern_type = 5
                if abs(lim_sum_l.bitstr.bit_count() - 2) <= abs(lim_sum_r.bitstr.bit_count() - 2):
                    fuse_to_succ_l = 1
                    split_succ_i = 0 if (succ_l.low.lim.bitstr ^ lim_sum.bitstr).bit_count() <= (succ_l.high.lim.bitstr ^ lim_sum.bitstr).bit_count() else 1
                else:
                    fuse_to_succ_l = 0
                    split_succ_i = 0 if (succ_r.low.lim.bitstr ^ lim_sum.bitstr).bit_count() <= (succ_r.high.lim.bitstr ^ lim_sum.bitstr).bit_count() else 1
                
                succ_info = (fuse_to_succ_l, split_succ_i, lim_sum)
            else: continue

            cache[key] = (connect_list, succ_l, succ_r, pattern_type, succ_info)
            candidates.append((node, connect_list, succ_l, succ_r, pattern_type, succ_info))
        return candidates

    def process_split(self, limbdd, split_limbdd, candidates):
        terminator_constructor_bottom = lambda _, __: None
        copy_map = {1 : LimBDDSignal(split_limbdd.terminator)}

        split_targets, split_info, cache = [], [], {}
        for candidate_info in candidates:
            node, connect_list, succ_l, succ_r, pattern_type, succ_info = candidate_info
            key, swap_key = (node.low.node.idx, node.high.node.idx), (node.high.node.idx, node.low.node.idx)

            if key in cache:
                new_successors_cache, connect_signals_cache = cache[key]
                new_successors = [new_successors_cache[0].copy(), new_successors_cache[1].copy()]
                connect_signals = [connect_signals_cache[0].copy(), connect_signals_cache[1].copy()]
            elif swap_key in cache:
                new_successors_cache, connect_signals_cache = cache[swap_key]
                new_successors = [new_successors_cache[1].copy(), new_successors_cache[0].copy()]
                connect_signals = [connect_signals_cache[1].copy(), connect_signals_cache[0].copy()]
            elif pattern_type == 1:
                succ_l_i, succ_r_i = succ_info
                split_successor = succ_l.successors[succ_l_i]

                # Copy downstream to split_limbdd.
                limbdd.clear_visited()
                split_signal = SplitPattern.copy_nodes_between(split_successor, [limbdd.terminator], split_limbdd, terminator_constructor_bottom, copy_map=copy_map)

                # Connect downstream to upstream.
                lim_sum = Lim.sum(succ_l.successors[succ_l_i].lim, succ_r.successors[succ_r_i].lim)
                lim_sum = Lim.extend(lim_sum, tail=(succ_l_i ^ succ_r_i))

                connect_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                connect_successors[succ_l_i] = split_signal.copy()
                connect_node = LimBDDNode(split_limbdd.get_node_idx(), succ_l.level, connect_successors[0], connect_successors[1], var_index=succ_l.var_index)
                last_level_signal = split_limbdd.insert_node_by_topo(connect_node)

                last_level_signal, _, lim_sum, __ = self.create_connect_nodes(connect_list, lim_sum, None, last_level_signal, None, split_limbdd)

                connect_signals = [None, None]
                connect_signals[0] = last_level_signal
                connect_signals[1] = last_level_signal.copy()
                connect_signals[1].lim = Lim.sum(connect_signals[1].lim, lim_sum)

                # Prepare original limbdd update.
                new_succ_l_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                new_succ_l_successors[succ_l_i ^ 1] = succ_l.successors[succ_l_i ^ 1].copy()
                new_succ_r_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                new_succ_r_successors[succ_r_i ^ 1] = succ_r.successors[succ_r_i ^ 1].copy()

                new_succ_l = LimBDDNode(limbdd.get_node_idx(), succ_l.level, new_succ_l_successors[0], new_succ_l_successors[1], var_index=succ_l.var_index)
                last_level_signal_l = limbdd.insert_node_by_topo(new_succ_l)
                new_succ_r = LimBDDNode(limbdd.get_node_idx(), succ_r.level, new_succ_r_successors[0], new_succ_r_successors[1], var_index=succ_r.var_index)
                last_level_signal_r = limbdd.insert_node_by_topo(new_succ_r)

                last_level_signal_l, last_level_signal_r, _, __ = self.create_connect_nodes(connect_list, None, None, last_level_signal_l, last_level_signal_r, limbdd)
                new_successors = [last_level_signal_l, last_level_signal_r]

                cache[key] = ([new_successors[0].copy(), new_successors[1].copy()], [connect_signals[0].copy(), connect_signals[1].copy()])
            elif pattern_type == 2:
                succ_l_is_oc, succ_r_is_oc, succ_l_i, succ_r_i = succ_info
                split_successor = succ_l.successors[succ_l_i]

                 # Copy downstream to split_limbdd.
                limbdd.clear_visited()
                split_signal = SplitPattern.copy_nodes_between(split_successor, [limbdd.terminator], split_limbdd, terminator_constructor_bottom, copy_map=copy_map)

                # Connect downstream to upstream.
                lim_sum = Lim.sum(succ_l.successors[succ_l_i].lim, succ_r.successors[succ_r_i].lim)
                lim_sum = Lim.extend(lim_sum, tail=(succ_l_i ^ succ_r_i))

                connect_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                connect_successors[succ_l_i] = split_signal.copy()
                connect_node = LimBDDNode(split_limbdd.get_node_idx(), succ_l.level, connect_successors[0], connect_successors[1], var_index=succ_l.var_index)
                last_level_signal = split_limbdd.insert_node_by_topo(connect_node)

                last_level_signal, _, lim_sum, __ = self.create_connect_nodes(connect_list, lim_sum, None, last_level_signal, None, split_limbdd)

                connect_signals = [None, None]
                connect_signals[0] = last_level_signal
                connect_signals[1] = last_level_signal.copy()
                connect_signals[1].lim = Lim.sum(connect_signals[1].lim, lim_sum)

                # Prepare original limbdd update.
                if succ_r_is_oc:
                    new_succ_l_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                    new_succ_l_successors[succ_l_i ^ 1] = succ_l.successors[succ_l_i ^ 1].copy()

                    new_succ_l = LimBDDNode(limbdd.get_node_idx(), succ_l.level, new_succ_l_successors[0], new_succ_l_successors[1], var_index=succ_l.var_index)
                    last_level_signal_l = limbdd.insert_node_by_topo(new_succ_l)

                    last_level_signal_l, _, __, ___ = self.create_connect_nodes(connect_list, None, None, last_level_signal_l, None, limbdd)
                    new_successors = [last_level_signal_l, LimBDDSignal(split_limbdd.terminator, weight=0)]
                else:
                    new_succ_r_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                    new_succ_r_successors[succ_r_i ^ 1] = succ_r.successors[succ_r_i ^ 1].copy()

                    new_succ_r = LimBDDNode(limbdd.get_node_idx(), succ_r.level, new_succ_r_successors[0], new_succ_r_successors[1], var_index=succ_r.var_index)
                    last_level_signal_r = limbdd.insert_node_by_topo(new_succ_r)

                    _, last_level_signal_r, __, ___ = self.create_connect_nodes(connect_list, None, None, None, last_level_signal_r, limbdd)
                    new_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), last_level_signal_r]

                cache[key] = ([new_successors[0].copy(), new_successors[1].copy()], [connect_signals[0].copy(), connect_signals[1].copy()])
            elif pattern_type == 3:
                succ_l_is_oc, succ_r_is_oc, succ_l_succ_i, succ_r_succ_i = succ_info
                if succ_l_is_oc: succ_oc, succ_ot, common_succ, succ_oc_succ_i = succ_l, succ_r, succ_l.successors[succ_l_succ_i].node, succ_l_succ_i
                else: succ_oc, succ_ot, common_succ, succ_oc_succ_i = succ_r, succ_l, succ_r.successors[succ_r_succ_i].node, succ_r_succ_i
                
                # Construct other_successor.
                lim = succ_oc.successors[succ_oc_succ_i].lim
                for succ_ot_succ in succ_ot.successors:
                    lim = Lim.sum(lim, succ_ot_succ.lim)
                other_successor = LimBDDSignal(common_succ, lim=lim)

                # Copy downstream to split_limbdd.
                limbdd.clear_visited()
                split_signal = SplitPattern.copy_nodes_between(succ_ot.low, [limbdd.terminator], split_limbdd, terminator_constructor_bottom, copy_map=copy_map)

                # Connect downstream to upstream.
                lim_sum_0 = Lim.sum(succ_oc.successors[succ_oc_succ_i].lim, succ_ot.successors[succ_oc_succ_i].lim)
                lim_sum_0 = Lim.extend(lim_sum_0)
                lim_sum_1 = Lim.sum(succ_oc.successors[succ_oc_succ_i].lim, succ_ot.successors[succ_oc_succ_i ^ 1].lim)
                lim_sum_1 = Lim.extend(lim_sum_1, tail=True)
                lim_sum = lim_sum_0 if lim_sum_0.bitstr.bit_count() <= lim_sum_1.bitstr.bit_count() else lim_sum_1

                connect_successors = [None, None]
                connect_successors[0] = split_signal
                connect_successors[1] = split_signal.copy()
                connect_successors[1].lim = Lim.sum(Lim.sum(succ_ot.low.lim, succ_ot.high.lim), connect_successors[1].lim)
                connect_node = LimBDDNode(split_limbdd.get_node_idx(), succ_ot.level, connect_successors[0], connect_successors[1], var_index=succ_ot.var_index)
                last_level_signal = split_limbdd.insert_node_by_topo(connect_node)

                if succ_l_is_oc: _, last_level_signal, __, lim_sum = self.create_connect_nodes(connect_list, None, lim_sum, None, last_level_signal, split_limbdd)
                else: last_level_signal, _, lim_sum, __ = self.create_connect_nodes(connect_list, lim_sum, None, last_level_signal, None, split_limbdd)

                connect_signals = [None, None]
                connect_signals[0] = last_level_signal
                connect_signals[1] = last_level_signal.copy()
                connect_signals[succ_r_is_oc].lim = Lim.sum(connect_signals[succ_r_is_oc].lim, lim_sum)

                # Prepare original limbdd update.
                new_succ_oc_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                new_succ_oc_successors[succ_oc_succ_i ^ 1] = other_successor
                new_succ_oc = LimBDDNode(limbdd.get_node_idx(), succ_oc.level, new_succ_oc_successors[0], new_succ_oc_successors[1], var_index=succ_oc.var_index)
                last_level_signal_oc = limbdd.insert_node_by_topo(new_succ_oc)

                if succ_l_is_oc: last_level_signal_oc, _, __, ___ = self.create_connect_nodes(connect_list, None, None, last_level_signal_oc, None, limbdd)
                else: _, last_level_signal_oc, __, ___ = self.create_connect_nodes(connect_list, None, None, None, last_level_signal_oc, limbdd)

                new_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                new_successors[succ_r_is_oc] = last_level_signal_oc

                cache[key] = ([new_successors[0].copy(), new_successors[1].copy()], [connect_signals[0].copy(), connect_signals[1].copy()])
            elif pattern_type == 4:
                succ_l_i, succ_r_i = succ_info
                split_successor = succ_l.successors[succ_l_i]

                # Copy downstream to split_limbdd.
                limbdd.clear_visited()
                split_signal = SplitPattern.copy_nodes_between(split_successor, [limbdd.terminator], split_limbdd, terminator_constructor_bottom, copy_map=copy_map)

                # Connect downstream to upstream.
                lim_sum = Lim.sum(succ_l.successors[succ_l_i].lim, succ_r.successors[succ_r_i].lim)
                lim_sum = Lim.extend(lim_sum, tail=(succ_l_i ^ succ_r_i))

                connect_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                connect_successors[succ_l_i] = split_signal.copy()
                connect_node = LimBDDNode(split_limbdd.get_node_idx(), succ_l.level, connect_successors[0], connect_successors[1], var_index=succ_l.var_index)
                last_level_signal = split_limbdd.insert_node_by_topo(connect_node)

                last_level_signal, _, lim_sum, __ = self.create_connect_nodes(connect_list, lim_sum, None, last_level_signal, None, split_limbdd)

                connect_signals = [None, None]
                connect_signals[0] = last_level_signal
                connect_signals[1] = last_level_signal.copy()
                connect_signals[1].lim = Lim.sum(connect_signals[1].lim, lim_sum)

                # Prepare original limbdd update.
                new_succ_l_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                new_succ_l_successors[succ_l_i ^ 1] = succ_l.successors[succ_l_i ^ 1].copy()
                new_succ_l = LimBDDNode(limbdd.get_node_idx(), succ_l.level, new_succ_l_successors[0], new_succ_l_successors[1], var_index=succ_l.var_index)
                last_level_signal_l = limbdd.insert_node_by_topo(new_succ_l)

                lim_sum_l = Lim.sum(succ_l.successors[succ_l_i ^ 1].lim, succ_r.successors[succ_r_i ^ 1].lim)
                lim_sum_l = Lim.extend(lim_sum_l, tail=(succ_l_i ^ succ_r_i))

                last_level_signal_l, _, lim_sum_l, __ = self.create_connect_nodes(connect_list, lim_sum_l, None, last_level_signal_l, None, limbdd)
                
                new_successors = [None, None]
                new_successors[0] = last_level_signal_l
                new_successors[1] = last_level_signal_l.copy()
                new_successors[1].lim = Lim.sum(new_successors[1].lim, lim_sum_l)

                cache[key] = ([new_successors[0].copy(), new_successors[1].copy()], [connect_signals[0].copy(), connect_signals[1].copy()])
            elif pattern_type == 5:
                fuse_to_succ_l, split_succ_i, lim_sum_candidates = succ_info
                if fuse_to_succ_l:
                    split_successor = succ_r.successors[split_succ_i]
                else:
                    split_successor = succ_l.successors[split_succ_i]

                # Copy downstream to split_limbdd.
                limbdd.clear_visited()
                split_signal = SplitPattern.copy_nodes_between(split_successor, [limbdd.terminator], split_limbdd, terminator_constructor_bottom, copy_map=copy_map)
                split_signal = split_limbdd.fuse_signal(split_signal, split_successor, lim_sum_candidates)

                # Connect downstream to upstream.
                if split_signal.weight:
                    connect_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                    connect_successors[split_succ_i] = split_signal.copy()
                    connect_node = LimBDDNode(split_limbdd.get_node_idx(), succ_l.level, connect_successors[0], connect_successors[1], var_index=succ_l.var_index)
                    last_level_signal = split_limbdd.insert_node_by_topo(connect_node)

                    if fuse_to_succ_l:
                        _, last_level_signal, __, ___ = self.create_connect_nodes(connect_list, None, None, None, last_level_signal, split_limbdd)
                        connect_signals = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                        connect_signals[1] = last_level_signal
                    else:
                        last_level_signal, _, __, ___ = self.create_connect_nodes(connect_list, None, None, last_level_signal, None, split_limbdd)
                        connect_signals = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                        connect_signals[0] = last_level_signal
                else:
                    connect_signals = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                
                # Prepare original limbdd update.
                lim_sum = Lim.sum(succ_l.successors[split_succ_i ^ 1].lim, succ_r.successors[split_succ_i ^ 1].lim)
                lim_sum = Lim.extend(lim_sum)
                
                if fuse_to_succ_l:
                    last_level_signal = LimBDDSignal(succ_l)
                    last_level_signal, _, lim_sum, __ = self.create_connect_nodes(connect_list, lim_sum, None, last_level_signal, None, limbdd)
                else:
                    last_level_signal = LimBDDSignal(succ_r)
                    _, last_level_signal, __, lim_sum = self.create_connect_nodes(connect_list, None, lim_sum, None, last_level_signal, limbdd)
                
                new_successors = [None, None]
                new_successors[0] = last_level_signal
                new_successors[1] = last_level_signal.copy()
                new_successors[fuse_to_succ_l].lim = Lim.sum(new_successors[fuse_to_succ_l].lim, lim_sum)

                cache[key] = ([new_successors[0].copy(), new_successors[1].copy()], [connect_signals[0].copy(), connect_signals[1].copy()])
            else: continue

            if new_successors[0].weight : new_successors[0].lim  = Lim.sum(new_successors[0].lim, node.low.lim  )
            if new_successors[1].weight : new_successors[1].lim  = Lim.sum(new_successors[1].lim, node.high.lim )
            if connect_signals[0].weight: connect_signals[0].lim = Lim.sum(connect_signals[0].lim, node.low.lim )
            if connect_signals[1].weight: connect_signals[1].lim = Lim.sum(connect_signals[1].lim, node.high.lim)
            
            split_targets.append(node)
            split_info.append((new_successors, connect_signals))

        # Define upstream terminator constructor.
        def terminator_constructor_upside(node, successor):
            assert(isinstance(node, LimBDDNode) and isinstance(successor, LimBDDSignal))

            pos = split_targets.index(node)
            _, connect_signals = split_info[pos]
            if successor.node is node.low.node: return connect_signals[0].copy()
            else: return connect_signals[1].copy()
            
        # Copy upstream to split_limbdd.
        limbdd.clear_visited()
        split_limbdd.root = self.copy_nodes_between(limbdd.root, split_targets, split_limbdd, terminator_constructor_upside, copy_map=copy_map)

        return split_limbdd, split_targets, split_info