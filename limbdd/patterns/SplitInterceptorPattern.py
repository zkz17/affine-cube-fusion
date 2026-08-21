from limbdd.Lim import Lim
from limbdd.Node import LimBDDNode, LimBDDSignal
from limbdd.LimBDD import LimBDD
from limbdd.patterns.SplitPattern import SplitPattern

class SplitInterceptorPattern(SplitPattern):
    def __init__(self):
        super().__init__()

    def identify_candidates(self, limbdd, level):
        assert(isinstance(limbdd, LimBDD) and level <= limbdd.root.level)

        if level < 3: return []

        candidates, cache = [], {}
        for node in limbdd.level2nodes[level].values():
            if not node.is_branch_node: continue

            succ_ft, succ_br = node.successors[0].node, node.successors[1].node
            if succ_ft.is_branch_node: succ_ft, succ_br = succ_br, succ_ft
            if not succ_ft.is_fallthrough_node or not succ_br.is_branch_node: continue

            key = (succ_br.idx, succ_ft.idx) if succ_br.idx < succ_ft.idx else (succ_ft.idx, succ_br.idx)
            if key in cache:
                succ_br_succ_ft, succ_br_succ_oc, bottom, split_i, bit_count = cache[key]
                candidates.append((node, succ_br, succ_ft, succ_br_succ_ft, succ_br_succ_oc, bottom, split_i, bit_count))
                continue

            succ_ft_succ = succ_ft.low.node
            if not succ_ft_succ.is_onechild_node: continue
            succ_ft_succ_i = 0 if succ_ft_succ.low.weight else 1

            succ_br_succ_ft, succ_br_succ_oc = succ_br.successors[0].node, succ_br.successors[1].node
            if succ_br_succ_ft.is_onechild_node: succ_br_succ_ft, succ_br_succ_oc = succ_br_succ_oc, succ_br_succ_ft
            if not succ_br_succ_oc is succ_ft_succ or not succ_br_succ_ft.is_fallthrough_node: continue

            bottom = succ_ft_succ.successors[succ_ft_succ_i].node
            if not succ_br_succ_ft.low.node is bottom: continue

            lim_sum_ft = Lim.sum(succ_ft.low.lim, succ_ft.high.lim)
            lim_sum_br = Lim.sum(succ_br.low.lim, succ_br.high.lim)
            lim_sum = Lim.sum(lim_sum_br, lim_sum_ft)
            lim_sum = Lim.sum(lim_sum, Lim.extend(succ_ft_succ.successors[succ_ft_succ_i].lim))

            lim_sum_0 = Lim.sum(lim_sum, Lim.extend(succ_br_succ_ft.high.lim , tail=(succ_ft_succ_i==0)))
            lim_sum_1 = Lim.sum(lim_sum, Lim.extend(succ_br_succ_ft.low.lim, tail=(succ_ft_succ_i==1)))
            bit_count_0 = lim_sum_0.bitstr.bit_count()
            bit_count_1 = lim_sum_1.bitstr.bit_count()
            bit_count, split_i = (bit_count_0, 0) if bit_count_0 <= bit_count_1 else (bit_count_1, 1)
            if bit_count > 2: continue

            cache[key] = (succ_br_succ_ft, succ_br_succ_oc, bottom, split_i, bit_count)
            candidates.append((node, succ_br, succ_ft, succ_br_succ_ft, succ_br_succ_oc, bottom, split_i, bit_count))
        return candidates

    def process_split(self, limbdd, split_limbdd, candidates):
        terminator_constructor_bottom = lambda _, __: None
        copy_map = {1 : LimBDDSignal(split_limbdd.terminator)}

        split_targets, split_info = [], []
        while len(candidates):
            node, succ_br, succ_ft, succ_br_succ_ft, succ_br_succ_oc, bottom, split_i, bit_count = candidates.pop()
            succ_br_i = 0 if node.low.node is succ_br else 1
            succ_br_succ_ft_i = 0 if succ_br.low.node is succ_br_succ_ft else 1

            if bit_count:
                # Construct split_successor.
                succ_br_succ_oc_succ_i = 0 if succ_br_succ_oc.successors[0].weight else 1
                split_successor = succ_br_succ_oc.successors[succ_br_succ_oc_succ_i]

                # Copy downstream to split_limbdd.
                limbdd.clear_visited()
                split_signal = SplitPattern.copy_nodes_between(split_successor, [limbdd.terminator], split_limbdd, terminator_constructor_bottom, copy_map=copy_map)

                # Connect downstream to upstream.
                connect_successors_level1 = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                split_signal.lim = Lim.sum(split_signal.lim, Lim.sum(succ_br_succ_ft.low.lim, succ_br_succ_ft.high.lim))
                connect_successors_level1[succ_br_succ_oc_succ_i ^ 1] = split_signal
                connect_node_level1 = LimBDDNode(split_limbdd.get_node_idx(), succ_br_succ_oc.level, connect_successors_level1[0], connect_successors_level1[1], var_index=succ_br_succ_oc.var_index)

                connect_signal_level1 = split_limbdd.insert_node_by_topo(connect_node_level1)

                connect_successors_oc_level2 = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                connect_successors_oc_level2[succ_br_succ_ft_i ^ 1] = connect_signal_level1.copy()
                connect_successors_oc_level2[succ_br_succ_ft_i ^ 1].lim = Lim.sum(connect_successors_oc_level2[succ_br_succ_ft_i ^ 1].lim, succ_br.successors[succ_br_succ_ft_i ^ 1].lim)
                connect_node_oc_level2 = LimBDDNode(split_limbdd.get_node_idx(), succ_br.level, connect_successors_oc_level2[0], connect_successors_oc_level2[1], var_index=succ_br.var_index)

                connect_signal_oc_level2 = split_limbdd.insert_node_by_topo(connect_node_oc_level2)
                connect_signal_oc_level2.lim = Lim.sum(connect_signal_oc_level2.lim, node.successors[succ_br_i].lim)

                connect_successors_ft_level2 = [None, None]
                connect_successors_ft_level2[0] = connect_signal_level1.copy()
                connect_successors_ft_level2[1] = connect_signal_level1.copy()
                connect_successors_ft_level2[0].lim = Lim.sum(connect_successors_ft_level2[0].lim, succ_ft.low.lim)
                connect_successors_ft_level2[1].lim = Lim.sum(connect_successors_ft_level2[1].lim, succ_ft.high.lim)
                connect_node_ft_level2 = LimBDDNode(split_limbdd.get_node_idx(), succ_ft.level, connect_successors_ft_level2[0], connect_successors_ft_level2[1], var_index=succ_br.var_index)

                connect_signal_ft_level2 = split_limbdd.insert_node_by_topo(connect_node_ft_level2)
                connect_signal_ft_level2.lim = Lim.sum(connect_signal_ft_level2.lim, node.successors[succ_br_i ^ 1].lim)
                '''
                print(str(split_signal.lim))
                print(str(node.successors[succ_br_i].lim), str(node.successors[succ_br_i ^ 1].lim))
                print(str(connect_signal_oc_level2.lim), str(connect_signal_ft_level2.lim))
                '''
                # Prepare upstream copy & update.
                lim_sum = Lim.extend(Lim.sum(succ_br_succ_ft.successors[succ_br_succ_oc_succ_i].lim, succ_br_succ_oc.successors[succ_br_succ_oc_succ_i].lim))
                new_succ_br_successors = [None, None]
                new_succ_br_successors[succ_br_succ_ft_i] = succ_br.successors[succ_br_succ_ft_i].copy()
                new_succ_br_successors[succ_br_succ_ft_i ^ 1] = LimBDDSignal(succ_br_succ_ft, lim=Lim.sum(lim_sum, succ_br.successors[succ_br_succ_ft_i ^ 1].lim))
                new_succ_br = LimBDDNode(limbdd.get_node_idx(), succ_br.level, new_succ_br_successors[0], new_succ_br_successors[1], var_index=succ_br.var_index)
                new_succ_br_signal = limbdd.insert_node_by_topo(new_succ_br)
                new_succ_br_signal.lim = Lim.sum(new_succ_br_signal.lim, node.successors[succ_br_i].lim)

                new_succ_ft_successors = [None, None]
                new_succ_ft_successors[0] = LimBDDSignal(succ_br_succ_ft, lim=succ_ft.low.lim)
                new_succ_ft_successors[1] = LimBDDSignal(succ_br_succ_ft, lim=succ_ft.high.lim)
                new_succ_ft = LimBDDNode(limbdd.get_node_idx(), succ_ft.level, new_succ_ft_successors[0], new_succ_ft_successors[1], var_index=succ_ft.var_index)
                new_succ_ft_signal = limbdd.insert_node_by_topo(new_succ_ft)
                new_succ_ft_signal.lim = Lim.sum(new_succ_ft_signal.lim, node.successors[succ_br_i ^ 1].lim)
                new_succ_ft_signal.lim = Lim.sum(new_succ_ft_signal.lim, Lim.extend(lim_sum))

                new_successors = [None, None]
                new_successors[succ_br_i] = new_succ_br_signal
                new_successors[succ_br_i ^ 1] = new_succ_ft_signal

                split_info_signals = [connect_signal_oc_level2, connect_signal_ft_level2]
            else:
                # Construct split_successor.
                split_successor = succ_br_succ_ft.successors[split_i]

                # Copy downstream to split_limbdd.
                limbdd.clear_visited()
                split_signal = SplitPattern.copy_nodes_between(split_successor, [limbdd.terminator], split_limbdd, terminator_constructor_bottom, copy_map=copy_map)

                # Connect downstream to upstream.
                connect_successors_level1 = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                connect_successors_level1[split_i] = split_signal
                connect_node_level1 = LimBDDNode(split_limbdd.get_node_idx(), succ_br_succ_ft.level, connect_successors_level1[0], connect_successors_level1[1], var_index=succ_br_succ_ft.var_index)

                connect_signal_level1 = split_limbdd.insert_node_by_topo(connect_node_level1)
                connect_signal_level1.lim = Lim.sum(connect_signal_level1.lim, succ_br.successors[succ_br_succ_ft_i].lim)

                connect_successors_level2 = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                connect_successors_level2[succ_br_succ_ft_i] = connect_signal_level1
                connect_node_level2 = LimBDDNode(split_limbdd.get_node_idx(), succ_br.level, connect_successors_level2[0], connect_successors_level2[1], var_index=succ_br.var_index)

                connect_signal_level2 = split_limbdd.insert_node_by_topo(connect_node_level2)
                connect_signal_level2.lim = Lim.sum(connect_signal_level2.lim, node.successors[succ_br_i].lim)

                # Prepare upstream copy & update.
                new_succ_br_succ_ft_successors = [LimBDDSignal(split_limbdd.terminator, weight=0), LimBDDSignal(split_limbdd.terminator, weight=0)]
                new_succ_br_succ_ft_successors[split_i ^ 1] = succ_br_succ_ft.successors[split_i ^ 1].copy()
                new_succ_br_succ_ft = LimBDDNode(limbdd.get_node_idx(), succ_br_succ_ft.level, new_succ_br_succ_ft_successors[0], new_succ_br_succ_ft_successors[1], var_index=succ_br_succ_ft.var_index)

                new_succ_br_successors = [None, None]
                new_succ_br_successors[succ_br_succ_ft_i ^ 1] = succ_br.successors[succ_br_succ_ft_i ^ 1].copy()
                new_succ_br_successors[succ_br_succ_ft_i] = limbdd.insert_node_by_topo(new_succ_br_succ_ft)
                new_succ_br_successors[succ_br_succ_ft_i].lim = Lim.sum(new_succ_br_successors[succ_br_succ_ft_i].lim, succ_br.successors[succ_br_succ_ft_i].lim)
                new_succ_br = LimBDDNode(limbdd.get_node_idx(), succ_br.level, new_succ_br_successors[0], new_succ_br_successors[1], var_index=succ_br.var_index)

                new_successors = [None, None]
                new_successors[succ_br_i ^ 1] = node.successors[succ_br_i ^ 1].copy()
                new_successors[succ_br_i] = limbdd.insert_node_by_topo(new_succ_br)
                new_successors[succ_br_i].lim = Lim.sum(new_successors[succ_br_i].lim, node.successors[succ_br_i].lim)

                split_info_signals = [connect_signal_level2, None]
                
            split_targets.append(node)
            split_info.append((new_successors, bit_count, split_info_signals))

        # Define upstream terminator constructor.
        def terminator_constructor_upside(node, successor):
            assert(isinstance(node, LimBDDNode) and isinstance(successor, LimBDDSignal))

            pos = split_targets.index(node)
            _, bit_count, split_signals = split_info[pos]
            if successor.node.is_branch_node: return split_signals[0].copy()
            elif bit_count: return split_signals[1].copy()
            else: return LimBDDSignal(split_limbdd.terminator, weight=0)
            
        # Copy upstream to split_limbdd.
        limbdd.clear_visited()
        split_limbdd.root = SplitPattern.copy_nodes_between(limbdd.root, split_targets, split_limbdd, terminator_constructor_upside, copy_map=copy_map)

        return split_limbdd, split_targets, split_info