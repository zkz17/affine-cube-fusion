from limbdd.Lim import Lim
from limbdd.Node import LimBDDNode, LimBDDSignal
from limbdd.LimBDD import LimBDD
from abc import ABC, abstractmethod

class SplitPattern(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def identify_candidates(self, limbdd, level):
        pass

    @abstractmethod
    def process_split(self, limbdd, split_limbdd, candidates):
        pass

    # Return a LimBDDSignal from the top of the copied part or None (empty).
    @staticmethod
    def copy_nodes_between(source, targets, split_limbdd, termination_constructor, level_drop=0, copy_map=None):
        assert(isinstance(source, LimBDDSignal) and isinstance(targets, list) and isinstance(split_limbdd, LimBDD))

        # copy_map 是本次拷贝的记忆表(会被写入),必须每次新建:共享默认字典会把上一次
        # 图重写的 node.idx → 拷贝信号带进本次调用,导致跨图错误复用结点。
        if copy_map is None: copy_map = {}

        if not len(targets): return
        relative_level = source.level - level_drop

        # Use cache.
        if source.node.idx in copy_map:
            if not copy_map[source.node.idx]: return None
            if not relative_level: return LimBDDSignal(split_limbdd.terminator, weight=source.weight)

            signal = copy_map[source.node.idx].copy()
            if signal.weight: signal.lim = Lim.sum(signal.lim, Lim.truncate(source.lim, relative_level))
            return signal

        if source.level == targets[0].level:
            copy_map[source.node.idx] = None
            if source.node in targets:

                if relative_level: # source.node is not terminator
                    new_successors = []
                    for successor in source.node.successors:
                        new_successors.append(termination_constructor(source.node, successor))

                    new_node = LimBDDNode(split_limbdd.get_node_idx(), relative_level, new_successors[0], new_successors[1], var_index=source.node.var_index)
                    
                    signal = split_limbdd.insert_node_by_topo(new_node)
                    copy_map[source.node.idx] = signal.copy()
                    
                    if signal.weight: signal.lim = Lim.sum(signal.lim, Lim.truncate(source.lim, relative_level))
                    return signal
                else: return LimBDDSignal(split_limbdd.terminator, weight=source.weight)

            return None
        elif source.level > targets[0].level:
            copy_map[source.node.idx] = None

            low, high = None, None
            if source.node.low.weight : low  = SplitPattern.copy_nodes_between(source.node.low,  targets, split_limbdd, termination_constructor, level_drop, copy_map=copy_map)
            if source.node.high.weight: high = SplitPattern.copy_nodes_between(source.node.high, targets, split_limbdd, termination_constructor, level_drop, copy_map=copy_map)

            if low or high:
                new_node = LimBDDNode(split_limbdd.get_node_idx(), relative_level, \
                                    low  if low  else LimBDDSignal(split_limbdd.terminator, weight=0), \
                                    high if high else LimBDDSignal(split_limbdd.terminator, weight=0), \
                                    var_index=source.node.var_index)
                
                signal = split_limbdd.insert_node_by_topo(new_node)
                copy_map[source.node.idx] = signal.copy()
                
                if signal.weight: signal.lim = Lim.sum(signal.lim, Lim.truncate(source.lim, relative_level))
                return signal
            
            return None
        else: return None
