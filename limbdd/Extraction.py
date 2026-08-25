# -*- coding: utf-8 -*-
"""
BDD piece -> CompleteBranch extraction helpers (limbdd-side bridge).

The CompleteBranch class itself and the closed-form fusion backend live in
`backend/`; this module keeps only the pieces that walk LimBDD graphs.
"""
from backend.CompleteBranch import CompleteBranch, _bits
from limbdd.LimBDD import LimBDD, SimpleBranch

def abs_lim(lim, var_order, anomalies=None):
    """Convert a graph-local Lim (level space) into absolute variable bit space: bit k = variable k."""
    m = 0
    for j in range(lim.bitnum):
        if (lim.bitstr >> j) & 1:
            if j >= len(var_order):
                if anomalies is not None:
                    anomalies['lim_overflow'] += 1
                continue
            m |= 1 << var_order[j]
    return m




def extract_complete_branches(limbdd, n, origin=0, anomalies=None):
    """
    Compressed extraction mirroring the branch2circ walk level by level (see docs/architecture.md).

    For each "root -> terminal" path, generate one CompleteBranch:
      - taken-edge positions (including one-child and branching-node taken edges) -> mask set + bitstr records polarity,
        and the taken edge's Lim is XORed into x_global;
      - fallthrough positions -> the two outgoing Lim values are normalized (common part hoisted to x_global), then appended in order as ft_lims;
      - a w0=w1=0 node should not appear (recorded as anomaly); a neg edge should not appear (recorded as anomaly);
      - an ft target must lie below its controlling variable (recorded as anomaly; tests catch it).
    """
    assert isinstance(limbdd, LimBDD)
    assert n >= 0
    assert not limbdd.root.neg, 'root edge should not be neg after split_neg'

    top_down = list(reversed(limbdd.var_order))
    pos_of = {v: i for i, v in enumerate(top_down)}
    terminator = limbdd.terminator
    results = []

    def walk(node, mask, bitstr, x_global, ft_lims, bits, bitnum):
        if node is terminator:
            results.append(CompleteBranch(n, mask, bitstr, x_global, ft_lims,
                                          source_branch=SimpleBranch(bits, bitnum),
                                          origin=origin))
            return
        if node.level == 0:
            if anomalies is not None:
                anomalies['zero_level'] += 1
            return
        w0, w1 = node.low.weight, node.high.weight
        if w0 == 0 and w1 == 0:
            if anomalies is not None:
                anomalies['zero_node'] += 1
            return
        if node.low.neg or node.high.neg:
            if anomalies is not None:
                anomalies['neg_edge'] += 1

        var = node.var_index
        if node.low.node is node.high.node and w0 == 1 and w1 == 1:
            # fallthrough: normalize the two outgoing Lim values (hoist the common part), append terms top-down.
            m0 = abs_lim(node.low.lim, limbdd.var_order, anomalies)
            m1 = abs_lim(node.high.lim, limbdd.var_order, anomalies)
            common = m0 & m1
            x_global ^= common
            m0 ^= common
            m1 ^= common
            if m0 or m1:
                for t in _bits(m0 | m1):
                    if pos_of.get(t, -1) <= pos_of[var]:
                        if anomalies is not None:
                            anomalies['ft_target_above'] += 1
                ft_lims = ft_lims + ((var, m0, m1),)
            walk(node.low.node, mask, bitstr, x_global, ft_lims, bits, bitnum)
        else:
            for b, succ in ((0, node.low), (1, node.high)):
                if succ.weight:
                    walk(succ.node,
                         mask | (1 << var),
                         bitstr | (b << var),
                         x_global ^ abs_lim(succ.lim, limbdd.var_order, anomalies),
                         ft_lims,
                         bits | (b << bitnum), bitnum + 1)

    walk(limbdd.root.node, 0, 0, 0, (), 0, 0)
    return results
