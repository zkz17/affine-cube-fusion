# -*- coding: utf-8 -*-
"""
Phase 4: CompleteBranch fusion emission backend (no-ancilla).

Pipeline: subgraph -> compressed extraction -> fusion pool -> piecewise to_circuit concatenation (with piece-root sandwich).
Engineering conventions: see docs/architecture.md; algebraic basis: see CompleteBranch_algebra.md.
"""
from circuit.MCXCircuit import MCXCircuit
from backend.CompleteBranch import CompleteBranch, _bits
from limbdd.Extraction import abs_lim, extract_complete_branches
from backend.fusion.BranchFusion import fuse_pool


def limbdd2circ_fused(limbdd, qubit_mgr, n, fuse=True):
    """
    Emit a single subgraph (already through the graph phase: perform_split/collect_branches) as a fused circuit.

    Piece function = X(root_lim) -> [fusion pool segments] -> X(root_lim);
    Each segment carries its own involutive sandwich via CompleteBranch.to_circuit, and the pool XOR invariant keeps the whole piece function unchanged.
    """
    branches = extract_complete_branches(limbdd, n)
    pool = fuse_pool(branches) if fuse else list(branches)

    root_mask = abs_lim(limbdd.root.lim, limbdd.var_order)
    root_qubits = [qubit_mgr.get_input_qubit(v) for v in _bits(root_mask)]

    circ = MCXCircuit(qubit_mgr)
    circ.add_lim_operator(root_qubits)
    for cb in pool:
        circ.merge(cb.to_circuit(qubit_mgr))
    circ.add_lim_operator(root_qubits)
    return circ
