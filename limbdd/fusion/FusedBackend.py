# -*- coding: utf-8 -*-
"""
Phase 4：CompleteBranch 融合发射后端（no-ancilla）。

流程：子图 → 压缩抽取 → 融合池 → 逐段 to_circuit 拼接（含片根三明治）。
工程规范见 CompleteBranch_fusion_plan.md §7.9/Phase 4；代数依据见 CompleteBranch_algebra.md。
"""
from circuit.MCXCircuit import MCXCircuit
from limbdd.CompleteBranch import CompleteBranch, extract_complete_branches, abs_lim, _bits
from limbdd.fusion.BranchFusion import fuse_pool


def limbdd2circ_fused(limbdd, qubit_mgr, n, fuse=True):
    """
    把单个子图（已完成图阶段：perform_split/collect_branches）发射为融合后电路。

    片函数 = X(root_lim) → [融合池各段] → X(root_lim)；
    各段经 CompleteBranch.to_circuit 自带对合三明治，池 XOR 不变量保证整片函数不变。
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
