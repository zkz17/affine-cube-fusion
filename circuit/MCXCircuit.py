import numpy as np

from utils.Qubit import Qubit, QubitManager
from utils.TruthTable import TruthTable
from circuit.Control import CtrlSequence

class STGate:
    __slots__ = ('target', 'ctrl_seq')

    def __init__(self, target, control_list=None):
        self.target = target # Qubit
        self.ctrl_seq = CtrlSequence()
        self.ctrl_seq.append_control_list(control_list if control_list is not None else [])

    def append_ctrl_sequence(self, ctrl_seq):
        assert(isinstance(ctrl_seq, CtrlSequence))

        self.ctrl_seq.append_ctrl_sequence(ctrl_seq)

    def copy(self):
        stg = STGate(self.target)
        stg.ctrl_seq = self.ctrl_seq.copy()
        return stg

class MCXCircuit:
    def __init__(self, qubit_mgr = None, target = None):
        self.qubit_mgr = qubit_mgr    # QubitManager
        self.target = target          # Qubit
        self.gate_sequence = []       # [ STGate ]
        self.tt = 0

        if not self.qubit_mgr: self.qubit_mgr = QubitManager()
        if not self.target: self.target = self.qubit_mgr.get_target_qubit()

    # Merge another MCXCircuit controlled by a list of Control.
    # Note: The corresponding Qubit in 'other' and 'controls' are assumed to be already in 'self.qubit_mgr'.
    def merge_controlled_circuit(self, other, controls = None, target = None):
        assert(isinstance(other, MCXCircuit))

        self.merge(other.controlled_circuit(controls if controls is not None else []))

    # Merge the gate_sequence of another MCXCircuit directly.
    # Note: 'self.qubit_mgr' is not updated.
    def merge(self, other, target=None):
        assert(isinstance(other, MCXCircuit))

        for stg in other.gate_sequence:
            stg_copy = stg.copy()
            if target and stg_copy.target == other.target: stg_copy.target = target
            self.add_gate_as_stg(stg_copy)

    # Return a controlled version of self.
    # Note: The corresponding Qubit of 'control_list' is assumed to be disjoint to existing STGate.
    def controlled_circuit(self, control_list):
        if not len(control_list): return self
        circ = MCXCircuit(target=self.target)
        circ.qubit_mgr.absorb(self.qubit_mgr)
        for stg in self.gate_sequence:
            stg_copy = stg.copy()
            if stg_copy.target == self.target: stg_copy.ctrl_seq.link_control_list(control_list)
            circ.add_gate_as_stg(stg_copy)
        return circ

    def is_circuit_target(self, qubit):
        return isinstance(qubit, Qubit) and qubit == self.target

    def get_input(self, index):
        return self.qubit_mgr.get_input_qubit(index)

    def get_ancilla(self, index):
        return self.qubit_mgr.get_ancilla_qubit(index)
    
    def add_lim_operator(self, qubit_list, control=None):
        if not len(qubit_list): return
        if self.empty():
            stg = STGate(self.target)
            stg.ctrl_seq.term_list.pop()
            stg.ctrl_seq.append_lim_operator(qubit_list, control)
            self.add_gate_as_stg(stg)
        else:
            self.gate_sequence[-1].ctrl_seq.append_lim_operator(qubit_list, control)

    def add_gate_as_stg(self, stg):
        assert(isinstance(stg, STGate))

        if not self.empty() and self.gate_sequence[-1].target == stg.target:
            self.gate_sequence[-1].append_ctrl_sequence(stg.ctrl_seq)
        else:
            self.gate_sequence.append(stg)

    def add_gate(self, controls, target):
        assert(isinstance(controls, list) and isinstance(target, Qubit))

        self.add_gate_as_stg(STGate(target, controls))

    def copy(self):
        result = MCXCircuit(target=self.target)
        result.qubit_mgr.absorb(self.qubit_mgr)
        result.gate_sequence = self.gate_sequence.copy()
        return result
    
    def empty(self):
        return not len(self.gate_sequence)
    
    def verify(self, tt):
        assert(isinstance(tt, TruthTable))

        self.compute_tt()
        return tt.is_equal_to(self.tt)
    
    def compute_tt(self):
        self.tt = 0
        ins_list, anc_list = [], []
        for var_index in sorted(self.qubit_mgr.ins.keys()):
            ins_list.append(self.get_input(var_index))
        for anc_index in sorted(self.qubit_mgr.anc.keys()):
            anc_list.append(self.get_ancilla(anc_index))

        n_ins = len(ins_list)
        n_states = 1 << n_ins
        states = np.arange(n_states, dtype=np.int64)
        anc_states = np.zeros(n_states, dtype=np.int64)
        res = np.zeros(n_states, dtype=np.bool_)

        # Canonical bit position of each qubit: input qubits first, then ancilla.
        canon_pos = {}
        for pos, qubit in enumerate(ins_list): canon_pos[qubit] = pos
        for pos, qubit in enumerate(anc_list): canon_pos[qubit] = n_ins + pos
        ins_mask = (1 << n_ins) - 1

        for stg in self.gate_sequence:
            seq = stg.ctrl_seq
            qubits = list(seq.qubit_map.keys())
            cpos = np.array([canon_pos.get(qubit, -1) for qubit in qubits], dtype=np.int64)
            mpos = np.array([seq.qubit_map[qubit] for qubit in qubits], dtype=np.int64)

            # Map the canonical state into this CtrlSequence's qubit order.
            mapped = np.zeros(n_states, dtype=np.int64)
            sa = states | (anc_states << n_ins)
            for i in range(len(qubits)):
                if cpos[i] >= 0:
                    mapped |= ((sa >> cpos[i]) & 1) << mpos[i]

            # Apply all terms of this gate on all states simultaneously.
            flip = np.zeros(n_states, dtype=np.bool_)
            for term in seq.term_list:
                if isinstance(term, CtrlSequence.ControlTerm):
                    flip ^= ((mapped & term.mask) == term.bits)
                else: # LimTerm
                    if term.control_pos == -1:
                        mapped ^= term.bits
                    else:
                        mapped ^= np.where(((mapped >> term.control_pos) & 1) == term.control_polar, term.bits, 0)

            # Unmap: restore canonical bits for qubits present in this sequence.
            for i in range(len(qubits)):
                if cpos[i] >= 0:
                    sa = (sa & ~(np.int64(1) << cpos[i])) | (((mapped >> mpos[i]) & 1) << cpos[i])
            states = sa & ins_mask
            anc_states = sa >> n_ins

            if stg.target.is_target(): res ^= flip
            elif stg.target.is_ancilla():
                pos = canon_pos[stg.target] - n_ins
                anc_states ^= np.where(flip, 1 << pos, 0)
            else: raise Exception(f'STGate acting on input qubit {str(stg.target)}!!!')

        bad = np.flatnonzero((states != np.arange(n_states)) | (anc_states != 0))
        if len(bad):
            bad_state = int(bad[0])
            raise Exception(f'Anomaly Found!!!\n  input_state : {bad_state}\n  output_state: {int(states[bad_state])}\n  anc_state  : {int(anc_states[bad_state])}')

        self.tt = int.from_bytes(np.packbits(res, bitorder='little').tobytes(), 'little')
        return self.tt

    def clear(self, restart=False):
        self.gate_sequence.clear()
        if restart:
            self.tt = 0
            self.qubit_mgr = QubitManager()
            self.target = self.qubit_mgr.get_target_qubit()

    def print(self):
        for i, stg in enumerate(self.gate_sequence):
            print(f'Gate {i}')
            CtrlSequence.print(stg.ctrl_seq)

    def simplify_terms(self):
        """
        相邻可合并 LimTerm 合并（Phase 4 的"相邻段包裹门抵消"pass，保语义）：
          - 两个无条件 X 组 → 位掩码异或；
          - 同控制位、同极性的两个条件翻转（且翻转位不含该控制位）→ 位掩码异或；
          - 同控制位、异极性、翻转位相同的两个条件翻转 → 无条件 X 组
            （[c=0→b][c=1→b] = X(b)，即 §7.8 公共部分上提在门级的复现）；
          - 合并后位掩码为 0 的项丢弃。
        注意：不合并 ControlTerm（多控 X 条件不可线性合并）。
        """
        for stg in self.gate_sequence:
            merged = []
            for term in stg.ctrl_seq.term_list:
                if isinstance(term, CtrlSequence.LimTerm):
                    if term.bits == 0:
                        continue
                    if merged and isinstance(merged[-1], CtrlSequence.LimTerm):
                        prev = merged[-1]
                        if prev.control_pos == term.control_pos and prev.control_polar == term.control_polar:
                            if prev.control_pos == -1 or not (((prev.bits | term.bits) >> prev.control_pos) & 1):
                                prev.bits ^= term.bits
                                if prev.bits == 0:
                                    merged.pop()
                                continue
                        elif (prev.control_pos != -1 and prev.control_pos == term.control_pos
                              and prev.control_polar != term.control_polar and prev.bits == term.bits):
                            prev.control_pos = -1
                            prev.control_polar = -1
                            continue
                merged.append(term.copy())
            stg.ctrl_seq.term_list = merged