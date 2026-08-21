from qiskit import QuantumRegister, QuantumCircuit
from circuit.Control import CtrlSequence
from circuit.MCXCircuit import MCXCircuit

def to_qiskit(mcx_circ):
    assert(isinstance(mcx_circ, MCXCircuit))

    qiskit_circ = QuantumCircuit()

    qregs = {}
    for var_index, qubit in mcx_circ.qubit_mgr.ins.items():
        qreg = QuantumRegister(1, f'x{var_index}')
        qregs[str(qubit)] = qreg
    for var_index, qubit in mcx_circ.qubit_mgr.anc.items():
        qreg = QuantumRegister(1, f'a{var_index}')
        qregs[str(qubit)] = qreg
    qregs[str(mcx_circ.qubit_mgr.target)] = QuantumRegister(1, 't')

    for qreg in qregs.values():
        qiskit_circ.add_register(qreg)

    for stg in mcx_circ.gate_sequence:
        if len(stg.ctrl_seq.term_list):
            for term in stg.ctrl_seq.term_list:
                if isinstance(term, CtrlSequence.ControlTerm):
                    ctrl_list = stg.ctrl_seq.cterm2clist(term)
                    control_qubits = []
                    ctrl_state = 0
                    pivot = 1
                    for control in ctrl_list:
                        control_qubits.append(qregs[str(control.qubit)][0])
                        ctrl_state += pivot * control.polar
                        pivot <<= 1
                    target_qubit = qregs[str(stg.target)][0]

                    if len(control_qubits):
                        qiskit_circ.mcx(control_qubits=control_qubits, target_qubit=target_qubit, ctrl_state=ctrl_state)
                    else:
                        qiskit_circ.x(target_qubit)
                elif isinstance(term, CtrlSequence.LimTerm):
                    qubit_list, control = stg.ctrl_seq.lterm2qlist(term)
                    if control:
                        for qubit in qubit_list:
                            qiskit_circ.mcx(control_qubits=[qregs[str(control.qubit)][0]], target_qubit=qregs[str(qubit)][0], ctrl_state=control.polar)
                    else:
                        for qubit in qubit_list:
                            qiskit_circ.x(qregs[str(qubit)][0])
        else:
            qiskit_circ.x(qregs[str(stg.target)][0])

    return qiskit_circ