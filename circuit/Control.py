from utils.Qubit import Qubit

class Control:
    __slots__ = ('qubit', 'polar')

    def __init__(self, qubit, polar=1):
        self.qubit = qubit
        self.polar = polar

    def __str__(self):
        return f'Control(Qubit: {str(self.qubit)}, polar: {self.polar})'

class CtrlSequence:
    __slots__ = ('term_list', 'qubit_map', 'num_vars')

    class ControlTerm:
        __slots__ = ('mask', 'bits')

        #   [ Control ]   : new  c_n-1 ... c2    ¬c0 old
        # ControlTerm.bits: high 1     ... 1  0  0   low
        # ControlTerm.mask: high 1     ... 1  0  1   low
        def __init__(self, mask, bits):
            self.mask = mask
            self.bits = bits

        # Change the current order of controls ( map ) to a new order ( new_map ).
        # Note: 'map' is assumed to be a subset of 'new_map'.
        def transform(self, map, new_map):
            new_mask, new_bits = 0, 0
            for qubit, pos in map.items():
                pivot = (1 << pos)
                if pivot & self.mask:
                    assert(qubit in new_map)

                    new_pivot = (1 << new_map[qubit])
                    new_mask += new_pivot
                    if pivot & self.bits: new_bits += new_pivot
            self.mask = new_mask
            self.bits = new_bits

        def lshift(self, operand=1):
            self.mask <<= operand
            self.bits <<= operand

        def rshift(self, operand=1):
            self.mask >>= operand
            self.bits >>= operand

        def link_bits(self, mask, bits):
            assert((self.mask & mask) == 0)
            self.mask += mask
            self.bits += (mask & bits)

        # self = self & other_term
        def link(self, other_term, num_vars):
            assert(isinstance(other_term, CtrlSequence.ControlTerm))
            other_term.lshift(num_vars)
            self.link_bits(other_term.mask, other_term.bits)
            other_term.rshift(num_vars)

        def copy(self):
            return CtrlSequence.ControlTerm(self.mask, self.bits)
        
        def perform_on_state(self, flip, state):
            if (state & self.mask) == self.bits: flip ^= 1
            return flip, state
        
    class LimTerm:
        __slots__ = ('bits', 'control_pos', 'control_polar')

        #     Lim     : high X_n-1 ... X2 I  X0 low
        # LimTerm.bits: high 1     ... 1  0  1  low
        def __init__(self, bits, control_pos=-1, control_polar=-1):
            self.bits = bits
            self.control_pos   = control_pos
            self.control_polar = control_polar

        # Change the current order of lim ( map ) to a new order ( new_map ).
        # Note: 'map' is assumed to be a subset of 'new_map'.
        def transform(self, map, new_map):
            new_bits = 0
            for qubit, pos in map.items():
                pivot = (1 << pos)
                if pivot & self.bits:
                    assert(qubit in new_map)

                    new_pivot = (1 << new_map[qubit])
                    new_bits += new_pivot
            self.bits = new_bits

            if self.control_pos != -1:
                for qubit, pos in map.items():
                    if pos == self.control_pos:
                        self.control_pos = new_map[qubit]
                        break

        def lshift(self, operand=1):
            self.bits <<= operand
            self.control_pos += operand

        def rshift(self, operand=1):
            self.bits >>= operand
            self.control_pos -= operand

        def link_bits(self, mask, bits):
            pass

        def link(self, other_term, num_vars):
            pass

        def copy(self):
            return CtrlSequence.LimTerm(self.bits, self.control_pos, self.control_polar)
        
        def perform_on_state(self, flip, state):
            if self.control_pos == -1 or ((state >> self.control_pos) & 1) == self.control_polar: state ^= self.bits
            return flip, state

    def __init__(self, qubit_list=None):
        self.term_list = []             # [ ControlTerm ]
        self.qubit_map = {}             # { Qubit : int }
        self.num_vars = 0               # int

        self.append_qubits(qubit_list if qubit_list is not None else [])
    # Translate a list of Control to a ControlTerm.
    def clist2cterm(self, ctrl_list):
        mask, bits = 0, 0
        for ctrl in ctrl_list:
            assert(isinstance(ctrl, Control))

            if not ctrl.qubit in self.qubit_map: self.append_qubits([ctrl.qubit])

            pivot = (1 << self.qubit_map[ctrl.qubit])
            mask += pivot
            bits += pivot * ctrl.polar
        return self.ControlTerm(mask, bits)
    
    # Translate a ControlTerm to a list of Control.
    def cterm2clist(self, ctrl_term):
        assert(isinstance(ctrl_term, self.ControlTerm))
        
        ctrl_list = []
        for qubit, pos in self.qubit_map.items():
            pivot = (1 << pos)
            if not ctrl_term.mask & pivot: continue
            ctrl_list.append(Control(qubit, polar=int((pivot & ctrl_term.bits) != 0)))
        return ctrl_list
    
    # Translate a list of Qubit and a Control to a LimTerm.
    def qlist2lterm(self, qubit_list, control):
        assert(not control or isinstance(control, Control))

        bits = 0
        self.append_qubits(qubit_list)
        for qubit in qubit_list:
            assert(isinstance(qubit, Qubit))

            pivot = (1 << self.qubit_map[qubit])
            bits += pivot

        control_pos, control_polar = -1, -1
        if control:
            if not control.qubit in self.qubit_map: self.append_qubits([control.qubit])
            control_pos, control_polar = self.qubit_map[control.qubit], control.polar
        return self.LimTerm(bits, control_pos, control_polar)
    
    # Translate a LimTerm to a list of Qubit and a Control.
    def lterm2qlist(self, lim_term):
        assert(isinstance(lim_term, self.LimTerm))

        qubit_list = []
        for qubit, pos in self.qubit_map.items():
            pivot = (1 << pos)
            if lim_term.bits & pivot: qubit_list.append(qubit)

        control = None
        for qubit, pos in self.qubit_map.items():
            if not pos == lim_term.control_pos: continue
            control = Control(qubit, lim_term.control_polar)

        return qubit_list, control

    # Add a list of Qubit to the current 'self.qubit_map'.
    # Note: Existeing Qubit are ignored.
    def append_qubits(self, qubit_list):
        for qubit in qubit_list:
            assert(isinstance(qubit, Qubit))

            if qubit not in self.qubit_map:
                self.qubit_map[qubit] = self.num_vars
                self.num_vars += 1

    # Append a ControlTerm in the form of [ Control ].
    def append_control_list(self, ctrl_list):
        assert(isinstance(ctrl_list, list))

        self.term_list.append(self.clist2cterm(ctrl_list))

    # Append a LimTerm in the form of [ Qubit ] and a Control.
    def append_lim_operator(self, qubit_list, control):
        self.term_list.append(self.qlist2lterm(qubit_list, control))

    # Append a ControlTerm or a LimTerm to the current term_list.
    def append_new_term(self, term, qubit_map=None):
        assert(isinstance(term, CtrlSequence.ControlTerm) or isinstance(term, CtrlSequence.LimTerm))

        if qubit_map:
            new_term = term.copy()
            new_term.transform(qubit_map, self.qubit_map)
            self.term_list.append(new_term)
        else:
            self.term_list.append(term.copy())

    # Append another CtrlSequence.
    def append_ctrl_sequence(self, other):
        assert(isinstance(other, CtrlSequence))

        self.append_qubits(other.qubit_map.keys())
        for term in other.term_list:
            self.append_new_term(term, other.qubit_map)

    # Link a list of Control to all existing terms in 'term_list'.
    # Note: The corresponding Qubit in 'ctrl_list' are assumed to be disjoint to each of the existing ControlTerm.
    def link_control_list(self, ctrl_list):
        assert(isinstance(ctrl_list, list))

        mask_o, bits_o = 0, 0
        mask_i, bits_i = 0, 0
        qubit_list = []
        pivot = 1
        for control in ctrl_list:
            assert(isinstance(control, Control))

            if control.qubit in self.qubit_map:
                pos = self.qubit_map[control.qubit]
                mask_i += (1 << pos)
                bits_i += (control.polar << pos)
            else:
                qubit_list.append(control.qubit)
                mask_o += pivot
                bits_o += pivot * control.polar
                pivot <<= 1

        for term in self.term_list:
            term.link_bits(mask_i, bits_i)
        for term in self.term_list:
            term.link(self.ControlTerm(mask_o, bits_o), self.num_vars)
        self.append_qubits([control.qubit])

    # Link a disjoint CtrlSequence.
    # TODO: Accomodate LimTerm.
    def link_ctrl_sequence(self, other):
        assert(isinstance(other, CtrlSequence))

        if not CtrlSequence.is_disjoint(self, other): raise Exception('Qubit range overlap when joining CtrlSequence. ')
        old_num_vars = self.num_vars
        self.append_qubits(other.qubit_map.keys())

        new_term_list = []
        for term in self.term_list:
            for other_term in other.term_list:
                new_term = term.copy()
                new_term.link(other_term, old_num_vars)
                new_term_list.append(new_term)
        self.term_list = new_term_list

    def copy(self):
        seq = CtrlSequence()
        seq.qubit_map = self.qubit_map.copy()
        seq.num_vars = self.num_vars

        for term in self.term_list:
            seq.term_list.append(term.copy())
        return seq
    
    # 'ins_list' and 'anc_list' combined is assumed to cover all qubits in 'self.qubit_map'.
    def check_state(self, ins_state, anc_state, ins_list, anc_list):
        '''
        print('******* ****** ******')
        CtrlSequence.print(self)
        print(' ------  ---  ------ ')
        print(f'ins_list:  {[str(qubit) for qubit in ins_list]}')
        print(f'anc_list:  {[str(qubit) for qubit in anc_list]}')
        print(f'qubit_map: {[(str(key), value) for key, value in self.qubit_map.items()]}')
        print(' ------  ---  ------ ')
        print(f'    input state : {bin(ins_state)[2:].zfill(len(ins_list) + len(anc_list))}')
        '''
        state = 0

        # Transform 'ins_state' to adapt to current order.
        for pos in range(len(ins_list)):
            if not ins_list[pos] in self.qubit_map: continue

            pivot = (1 << pos)
            if ins_state & pivot:
                ins_state -= pivot
                state += (1 << self.qubit_map[ins_list[pos]])

        # Transform 'anc_state' to adapt to current order.
        for pos in range(len(anc_list)):
            if not anc_list[pos] in self.qubit_map: continue

            pivot = (1 << pos)
            if anc_state & pivot:
                anc_state -= pivot
                state += (1 << self.qubit_map[anc_list[pos]])

        #print(f'    fused state : {bin(state)[2:].zfill(len(ins_list) + len(anc_list))}')

        # Check state.
        flip = 0
        for term in self.term_list:
            flip, state = term.perform_on_state(flip, state)
            #print(f'        state = {bin(state)[2:].zfill(len(ins_list) + len(anc_list))}, flip = {flip}')

        # Transform 'ins_state' back to circuit order.
        for pos in range(len(ins_list)):
            if not ins_list[pos] in self.qubit_map: continue

            if state & (1 << self.qubit_map[ins_list[pos]]): ins_state += (1 << pos)

        # Transform 'anc_state' back to circuit order.
        for pos in range(len(anc_list)):
            if not anc_list[pos] in self.qubit_map: continue
            
            if state & (1 << self.qubit_map[anc_list[pos]]): anc_state += (1 << pos)

        #print(f'    output state: {bin(ins_state)[2:].zfill(len(ins_list) + len(anc_list))}, flip = {flip}')
        #print('******* ****** ******')
        return flip, ins_state, anc_state

    @staticmethod
    def is_disjoint(seq1, seq2):
        assert(isinstance(seq1, CtrlSequence) and isinstance(seq2, CtrlSequence))

        return seq1.qubit_map.keys().isdisjoint(seq2.qubit_map.keys())
    
    @ staticmethod
    def print_term(seq, term):
        if isinstance(term, CtrlSequence.ControlTerm):
            ctrl_list = seq.cterm2clist(term)
            print([str(ctrl) for ctrl in ctrl_list])
        elif isinstance(term, CtrlSequence.LimTerm):
            qubit_list, control = seq.lterm2qlist(term)
            print(str(control), [str(qubit) for qubit in qubit_list])
        else:
            pass

    @staticmethod
    def print(seq):
        assert(isinstance(seq, CtrlSequence))

        for term in seq.term_list:
            CtrlSequence.print_term(seq, term)