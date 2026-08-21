class Qubit:
    __slots__ = ('index', 'type')

    def __init__(self, index, type = 0):
        self.index = index
        # type 0: data    qubit
        # type 1: input   qubit
        # type 2: ancilla qubit
        self.type = type
        assert(self.type in [0, 1, 2])

    def is_target(self):
        return self.type == 0
    
    def is_input(self):
        return self.type == 1
    
    def is_ancilla(self):
        return self.type == 2

    def __eq__(self, value):
        return isinstance(value, Qubit) and self.type == value.type and self.index == value.index

    def __str__(self):
        if self.type == 0: return 't'
        elif self.type == 1: return f"q{self.index}"
        elif self.type == 2: return f"a{self.index}"
    
    def __hash__(self):
        return hash((self.index, self.type))
    
    def __eq__(self, other):
        if isinstance(other, Qubit):
            return self.index == other.index and self.type == other.type
        return False

class QubitManager:
    def __init__(self):
        self.target = Qubit(0)
        self.ins = {} # { int : Qubit }
        self.anc = {} # { int : Qubit }
        self.ins_counter = 0
        self.anc_counter = 0

    def clear(self):
        self.ins = {}
        self.anc = {}
        self.ins_counter = 0
        self.anc_counter = 0

    def absorb(self, other):
        assert(isinstance(other, QubitManager))

        for var_index in other.ins.keys():
            if var_index in self.ins: continue
            self.get_input_qubit(var_index)

        for anc_index in other.anc.keys():
            if anc_index in self.anc: continue
            self.get_ancilla_qubit(anc_index)

    def get_input_qubit(self, index):
        if not index in self.ins:
            self.ins[index] = Qubit(self.ins_counter, 1)
            self.ins_counter += 1
        return self.ins[index]
    
    def get_input_num(self):
        return self.ins_counter

    def get_ancilla_qubit(self, index):
        if not index in self.anc:
            self.anc[index] = Qubit(self.anc_counter, 2)
            self.anc_counter += 1
        return self.anc[index]
    
    def get_ancilla_num(self):
        return self.anc_counter
    
    def get_target_qubit(self):
        return self.target

    def has_qubit(self, qubit):
        assert(isinstance(qubit, Qubit))

        if qubit.is_target(): return qubit == self.target
        elif qubit.is_input(): return qubit in self.ins.values()
        elif qubit.is_ancilla(): return qubit in self.anc.values()
        else: return False
    
    def __str__(self):
        return f"controls: {[str(qubit) for qubit in self.ins.values()]}\ntarget: {self.target}\nancilla: {[str(qubit) for qubit in self.anc.values()]}"