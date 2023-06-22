import lattice_qft.core.basic_circuits as basic_circuits


class TrotterizedEvolution():
    """
    Class that implements quantum circuit for Suzuki-Trotter expanded time evolution of a Hamiltonian.
    """
    def __init__(self, hamiltonian, order, time, steps, correct_phase=False, phase_offset=0):
        """
        Initializes parameters.
        :param hamiltonian: (TimeEvolutionOfHamiltonian) Hamiltonian object to be trotterized
        :param order: (int) Order of Suzuki-Trotter expansion.
        :param time: (Float) Length of time to evolve over.
        :param steps: (Int) Number of steps to evolve over.
        :param correct_phase: (Bool) True if phase error from Qiskit Rz implementation will be corrected.
        """
        self.hamiltonian = hamiltonian
        self.order = order
        self.time = time
        self.steps = steps
        self.correct_phase = correct_phase
        self.phase_offset = phase_offset

    def get_hamiltonian(self):
        return self.hamiltonian

    def set_hamiltonian(self, hamiltonian):
        self.hamiltonian = hamiltonian

    def get_degree(self):
        return self.order

    def set_degree(self, order):
        self.order = order

    def get_time(self):
        return self.time

    def set_time(self, time):
        self.time = time

    def get_steps(self):
        return self.steps

    def set_steps(self, steps):
        self.steps = steps

    def get_correct_phase(self):
        return self.correct_phase

    def set_correct_phase(self, correct_phase):
        self.correct_phase = correct_phase

    def get_phase_offset(self):
        return self.phase_offset

    def set_phase_offset(self, offset):
        self.phase_offset = offset

    def set_hamiltonian_params(self, hamiltonian):
        hamiltonian.set_time(self.time/self.steps)
        hamiltonian.set_correct_phase = self.correct_phase

    def unfold_expansion(self):
        order = self.order
        num_ops = self.hamiltonian.get_num_terms()
        coeffs = []
        if order ==  1:
            for n in range(num_ops):
                coeffs.append(tuple([n, 1]))
        if order == 2:
            for n in range(2*(num_ops) - 1):
                if n == (num_ops - 1):
                    coeffs.append(tuple([n, 1]))
                else:
                    if n < (num_ops-1):
                        coeffs.append(tuple([n , 0.5]))
                    else:
                        coeffs.append(tuple([num_ops-2-(n%num_ops), 0.5]))
        return coeffs

    def build(self, qc, q, q_ancillas=None, params=None):
        """
        Builds Quantum Circuit for Trotteized time evolution.
        :param qc: (QuantumCircuit) Quantum Circuit to build on.
        :param q: (QuantumRegister) Quantum Register in qc to build on.
        :param q_ancillas: (QuantumRegister) Contains a single ancilla qubit.
        :param params: None.
        :return: None, operates on quantum circuit in-place.
        """
        rz_phase_error = 0
        self.set_hamiltonian_params(self.hamiltonian)
        expansion = self.unfold_expansion()
        for n in range(self.steps):
            times = [n*(self.time/self.steps)]*len(expansion)
            activation_func_vals = [None]*len(expansion)
            tmp_params = [expansion, activation_func_vals]
            self.hamiltonian.build(qc, q, q_ancillas, tmp_params)
            if self.correct_phase:
                rz_phase_error += self.hamiltonian.get_total_phase_offset(q.size, expansion, activation_func_vals)
        if self.correct_phase:
            qc.compose(basic_circuits.phase(q, - (rz_phase_error - (self.phase_offset * self.time))), inplace=True)


