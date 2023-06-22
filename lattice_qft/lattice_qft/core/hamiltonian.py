


class TimeEvolutionOfHamiltonian():
    """
    Circuit Factory which constructs circuit simulating time evolution of a Hamiltonian (Non-Trotterized, single timestep)
    """
    def __init__(self, params=[], operators=[], correct_phase=False):
        """
        Initializes values.
        :param params: (list) list of parameters matched wtih operators.
        :param operators: (list) list of BasicOperator-derived objects which are matched with params.
        :param correct_phase: (bool) True if phase error from Qiskit Rz is to be corrected.
        """
        self.timestep_length = 0
        self.params = params
        self.operators = operators
        self.correct_phase = correct_phase

    def set_time(self, timestep_length):
        self.timestep_length = timestep_length

    def get_time(self):
        return self.time

    def set_correct_phase(self, correct_phase):
        self.correct_phase = correct_phase

    def get_correct_phase(self):
        return self.correct_phase

    def set_params(self, new_params):
        self.params = new_params

    def get_params(self):
        return self.params

    def add_term(self, params, operator):
        if isinstance(params, list):
            self.params.append(params)
        else:
            self.params.append([params])
        self.operators.append(operator)

    def add_terms(self, params, operators):
        for i in range(len(params)):
            print('params len: ',len(params))
            if isinstance(params[i], list):
                self.params.append(params[i])
            else:
                self.params.append([params[i]])
            self.operators.append(operators[i])

    def get_terms(self):
        return list(zip(self.params, self.operators))

    def get_num_terms(self):
        return len(self.operators)

    def clear_all(self):
        self.params = []
        self.operators = []

    def build(self, qc, q, q_ancillas=None, params=None):
        """
        Builds
        :param qc: (QuantumCircuit) Quantum Circuit to build on.
        :param q: (QuantumRegister) Quantum Register in qc to build on.
        :param q_ancillas: (QuantumRegister) Contains a single ancilla qubit.
        :param params: (list) contains two elements: 1) list of expansion coefficients from Trotterization, 2) values of activation function for time-dependent parameters.
        :return: None, operates on quantum circuit in-place.
        """
        expansion = params[0]
        activation_func_vals = params[1]
        #additional_regs = params[2]
        for n in range(len(expansion)):
            op_index = expansion[n][0]
            op_coeff = expansion[n][1]
            op_params = self.params[op_index]
            scaled_op_params = [self.timestep_length * op_coeff*i if not isinstance(i, TimeDependentParameter)
                                else self.timestep_length * op_coeff*i.get_val_scaled(activation_func_vals[n]) for i in op_params]
            #print(qc.draw())
            # qc.compose(self.operators[op_index].build_operator_circuit([q], q_ancillas, scaled_op_params,
            #                                                       self.correct_phase),inplace=True)
            qc.compose(self.operators[op_index].build_operator_circuit([q], q_ancillas, scaled_op_params,
                                                                  ),inplace=True)

    def get_total_phase_offset(self, Nq, expansion, activation_func_vals):
        total = 0
        for n in range(len(expansion)):
            op_index = expansion[n][0]
            op_coeff = expansion[n][1]
            op_params = self.params[op_index]
            scaled_op_params = [self.timestep_length * op_coeff * i if not isinstance(i, TimeDependentParameter)
                                else self.timestep_length * op_coeff * i.get_val_scaled(activation_func_vals[n]) for i in op_params]
            phase_offset = self.operators[op_index].phase(Nq, scaled_op_params)
            total += phase_offset
        return total


class TimeDependentParameter:
    """
    Class which represents parameters for Operators which can change value.
    """
    def __init__(self, val):
        """
        Initializes value.
        :param val: Sets value to be scaled in time dependence.
        """
        self.val = val

    def set_val(self, val):
        self.val = val

    def get_val(self):
        return self.val

    def get_val_scaled(self, scaling):
        return self.val*scaling






