from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise

####################################################################
def run_sv(qc):
   """
   Simulate circuit on statevector simulator backend.
   :param qc: (QuantumCircuit) Quantum circuit to simulate.
   :return: (np.array) vector representing statevector of quantum circuit.
   """
   backend_sv = Aer.get_backend('statevector_simulator')
   job = execute(qc, backend_sv)
   r = job.result()
   outputstate = r.get_statevector(qc, decimals=8)
   return outputstate

####################################################################
def run_sv_noisy(qc, noise_model):
   """
   Simulate circuit on statevector simulator backend.
   :param qc: (QuantumCircuit) Quantum circuit to simulate.
   :return: (np.array) vector representing statevector of quantum circuit.
   """
   backend_sv = Aer.get_backend('statevector_simulator')
   job = execute(qc, backend_sv, noise_model=noise_model)
   r = job.result()
   outputstate = r.get_statevector(qc, decimals=8)
   return outputstate

####################################################################
def run_qasm(qc, num_shots, noise_model=None, device=''):
   """
   Simulate circuit on QASM simulator backend.
   :param qc: (QuantumCircuit) Quantum circuit to simulate.
   :param num_shots: (int) shots in run.
   :param device: (str) name of IBM device to simulate.
   :return: (dictionary) dict representing sampling of probability density function of quantum circuit.
   """
   backend_qasm = Aer.get_backend('qasm_simulator')
   if device == '':
      job = execute(qc, backend_qasm,shots = num_shots, noise_model=noise_model)
   else:
      provider = IBMQ.load_account()
      backend = provider.get_backend(device)
      properties = backend.properties()
      coupling_map = backend.configuration().coupling_map
      noise_model = noise.device.basic_device_noise_model(properties)
      basis_gates = noise_model.basis_gates
      job = execute(qc, backend_qasm, shots = num_shots, coupling_map = coupling_map, noise_model = noise_model, basis_gates = basis_gates)

   r = job.result().get_counts()
   return r

####################################################################
def run_qasm_full_data(qc, num_shots, noise_model, device=''):
   """
   Simulate circuit on QASM simulator backend.
   :param qc: (QuantumCircuit) Quantum circuit to simulate.
   :param num_shots: (int) shots in run.
   :param device: (str) name of IBM device to simulate.
   :return: (dictionary) dict representing sampling of probability density function of quantum circuit.
   """
   backend_qasm = Aer.get_backend('qasm_simulator')
   if device == '':
      job = execute(qc, backend_qasm,shots = num_shots, noise_model=noise_model)
   else:
      provider = IBMQ.load_account()
      backend = provider.get_backend(device)
      properties = backend.properties()
      coupling_map = backend.configuration().coupling_map
      noise_model = noise.device.basic_device_noise_model(properties)
      basis_gates = noise_model.basis_gates
      job = execute(qc, backend_qasm, shots = num_shots, coupling_map = coupling_map, noise_model = noise_model, basis_gates = basis_gates)

   r = job.result()
   return r

####################################################################
def run_qasm_noise(qc, num_shots, device=''):
   """
   Simulate circuit on QASM simulator backend.
   :param qc: (QuantumCircuit) Quantum circuit to simulate.
   :param num_shots: (int) shots in run.
   :param device: (str) name of IBM device to simulate.
   :return: (dictionary) dict representing sampling of probability density function of quantum circuit.
   """
   backend_qasm = Aer.get_backend('qasm_simulator')
   if device == '':
      job = execute(qc, backend_qasm,shots = num_shots, noise_model=noise_model)
   else:
      provider = IBMQ.load_account()
      backend = provider.get_backend(device)
      properties = backend.properties()
      coupling_map = backend.configuration().coupling_map
      noise_model = noise.device.basic_device_noise_model(properties)
      basis_gates = noise_model.basis_gates
      job = execute(qc, backend_qasm, shots = num_shots, coupling_map = coupling_map, noise_model = noise_model, basis_gates = basis_gates)

   r = job.result().get_counts()
   return r

####################################################################
def run_unitary(qc):
   """
   Simulate circuit on unitary simulator backend.
   :param qc: (QuantumCircuit) Quantum circuit to simulate.
   :return: (np.array) unitary matrix of quantum circuit.
   """
   backend_unitary = Aer.get_backend('unitary_simulator')
   job = execute(qc, backend_unitary).result()
   unitary = job.get_unitary(qc)
   return unitary



