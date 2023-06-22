import qiskit
from lattice_qft.core.error_checking_utils import *
if qiskit.__qiskit_version__['qiskit'] != '0.40.1':
    VersionError('qiskit', '0.40.0', qiskit.__qiskit_version__['qiskit'])

qiskit_objects = ['QuantumCircuit', 'QuantumRegister', ]
QC_methods = ['data', 'initialize', 'compose', 'append','p']

__all__ = [
    'QuantumCircuit'
]

for method in qiskit_objects:
    if method in dir(qiskit):
        pass
    else:
        raise ValueError  #Remove
for method in QC_methods:
    if method in dir(qiskit.QuantumCircuit):
        pass
    else:
        raise ValueError  #Remove

from lattice_qft.core.hamiltonian import TimeEvolutionOfHamiltonian