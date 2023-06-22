from lattice_qft.core.lattice import Lattice
from lattice_qft.Scalar_Field_Theory.basic_operator_implementations import PhiOperator
from qiskit import QuantumCircuit

"""
Random Test Code & Functions. To be removed in final release
"""

lat  = Lattice(1, 4, 1, 1, 1, twisted = False)

phi = PhiOperator(1)
nL= 3 # Number of lattice sites
nQ= 2
dx= 1. # Lattice spacing, usually called a in the lattice literature
g= 0.526 # Wilson line coupling constant
dim= 3
num_ancilla= 1
trotter_per_dt1= 1
trotter_per_dt2= 2
import numpy as np

test_arr = [[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]],[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]],[[2., 2., 2.],[2., 2., 2.],[2., 2., 2.]]]


for i in range(3):
    for j in range(3):
        print('i: ', i, 'j: ', j)
            
        

def correlation_matrix(positions, momenta):
    # Check if the dimensions of positions and momenta match
    if positions.shape != momenta.shape:
        raise ValueError("Dimensions of positions and momenta don't match.")
    
    num_points = positions.shape[0]
    correlation_mat = np.zeros((num_points, num_points))
    
    # Compute the correlation matrix
    for i in range(num_points):
        for j in range(num_points):
            correlation_mat[i, j] = np.dot(positions[i], positions[j]) + np.dot(momenta[i], momenta[j])
    
    return correlation_mat

# Example usage
positions = np.array([[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[1., 1., 1.],[1., 1., 1.],[1., 1., 1.],[2., 2., 2.],[2., 2., 2.],[2., 2., 2.]])
momenta = np.array([[0.1, 0., 0.2],[0.2, 0.9, 0.1],[0., 0.8, 0.],[0.2, 1., 1.],[1., 1., 1.],[1., 1., 1.],[0., 2.5, 2.],[2., 2., 2.],[2., 2., 2.]])

correlation_mat = correlation_matrix(positions, momenta)



print(momenta[0])

test = [[0],[1],[2],[3],[4]]

print(np.dot(test,2))





