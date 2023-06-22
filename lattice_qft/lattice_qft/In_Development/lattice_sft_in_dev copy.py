import sys
# Local Imports

#sys.path.append("modules")
from lattice_qft.In_Development.lattice_in_dev import *
from lattice_qft.core.arithmetic_pkg.shear import construct_shear_circuit_theoretical
import lattice_qft.Scalar_Field_Theory.basic_operator_implementations as basic_op_cf
import scipy

# def array_modifier(multi_array):
#     temp_array = []
#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 temp_array.append(multi_array[i][j][k].tolist())
#     return np.array(temp_array)

class sft_lattice(Lattice):
    
    def __init__(self, dimension, nL, dx, nQ, num_ancilla, twisted=True):
        super().__init__(dimension, nL, dx, nQ, num_ancilla, twisted)
        self._nPhi = 2**nQ
        self.phiMax = np.sqrt(
            1
            / np.average(self.omega_list())
            * np.pi
            / 2
            * (self._nPhi - 1) ** 2
            / self._nPhi
            )
    @property
    def phiMax(self):
        return self._phiMax
    @phiMax.setter
    def phiMax(self, custom_phimax):
        self._phiMax = custom_phimax


    def omega_list(self):  # SFT
        return np.abs(
            (2.0
            / self._dx)
            * (np.sin(self.p_list()._grid * self._dx / 2.0))
        )

    def Gij_matrix(self):  # SFT
        """
        :return: (2D Array) Inverse of the correlation matrix, that gives the ground state of a 1D scalar field theory,
                                see https://arxiv.org/pdf/2102.05044.pdf Eq. S54.
        """
        #assert(self._dimension == 1)
        omegalist = self.omega_list()
        correlation = np.zeros((self._nL**(self._dimension), )*2)
        # for k in range(self._nL):
        #     for n in range(self._nL):
        #         xs, ps = self.x_list()._grid[k][n], self.p_list()._grid[k][n]
        #         #print(xs)
        #print(self.x_list()._grid)
        
        
        pos_coords = self.grids_to_points(self.x_list()._grid)
        m_coords = self.grids_to_points(self.p_list()._grid)
        o_coords = self.grids_to_points(self.omega_list())
        # for i in range(self._nL):
        #     for j in range(self._nL):
        #         for k in range(self._nL):
        #             o_coords.append([xo[i][j][k],yo[i][j][k],zo[i][j][k]])
        # for i in range(self._nL):
        #     for j in range(self._nL):
        #         for k in range(self._nL):
        #             pos_coords.append([x[i][j][k],y[i][j][k],z[i][j][k]])
        # for i in range(self._nL):
        #     for j in range(self._nL):
        #         for k in range(self._nL):
        #             m_coords.append([xp[i][j][k],yp[i][j][k],zp[i][j][k]]) 
        # use np.einsum     
        
        for i in range(self._nL**self._dimension):
            for j in range(self._nL**self._dimension):
                explist = np.exp(np.dot(m_coords,
                            (pos_coords[i] - pos_coords[j]))*1.0j
                    )
            
                # Should be times dx in the equation below!
                correlation[i, j] = ((np.sum(np.dot(np.real(explist),o_coords))
                    * (self._dx)) / (self._nL)
                )
        return correlation

    def KitaevWebbDMDecomposition(self):  # SFT
        """
        :return: (Tuple(2D array, 2D array)) D and M from MDM decomposition, see https://arxiv.org/pdf/0801.0342.pdf.
        """
        g = scipy.linalg.cholesky(self.Gij_matrix())
        d = np.real(np.diag(np.diag(g) ** 2))
        l = np.real(g @ np.diag(np.diag(g) ** -1))
        m = np.linalg.inv(l.conj().T)
        return d, m

    def apply_double_phi(self, pos1, pos2, fact):  # SFT
        """
        Applies the exponential of Exp[-i fact phi_pos1 phi_pos2]
        :param pos1: Lattice position of first phi
        :param pos2: Lattice position of second phi
        :param fact: The pre-factor
        :return: The quantum circuit
        """
        # This uses the fact that the phi operator can be written as
        # phi^(n) ~ pre * Sum_i 2^i sigma^(n)_i
        # phi^(n) phi^(m) ~ pre^2 * Sum_{i>j} 2^{i+j+1} sigma^(n)_i sigma^(n)_j
        # with pre = phi_max / (2^nQ - 1)
        qc = QuantumCircuit(self._q_register)
        # Deal with the wrapping and boundary condition
        wrap = 0
        point1 = [0] * self._dimension
        point2 = [0] * self._dimension
        for i in range(self._dimension):
            (tmpwrap, point1[i]) = divmod(pos1[i], self._nL)
            wrap += tmpwrap
            (tmpwrap, point2[i]) = divmod(pos2[i], self._nL)
            wrap += tmpwrap

        if self._twisted:
            fact *= (-1) ** wrap

        qbit_map = self._subregister_map(
            point1
        ) + self._subregister_map(point2)
        qreg = QuantumRegister(2 * self._nQ)
        prefact = fact * (self._phiMax / (2**self._nQ - 1)) ** 2

        for i in range(self._nQ):
            for j in range(self._nQ):
                fact = prefact * 2 ** (i + j)
                j += self._nQ  # Different lattice site
                qc_sub = basic_circuits.exp_pauli_product(
                    qreg, fact, [["Z", i], ["Z", j]]
                )  # minus sign because we are doing exp[-I t]
                qc.compose(qc_sub, qbit_map, inplace=True)
        return qc

    # Ground state for 1D lattice
    def ground_state(
        self, qc, q, full_correlation=False, shear=False, q_ancillas=None, params=None
    ):
        if full_correlation:
            self._shear = False  # Override shear if preparing fully correlated state

        # Compute optimal number of arithmetic qubits for shearing
        if (
            self._nL <= 2
        ):  # Never the case in practice, but this is a precondition that ensures r >= nQ.
            r = self._nQ
        else:
            r = int(self._nQ - 1 + np.ceil(np.log2(self._nL - 1)))

        self._shear_ancillas = (2 * r) + 1

        # def set_full_correlation(self, full_correlation):
        # self._full_correlation = full_correlation

        # def build(self, qc, q, q_ancillas=None, params=None):

        Gij = self.Gij_matrix()  # Correlation matrix
        correlation = inv(Gij)  # Inverse of the correlation matrix
        D, M = self.KitaevWebbDMDecomposition()

        if full_correlation:  # i.e. Qiskit default prep
            # Set up bounds list
            bound = (
                -self._phiMax,
                self._phiMax,
            )  # This seems bad, just have phi_max as a parameter
            bounds = []
            for j in range(self._nL**self._dimension):
                bounds.append(bound)

            # Prepare ground state
            ground_state = NormalDistribution(
                num_qubits=[self._nQ] * (self._nL** self._dimension),
                mu=[0] * (self._nL**self._dimension),
                sigma=correlation/2.0 ,
                bounds=bounds,
            )

            # Apply ground state prep circuit
            qc.compose(ground_state, q[:], inplace=True)

        else:  # i.e. Prepare 1D Gaussians using Qiskit, then shear

            covariance_diag = np.diagonal(D)

            # Prepare 1D Guassians
            ground_state = QuantumCircuit(q)
            for i in range(self._nL):
                [qbit_register, qbit_map] = self._get_subregister([i])
                qc_sub = QuantumCircuit(qbit_register)
                qc_sub.compose(
                    NormalDistribution(
                        self._nQ,
                        mu=0.0,
                        sigma=1 / covariance_diag[i] / 2,
                        # sigma=1/covariance_diag[i],
                        bounds=(-self._phiMax, self._phiMax),
                    ),
                    qbit_register[:],
                    inplace=True,
                )
                if shear == True:
                    qc_sub.x(
                        qbit_register[-1]
                    )  # Put into two's complement for shearing
                ground_state.compose(qc_sub, qbit_map, inplace=True)

            # Apply 1D Gaussian prep circuits
            qc.compose(ground_state, q[:], inplace=True)

            # Schmear
            if shear == True:

                qc.compose(
                    construct_shear_circuit_theoretical(
                        self._nL,
                        self._nQ,
                        M,
                        dim=self._dimension,
                        dx=self._dx,
                    ),
                    q_ancillas[:] + q[:],
                    inplace=True,
                )

                # Switch back to regular indexing (Not two's complement)
                for i in range(self._nL):
                    [qbit_register, qbit_map] = self._get_subregister([i])
                    qc.x(q[qbit_map[-1]])

        return qc
    def ground_state_energy(self):
        return np.sum(self.omega_list()) / 2.


class evolution(sft_lattice):
    """
    Class hanlding the time evolution for the given scalar lattice
    """

    def __init__(self, sft_lattice, evolve_time=1.0, trotter_steps=1.0):
        """
        :lattice:       (Lattice) Lattice on which the time evolution operates
        :evolve_time:   (float)   amount of time to evolve for
        :trotter_steps: (int)     number of trotter steps
        """
        self.lattice = sft_lattice
        #self.sft = TypeLatt
        self._evolve_time = evolve_time
        self._trotter_steps = trotter_steps

    def set_evolve_time(self, evolve_time):
        self._evolve_time = evolve_time

    def set_trotter_steps(self, trotter_steps):
        self._trotter_steps = trotter_steps

    def build(self, qc, q, q_ancillas=None, params=None):
        # Create the circuits involving pi and phi fields
        evolve_H_Pi = QuantumCircuit(self.lattice.get_q_register())
        evolve_H_Phi = QuantumCircuit(self.lattice.get_q_register())
        phi2 = basic_op_cf.Phi2Operator(self.lattice._phiMax)
        pi2 = basic_op_cf.Pi2Operator(self.lattice._phiMax)

        if self._trotter_steps == 0:
            t = 0
        else:
            t = self._evolve_time / self._trotter_steps

        for i in range(self.lattice._nL):  # Assumes a 1D Lattice
            evolve_H_Pi.compose(
                self.lattice.apply_single_operator(
                    [[i], pi2, [t * 0.5 * self.lattice._dx]]
                ),
                inplace=True,
            )
            evolve_H_Phi.compose(
                self.lattice.apply_single_operator(
                    [[i], phi2, [t * 1.0 / self.lattice._dx]]
                ),
                inplace=True,
            )
            evolve_H_Phi.compose(
                self.lattice.apply_double_phi([i - 1], [i], t * (-1.0) / self.lattice._dx),
                inplace=True,
            )
            # evolve_H_Phi.compose(self._lattice.apply_double_phi([i+1], [i], t * (-.5) / self._lattice._dx), inplace=True)
            # evolve_H_Phi.compose(self._lattice.apply_double_phi([i-1], [i], t * (-.5) / self._lattice._dx), inplace=True)

        # Combine them using Suzuki-Trotter
        evolve_Trotter = QuantumCircuit(self.lattice.get_q_register())
        for i in range(self._trotter_steps):
            evolve_Trotter.compose(evolve_H_Phi, inplace=True)
            evolve_Trotter.compose(evolve_H_Pi, inplace=True)

        qc.compose(evolve_Trotter, inplace=True)

    def build_connection(self, qc_evo, qc, qreg, areg, params=None):

        a_length = areg.size
        print(a_length)
        controlled_gate = qc_evo.to_gate().control(a_length)
        q_length = qreg.size
        qbit_list = [i for i in range(q_length)]
        qbit_list.insert(0, q_length)
        qc.append(controlled_gate, qbit_list)
        # print(qc.draw())


class wilson_line(sft_lattice):
    """
    Class hanlding the creationg of the Wilson line operators
    """

    def __init__(self, sft_lattice, dir1=[1], dir2=[-1], g=1, trotter_steps_per_dt=1):
        """
        :lattice:              (Lattice)   Lattice on which this Wilson line operator acts
        :dir1:                 (list(int)) vector of size dimension giving the **positive** direction of the Wilson line
        :dir2:                 (list(int)) vector of size dimension giving the **negative** direction of the Wilson line
        :g:                    (float)     value of the coupling constant
        :trotter_steps_per_dt: (int)       number of Trotter steps used for one time step dt = dx

        """
        self.lattice = sft_lattice
        self._direction1 = dir1
        self._direction2 = dir2
        self._g = g
        self._trotter_steps_per_dt = trotter_steps_per_dt

    def build(self, qc, q, q_ancillas=None, params=None):
        # direction1 = params[0]
        # direction2 = params[1]
        # g = params[2]
        # trotter_steps_per_dt = params[3]

        # Create the circuit for the time evolution over a single time step
        evolve = evolution(
            self.lattice,
            evolve_time=-self.lattice._dx,
            trotter_steps=self._trotter_steps_per_dt,
        )
        qc_evolution_step = QuantumCircuit(self.lattice.get_q_register())
        evolve.build(qc_evolution_step, self.lattice.get_q_register())
        # print("evolve params: time= %f, steps= %d" %(self._lattice._dx, self._trotter_steps_per_dt))
        point = [0] * self.lattice._dimension
        phi = basic_op_cf.PhiOperator(self.lattice._phiMax)

        # Find the center of the lattice to puth the cusp of the lattice
        center = int((self.lattice._nL - 1) / 2)

        # Create a circuit for a single time evolution step with time dt
        qc_result = QuantumCircuit(self.lattice.get_q_register())
        pos1 = [center] * self.lattice._dimension
        pos2 = [center] * self.lattice._dimension
        tot_steps = 0
        while True:
            tot_steps += 1
            # Get the position of the fields for the Wilson lines
            pos1 = list(map(add, pos1, self._direction1))
            pos2 = list(map(add, pos2, self._direction2))

            # Add a single evolution step
            qc_result.compose(qc_evolution_step.to_gate(label="Evolve"), inplace=True)
            # print('Forward time= ' + str(self._lattice._dx))

            # Add the exponential of the field operators
            # Note that apply_single_operator_list applies exp(op) with the opposite sign given
            qc_result.compose(
                self.lattice.apply_single_operator_list(
                    [
                        [pos1, phi, [-self._g * self.lattice._dx]],
                        [pos2, phi, [self._g * self.lattice._dx]],
                    ]
                ),
                inplace=True,
            )

            # Exit out of while loop if we have reached the end of the lattice
            for dim in range(self.lattice._dimension):
                finish = False
                if pos1[dim] == 0 or pos1[dim] == self.lattice._nL - 1:
                    finish = True
                if pos2[dim] == 0 or pos2[dim] == self.lattice._dimension - 1:
                    finish = True
            if finish:
                break
        # Add the full time evolution
        qc_devolution_step = QuantumCircuit(self.lattice.get_q_register())

        # evolve.set_evolve_time(-1 * tot_steps * self.lattice._dx)
        evolve.set_evolve_time(1 * tot_steps * self.lattice._dx)
        # print(evolve._evolve_time)

        evolve.set_trotter_steps(tot_steps * self._trotter_steps_per_dt)
        evolve.build(qc_devolution_step, self.lattice.get_q_register())

        # evolve.build(qc_result, self.lattice.get_q_register())
        qc_result.compose(
            qc_devolution_step.to_gate(label="inv(Evolve)"),
            qc_result.qubits,
            inplace=True,
        )

        # print('Circuit total steps: %d' %(tot_steps * self._trotter_steps_per_dt))
        # print('Circuit total time: %d' %(tot_steps * self._lattice._dx))
        # print('Backward time= ' + str(-1 * tot_steps * self._lattice._dx))

        qc.compose(qc_result, inplace=True)





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


from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
#Lattice()
lat = sft_lattice(2, 2, 1, 1, 1)
lr = lat.get_q_register() 
testqc = QuantumCircuit(lr)
import scipy

print(lat.Gij_matrix())


def array_to_coords(array):
    coords_output = []

    def generate_coordinates(dimensions, start, end):
        coordinates = []

        def generate_coordinates_recursive(current_dimension, current_coords):
            if current_dimension == dimensions:
                coordinates.append(tuple(current_coords))
                return

            for i in range(start, end + 1):
                new_coords = current_coords + [i]
                generate_coordinates_recursive(current_dimension + 1, new_coords)

        generate_coordinates_recursive(0, [])
        return coordinates
    
    array_indices = generate_coordinates(len(array),0,len(array)-1)
    for dim in iter(array):
        for x in array_indices:
            coords_output.append([dim[x]])
    return np.array(coords_output)



import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#print(array_to_coords(array))

from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city
import matplotlib.pyplot as plt
lat.ground_state(testqc,lr, full_correlation=True)
print(testqc.draw())
#testqc.measure_all()

simulator = Aer.get_backend('aer_simulator')
counts = execute(testqc, simulator, shots=1000).result().get_counts()
plot_histogram(counts)
plt.show()
#x, y = lat.x_list()._grid
# coords =[]
# for i in range(3):
#     for j in range(3):
#         coords.append([x[i][j],y[i][j]])
#print(lat.omega_list())
#print(scipy.linalg.cholesky(lat.Gij_matrix()))

#print(lat.Gij_matrix())
# xv, yv, zv =lat.omega_list()
#import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(x, y, marker='o')
# plt.show()
