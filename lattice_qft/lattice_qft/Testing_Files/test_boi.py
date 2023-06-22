####################################################################################################
# General Imports                                                                                  #
####################################################################################################
import unittest
import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm
import math
# import sys
# sys.path.append('./')
# sys.path.append('../')
from qiskit import QuantumCircuit, QuantumRegister, execute, quantum_info, Aer


####################################################################################################
# Local Imports                                                                                    #
####################################################################################################
from test_base import TestCaseUtils # custom class that includes utility functions for testing
from lattice_qft.Scalar_Field_Theory.classical import *
import lattice_qft.core.basic_circuits as bc
import lattice_qft.Scalar_Field_Theory.basic_operator_implementations as boi
import lattice_qft.Scalar_Field_Theory.lattice_sft as lattice
#import harmonic_oscillator.settings as settings

global simulator_state
simulator_state= Aer.get_backend('statevector_simulator')


####################################################################################################
# Testing class                                                                                    #
####################################################################################################
class boiTestCase(TestCaseUtils):
    '''
        Class for testing the functions defined in basic_operator_implementations.py and lattice.py.
    '''


    #@unittest.skip('temp')
    def test_phi(self):
        '''
        Tests static method: boi.PhiOperator.phi().
        '''
        print('\nTesting PhiOperator().phi()...')
        prefact= 1.

        for nL in range(2, 5):
            for nQ in range(2, 5):
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)

                lat= Lattice(nL, nQ, dx=1, twist=1)
                #phi_max= lat.phiMax

                qc.compose(boi.PhiOperator(lat.phiMax).phi(q, prefact), inplace=True)
                
                mat_qc= quantum_info.Operator(qc).data
                mat_exp= expm(-1j * phiOp(lat) * prefact)
                #msg= '\n QuantumCircuit does not implement the expected operator. \n\n Expected Operator: ' + str(mat_exp) + ',\n\n QC Operator: ' + str(mat_qc)
                msg= '\n QuantumCircuit does not implement the expected e^(-i * φ) operator. Params: nL= %d, nQ= %d, prefactor= %.3f. \n' %(nL, nQ, prefact)
                self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)

    ####################################################################################################
    #@unittest.skip('temp')
    def test_phi2(self):
        '''
        Tests static method: boi.Phi2Operator.phi2()
        Things to test:
            even/odd register size -- check
            different prefactors   -- ...
        '''
        print('\nTesting Phi2Operator().phi2()...')
        prefact= 1.

        for nL in range(2, 5):
            for nQ in range(2, 5):
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)

                lat= Lattice(nL, nQ, dx=1, twist=1)
                #settings.phi_max= lat.phiMax

                qc.compose(boi.Phi2Operator(lat.phiMax).phi2(q, prefact), inplace=True)
                
                mat_qc= quantum_info.Operator(qc).data
                mat_exp= expm(-1j * phi2Op(lat) * prefact)
                msg= '\n QuantumCircuit does not implement the expected e^(-i * φ^2) operator. Params: nL= %d, nQ= %d, prefactor= %.3f. \n' %(nL, nQ, prefact)
                self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)

    ####################################################################################################
    #@unittest.skip('temp')
    def test_pi2(self):
        '''
        Tests static method: boi.Pi2Operator.phi2()
        Things to test:
            even/odd register size -- check
            different prefactors   -- ...
            swap True/False        -- check
        '''
        print('\nTesting Pi2Operator().pi2()...')
        prefact= 1.

        for nL in range(2, 5):
            for nQ in range(2, 5):
                for swap in [True, False]:
                    q= QuantumRegister(nQ)
                    qc= QuantumCircuit(q)

                    lat= Lattice(nL, nQ, dx=1, twist=1)
                    #settings.phi_max= lat.phiMax

                    qc.compose(boi.Pi2Operator(lat.phiMax).pi2(q, prefact, swap=swap), inplace=True)

                    mat_qc= quantum_info.Operator(qc).data
                    mat_exp= expm(-1j * pi2Op(lat) * prefact)
                    msg= '\n QuantumCircuit does not implement the expected e^(-i * π^2) operator. Params: nL= %d, nQ= %d, prefactor= %.3f. \n' %(nL, nQ, prefact)
                    self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)

    ####################################################################################################
    #@unittest.skip('temp')
    def test_phi4(self):
        '''
        Tests static method: boi.Phi4Operator.phi4()
        Things to test:
            even/odd register size -- check
            different prefactors   -- ...
        '''
        print('\nTesting Phi4Operator().phi4()...')
        prefact= 1.

        for nL in range(2, 5):
            for nQ in range(2, 5):
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)

                lat= Lattice(nL, nQ, dx=1, twist=1)
                #settings.phi_max= lat.phiMax

                qc.compose(boi.Phi4Operator(lat.phiMax).phi4(q, prefact), inplace=True)
                
                mat_qc= quantum_info.Operator(qc).data
                mat_exp= expm(-1j * phi4Op(lat) * prefact)
                msg= '\n QuantumCircuit does not implement the expected e^(-i * φ^4) operator. Params: nL= %d, nQ= %d, prefactor= %.3f. \n' %(nL, nQ, prefact)
                self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)

    ####################################################################################################
    #@unittest.skip('temp')
    def test_phi2phi4(self):
        '''
        Tests static method: boi.Phi2Phi4Operator.phi2_phi4()
        Things to test:
            even/odd register size -- check
            different prefactors   -- ...
        '''
        print('\nTesting Phi2Phi4Operator().phi2_phi4()...')
        prefact2= 1.
        prefact4= 1.

        for nL in range(2, 6):
            for nQ in range(2, 6):
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)

                lat= Lattice(nL, nQ, dx=1, twist=1)
                #settings.phi_max= lat.phiMax

                qc.compose(boi.Phi2Phi4Operator(lat.phiMax).phi2_phi4(q, prefact2, prefact4), inplace=True)
                
                mat_qc= quantum_info.Operator(qc).data
                mat_exp= expm(-1j * (phi2Op(lat) * prefact2) -1j * (phi4Op(lat) * prefact4))
                msg= '\n QuantumCircuit does not implement the expected e^(-i * (φ^2 + φ^4)) operator. Params: nL= %d, nQ= %d, φ^2 prefactor= %.3f, φ^4 prefactor= %.3f. \n' %(nL, nQ, prefact2, prefact4)
                self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)

    ####################################################################################################
    # Note: PhiTensorX and PiTensorY aren't used anywhere in the code.
    # TODO
    @unittest.skip('temp')
    def test_PhiTensorX(self):
        pass

    ####################################################################################################
    # TODO
    @unittest.skip('temp')
    def test_PiTensorY(self):
        pass

    ####################################################################################################
    #@unittest.skip('temp')
    def test_apply_single_operator(self):
        '''
        Tests lattice.Lattice().apply_single_operator()

        Things to test:
            different operators         -- check
            different sites             -- check
            different lattice size      -- check
            different phi digitizations --
            different dx                --
            twisted= True/False         --
            different prefactors        -- 
        '''
        print('\nTesting lattice.Lattice().apply_single_operator()...')
        twisted= True
        dx= 1
        params= [1.] # prefactor = 1
   
        for nL in range(2, 4):
            for nQ in range(2, 3):
                #print('nL, nQ = ' + str((nL, nQ)))
                lat_q= lattice.Lattice(1, nL, dx, nQ, 1, twisted=twisted) # dimension, num_ancilla= 1
                lat_cl= Lattice(nL, nQ, dx=dx, twist=int(twisted))
                #settings.phi_max= lat_cl.phiMax

                for ops in [(boi.PhiOperator(lat_cl.phiMax), phiOp(lat_cl)), 
                            (boi.Phi2Operator(lat_cl.phiMax), phi2Op(lat_cl)), 
                            (boi.Pi2Operator(lat_cl.phiMax), pi2Op(lat_cl))]:

                    for j in range(nL): # Need to add out of range positions
                        q= QuantumRegister(nQ*nL)
                        qc= QuantumCircuit(q)

                        pos= [j]
                        qc.compose(lat_q.apply_single_operator([pos, ops[0], params]), inplace=True)
                        mat_qc= quantum_info.Operator(qc).data

                        mat_exp= np.ones((1,1))
                        for i in range(nL):
                            if i == pos[0]:
                                mat_exp= np.kron(expm(-1j * ops[1]), mat_exp) # qiskit site indexing is the reverse of reading the tensor product left-to-right
                            else:
                                mat_exp= np.kron(np.identity(2**nQ), mat_exp) # qiskit site indexing is the reverse of reading the tensor product left-to-right

                        msg= '\n QuantumCircuit does not implement the expected ' + str(ops[0]) + ' operator. Params: nL= %d, nQ= %d, position= %d. \n' %(nL, nQ, j)
                        self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)

    ####################################################################################################
    #@unittest.skip('temp')
    def test_apply_single_operator_list(self):
        '''
        Tests lattice.Lattice().apply_single_operator()

        Things to test:
            different combos of operators -- check
            different sites               -- check
            different lattice size        -- check
            different phi digitizations   --
            different dx                  --
            twisted= True/False           --
            different prefactors          -- 
        '''
        print('\nTesting lattice.Lattice().apply_single_operator_list()...')
        twisted= True
        dx= 1
        nL= 2
        nQ= 2
        prefact= [1.] # prefactor

        lat_q= lattice.Lattice(1, nL, dx, nQ, 1, twisted=twisted) # dimension, num_ancilla= 1
        lat_cl= Lattice(nL, nQ, dx=dx, twist=int(twisted))
        #settings.phi_max= lat_cl.phiMax

        phi= boi.PhiOperator(lat_cl.phiMax)
        phi2= boi.Phi2Operator(lat_cl.phiMax)
        pi2= boi.Pi2Operator(lat_cl.phiMax)

        op_map= {phi: phiOp(lat_cl),
                 phi2: phi2Op(lat_cl),
                 pi2: pi2Op(lat_cl)
                }

        op_lists= [ [[[0], phi, prefact], [[1], phi, prefact]], [[[1], phi, prefact], [[0], phi, prefact]] ]
        for op_list in op_lists: # Need to add out of range positions
            q= QuantumRegister(nQ*nL)
            qc= QuantumCircuit(q)

            qc.compose(lat_q.apply_single_operator_list(op_list), inplace=True)

            mat_qc= quantum_info.Operator(qc).data

            site_ops= [] # list of length nL that stores the individual site operators
            for i in range(nL):
                site_ops.append(np.identity(2**nQ))

            for op in op_list:
                op_cl= op_map[op[1]]
                site= op[0][0]
                prefact= op[2]
                site_ops[site]= site_ops[site].dot(expm(-1j * op_cl * prefact))

            mat_exp= np.ones((1,1))
            for i in range(nL):
                mat_exp= np.kron(site_ops[i], mat_exp) # qiskit site indexing is the reverse of reading the tensor product left-to-right
    
            msg= '\n QuantumCircuit does not implement the expected operator list. Params: nL= %d, nQ= %d, prefactor= %.3f, operators: ' %(nL, nQ, prefact[0]) + str(op_list) + '. \n'
            self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)

    ####################################################################################################
    #@unittest.skip('temp')
    def test_apply_double_phi(self):
        '''
        Things to test:
            different sites             -- check
            different lattice size      -- 
            different phi digitizations --
            different dx                --
            twisted= True/False         --
            different prefactors        -- 
        '''
        print('\nTesting lattice.Lattice().apply_double_phi()...')
        twisted= True
        dx= 1
        prefact= 1.
   
        for nL in range(3, 5):
            for nQ in range(2, 3):
                #print('nL, nQ = ' + str((nL, nQ)))
                lat_q= lattice.sft_lattice(1, nL, dx, nQ, 1, twisted=twisted) # dimension, num_ancilla= 1
                lat_cl= Lattice(nL, nQ, dx=dx, twist=int(twisted))
                #settings.phi_max= lat_cl.phiMax

                if nL == 3: posList= [ [[0],[1]], [[0],[2]], [[1],[2]] ]
                if nL == 4: posList= [ [[0],[1]], [[0],[2]], [[0],[3]], [[1],[2]], [[1],[3]], [[2],[3]] ]
                for pos in posList:
                    q= QuantumRegister(nQ*nL)
                    qc= QuantumCircuit(q)

                    qc.compose(lat_q.apply_double_phi(pos[0], pos[1], prefact), inplace=True)
                    mat_qc= quantum_info.Operator(qc).data

                    phi= phiOp(lat_cl)
                    mat_exp= expm(-1j * operatorSite([phi]*len(pos), [pos[0][0], pos[1][0]], lat_cl, swapsites=True) * prefact)
                    msg= '\n QuantumCircuit does not implement the expected operator list. Params: nL= %d, nQ= %d, prefactor= %.3f, positions: ' %(nL, nQ, prefact) + str(pos) + '. \n'
                    self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)

    ####################################################################################################
    # TODO
    @unittest.skip('temp')
    def test_apply_single_prep(self):
        '''
        Tests lattice.Lattice().apply_single_prep

        Note: note used anywhere
        '''
        pass

    ####################################################################################################
    @unittest.skip('temp')
    def test_ground_state(self):
        '''
        Things to test:
            full_correlation or not -- check
            sheared or not          -- TBD
            different lattice sizes -- check
        '''
        print('\nTesting lattice.ground_state().build()...')

        twisted= True
        dx= 1
        
        for nL in range(3, 5):
            for nQ in range(2, 4):
                lat_q= lattice.Lattice(1, nL, dx, nQ, 1, twisted=twisted) # dimension, num_ancilla= 1
                lat_cl= Lattice(nL, nQ, dx=dx, twist=int(twisted))
                #settings.phi_max= lat_cl.phiMax

                # Full correlation
                gs= lattice.ground_state(lat_q, full_correlation=True, shear= False)

                qc= QuantumCircuit(lat_q.get_q_register())
                gs.build(qc, lat_q.get_q_register())

                #print(qc.draw())
                #print(corr, params)
                sv_sim= execute(qc, simulator_state).result().get_statevector()
                sv_exp= createEigenstate([0]*nL, lat_cl)
                msg='\n QuantumCircuit does not construct the expected ground state. Params: full_correlation= True, nL= %d, nQ= %d. \n' %(nL, nQ)
                self.compareSV(sv_sim, sv_exp, delta= 1e-14, msg= msg)


                # Independent 1D Gaussians 
                gs= lattice.ground_state(lat_q, full_correlation=False, shear= False)

                qc= QuantumCircuit(lat_q.get_q_register())
                gs.build(qc, lat_q.get_q_register())

                #print(qc.draw())
                sv_sim= execute(qc, simulator_state).result().get_statevector()
                sv_exp= createKWstate([0]*nL, lat_cl)
                msg='\n QuantumCircuit does not construct the expected ground state. Params: full_correlation= False, shearing= False, nL= %d, nQ= %d. \n' %(nL, nQ)
                self.compareSV(sv_sim, sv_exp, delta= 1e-14, msg= msg)
                

                # Independent 1D Gaussians + Shearing
                if nL >= 3: r= int(nQ - 1 + np.ceil(np.log2(nL-1))) # number of ancillary qubits
                else: r= nQ

                lat_q= lattice.Lattice(1, nL, dx, nQ, 2*r + 1, twisted=twisted) # dimension, num_ancilla= 2r+1
                gs= lattice.ground_state(lat_q, full_correlation=False, shear= True)

                qc= QuantumCircuit(lat_q.get_a_register(), lat_q.get_q_register())
                gs.build(qc, lat_q.get_q_register(), lat_q.get_a_register())

                #print(qc.draw())
                sv_sim= execute(qc, simulator_state).result().get_statevector()
                sv_exp= createKWground(lat_cl)

                # Ignore ancillary qubits
                #print(len(sv_sim), len(sv_exp), 'nL= %d, nQ= %d' %(nL, nQ))
                sv_sim= sv_sim[np.where(sv_sim >= 1e-15)]
                sv_exp= sv_exp[np.where(sv_exp >= 1e-15)]
                msg='\n QuantumCircuit does not construct the expected ground state. Params: full_correlation= False, shearing= True, nL= %d, nQ= %d. \n' %(nL, nQ)
                self.compareSV(sv_sim, sv_exp, delta= 1e-14, msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_hamiltonian(self):
        '''
        Things to test for the operator:
            different lattice sizes -- check
            different trotter times -- check
            different trotter steps -- check
        '''
        print('\nTesting lattice.evolution().build()...')
        twisted= True
        dx= 1
        
        # Test Operator
        for nL in range(2, 4):
            for nQ in range(2, 4):
                lat_q= lattice.Lattice(1, nL, dx, nQ, 1, twisted=twisted) # dimension, num_ancilla= 1
                lat_cl= Lattice(nL, nQ, dx=dx, twist=int(twisted))
                #settings.phi_max= lat_cl.phiMax

                for tt in [0, 1., -1.]: # tt: trotter time
                    for ts in [0, 1, 2]: # ts: trotter steps
                        evo= lattice.evolution(lat_q, evolve_time= tt, trotter_steps= ts) # evolve_time= 1, trotter_steps= 1
                        qc= QuantumCircuit(lat_q.get_q_register())
                        evo.build(qc, lat_q.get_q_register()) 

                        mat_qc= quantum_info.Operator(qc).data
                        mat_exp= evolveHTrotter(tt, ts, lat_cl)
                        msg='\n QuantumCircuit does not implement the expected Hamiltonian operator. Params: nL= %d, nQ= %d, Trotter time= %.3f, steps= %d. \n' %(nL, nQ, tt, ts)
                        self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)


        # Test Eigenstatedness of the ground state, statevectors after applying time evolution
        for nL in range(3, 4):
            for nQ in range(2, 3):
                # Test the method used to generate plots -- i.e. apply inverse ground state
                lat_q= lattice.Lattice(1, nL, dx, nQ, 1, twisted=twisted) # dimension, num_ancilla= 1
                lat_cl= Lattice(nL, nQ, dx=dx, twist=int(twisted))
                #settings.phi_max= lat_cl.phiMax

                for tt in [0, 1.0, -1.0]: # tt: trotter time
                    for ts in [0, 1, 2]: # ts: trotter steps
                        qc= QuantumCircuit(lat_q.get_q_register())

                        gs= lattice.ground_state(lat_q, full_correlation=False, shear= False)
                        gs.build(qc, lat_q.get_q_register())

                        evo= lattice.evolution(lat_q, evolve_time= tt, trotter_steps= ts) # evolve_time= 1, trotter_steps= 1
                        evo.build(qc, lat_q.get_q_register()) 

                        q_aux= QuantumRegister(nQ*nL)
                        qc_aux= QuantumCircuit(q_aux)
                        gs.build(qc_aux, q_aux)

                        qc.compose(qc_aux.inverse(), inplace=True)

                        #print(qc.draw())
                        sv_sim= execute(qc, simulator_state).result().get_statevector()

                        ground= createKWstate([0]*nL, lat_cl)
                        sv_exp0= ground.conj() @ evolveHTrotter(tt, ts, lat_cl).dot(ground)
                        
                        # if ts == 0, phase= 1
                        phase= np.exp(-1j * tt * lat_q.ground_state_energy() * int(ts != 0))

                        #print(sv_sim, sv_exp)
                        #print('Expected probability in the 0 state, tt= %.2f, ts=%d:  ' %(tt, ts) + str(abs(sv_exp0)**2))
                        msg='\n Ground state is not an eigenstate of the Hamiltonian. Params: nL= %d, nQ= %d, Trotter time= %.3f, steps= %d. \n' %(nL, nQ, tt, ts)
                        self.assertAlmostEqual(abs(sv_sim[0])**2, abs(sv_exp0)**2, delta= 1e-14, msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_wilson_line(self):
        '''
        Things to test for the operator:
            different lattice sizes -- check
            different trotter times -- check
            different trotter steps -- check
        '''
        print('\nTesting lattice.wilson_line().build()...')
        twisted= True
        dx= 1
        for g in [1, -0.7, 2.4]:

            # Test Operator
            #for nL, nQ in [(3, 2), (3, 3), (3, 4), (5, 2)]:
            for nL, nQ in [(3, 2), (3, 3)]:
                lat_q= lattice.Lattice(1, nL, dx, nQ, 1, twisted=twisted) # dimension, num_ancilla= 1
                lat_cl= Lattice(nL, nQ, dx=dx, twist=int(twisted))
                #settings.phi_max= lat_cl.phiMax

                for ts in [0, 1, 2]: # ts: trotter steps
                    #print('Constructing QC...')
                    wil= lattice.wilson_line(lat_q, [1], [-1], g, ts)
                    qc= QuantumCircuit(lat_q.get_q_register())
                    #wil.build(qc, lat_q.get_q_register(), params= [[1], [-1], g, ts]) 
                    wil.build(qc, lat_q.get_q_register()) 

                    mat_qc= quantum_info.Operator(qc).data

                    #print('Computing classical...')
                    mat_exp= WilsonTrotter(g, ts, lat_cl)

                    #print(np.diag(mat_qc))
                    #print(np.diag(mat_exp))
                    msg= '\n QuantumCircuit does not implement the expected Wilson line operator. Params: nL= %d, nQ= %d, g= %.3f, Trotter steps= %d. \n' %(nL, nQ, g, ts)
                    self.compareOperators(mat_qc, mat_exp, delta= 1e-14, msg= msg)


########################################################################################################
if __name__ == '__main__':
    unittest.main()