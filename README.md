N.B. This code was originally devleped for 2102.05044 and then was heavily edited/modernized by Plato Deliyannis and Clement Charles (current developer/maintainer).  This code is nearly complete, but it is not quite ready for v1.  Thank you for your patience and stay tuned for the v1 release of the code!  You will find it at https://github.com/LBNL-HEP-QIS.

# Tools for Quantum Field Theory on a Lattice

This code is compatible with Qiskit Version 0.39.5

  conda env create -f environment.yaml

Then, 

  conda activate qiskit_3

and then you should be able to run the code, e.g.:

  python HO_3q_test.py


# Notes on the code

(1) Classes ground_state, evolution, and wilson_line in harmonic_oscillator.lattice.py use the method
          
          qiskit.aqua.utils.circuit_factory.CircuitFactory.build()
    
    One could also invoke 

          qiskit.aqua.utils.circuit_factory.CircuitFactory.build_controlled()

    to construct a controlled version of the same circuit. However, there is an issue in the source where
    Rz gates are mapped to U1 gates incorrectly. Instead of using build_controlled(), first construct the circuit
    using build(), then use 

          qiskit.circuit.Gate.control()


(2a) The top directory <QFT_Dev>, and subdirectories <harmonic_oscillator> and <arithmetic_pkg> are packaged, so
     that dot notation can be used for importing different modules.


(2b) Internal imports all reference absolute paths from the top directory, QFT_Dev. Therefore, when constructing a
     Jupyter notebook, include

          sys.path.append(<path to QFT_Dev from current directory>)

    to avoid issues.

     
(3) Unit tests on individual functions can be enabled/disabled by uncommenting/commenting lines of the form

          @unittest.skip('temp')
