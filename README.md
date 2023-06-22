N.B. This code was originally devleped for 2102.05044 and then was heavily edited/modernized by Plato Deliyannis and Clement Charles (current developer/maintainer).  This code is nearly complete, but it is not quite ready for v1.  Thank you for your patience and stay tuned for the v1 release of the code!  You will find it at https://github.com/LBNL-HEP-QIS.

# Getting started

  conda create --name Tutorial
  conda activate Tutorial
  conda install pip
  pip install qiskit
  pip install matplotlib
  pip install gmpy2
  pip install qiskit_finance
  pip install jupyter
  python run_tests.py  #make sure this runs before doing anything else ... thank you Clement for making this nice test script!

# Classical studies

  jupyter notebook
  #go to lattice_qft/lattice_qft/examples/Day1Tutorial
  #should just run, but need to change sys.path.append(<path to QFT_Dev from current directory>)

# Quantum runs

