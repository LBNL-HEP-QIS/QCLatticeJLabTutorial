# Script for executing unit tests within QFT_Dev and subdirectories.
# Comment/uncomment to disable/enable different tests.

import sys
import subprocess as sp

print('\n\n\n Running Testing Files/test_bc.py \n\n\n')
sp.call(['python', 'Testing_Files/test_bc.py'])

print('\n\n\n Running Testing Files/test_boi.py \n\n\n')
sp.call(['python', 'Testing_Files/test_boi.py'])

print('\n\n\n Running arithmetic_pkg/test_arithmetic.py \n\n\n')
sp.call(['python', 'arithmetic_pkg/test_arithmetic.py'])

print('\n\n\n Running arithmetic_pkg/test_series.py \n\n\n')
sp.call(['python', 'arithmetic_pkg/test_series.py'])