# ODL Modification

Users need to substitute original files in ODL with files provided in this folder. The reconstruction result won't change if users refuse to do this change. However, files in this foler includes functions that could be helpful during reconstruction. 

Modified "iterative.py":
In this function, users have the options on: 
1. Visualizing time spent by Conjugate Gradient Normal during each iteration.
2. Recording the difference between Radon tranform of reconstructed image (after each iteration) and projection data.
Path of original "iterative.py": .../odl/odl/solvers/iterative/iterative.py

Modified "forwardbackward.py":
In this function, users have the choice of:
1. Visualizing time spent by Forward-Backward Prime-Dual Algorithm during each iteration.
2. Recording the difference between Radon tranform of reconstructed image (after each iteration) and projection data.
3. Recording the output of minimization function after each iteration.
Path of original "iterative.py": .../odl/odl/solvers/nonsmooth/forward_backward.py

