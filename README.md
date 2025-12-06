# FALCON
Framework for Airfoil CFD and anaLysis OptimizatioN

Tool for PARSEC and CST parameterization with Xfoil and SU2 analysis

Features:

-> User-friendly GUI enables airfoil to be chosen from a simple drop down list for analysis

-> Airfoil paramterization using CST and PARSEC of any chosen airfoil and gives the respective paramters

-> XFOIL analysis embedded in GUI for simulation over a range of angle of attack

-> Automated meshing of airfoil using GMSH

-> CFD analysis of airfoil in SU2 over a range of angle of attack with automated post-processing plots

-> Integrates SU2 with validated, automated solver settings chosen on the basis of the given initial conditions.

-> Real-time residual monitoring

Instructions for installation:

Pre-requisites:

-> SU2: Download Window's version at https://su2code.github.io/download.html. Add the address of the folder containing the executables (bin folder) to the system's PATH variable (Variable Name: SU2_CFD).

-> Microsoft MPI: Install msmpisetup.exe at https://www.microsoft.com/en-us/download/details.aspx?id=100593 . Verify that the path of the executable is stored in the system's PATH (Variable Name: mpiexec).

-> XFoil: Download Windows version at https://web.mit.edu/drela/Public/web/xfoil/

-> Python: Can be installed at https://www.python.org/downloads/

Python packages in requirements.txt can be installed in terminal using: python -m pip install -r requirements.txt
Run main.py for using the tool.


