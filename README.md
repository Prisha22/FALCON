# Airfoil Parameterization GUI
FALCON:
Framework for Airfoil CFD and anaLysis OptimizatioN
Tool for PARSEC and CST parameterization with Xfoil and SU2 analysis

Applications to be installed:
SU2: Download Window's version at https://su2code.github.io/download.html. Add the address of the folder containing the executables (bin folder) to the system's PATH variable (Variable Name: SU2_CFD).

Microsoft MPI: Install msmpisetup.exe at https://www.microsoft.com/en-us/download/details.aspx?id=100593 . Verify that the path of the executable is stored in the system's PATH (Variable Name: mpiexec).

XFoil: Download Windows version at https://web.mit.edu/drela/Public/web/xfoil/

Python: Can be installed at https://www.python.org/downloads/

Python packages in requirements.txt can be installed in terminal using: python -m pip install -r requirements.txt

Run main.py for using the tool.

Functions:

read_airfoil_coordinates: Parses through airfoil data file and outputs x,y coordinates as a numpy array

parsec: parameterizes the airfoil using PARSEC method

cst: parameterizes the airfoil using CST method

interpolate: Interpolates the chosen airfoil to achieve the number of coordinates as desired by the user and writes the interpolated data in a file - 'output.dat'

xfoil1: runs xfoil for analysis of output.dat at multiple angle of attack using pyxfoil package. Generate lift, drag plots along with pressure coefficient plot

meshing: Generates a structured C-mesh around output.dat using gmsh

hybrid: Generates a hybrid C-mesh around output.dat using gmsh

su2-analyzer: Has a default incompressible setting and allows user to change solver settings. Analyzes output.dat at multiple angle of attack and displays real-time residual data using generated 'history.csv' file. Also, saves lift, drag data.

main: Wraps all the above functions.

base.cfg: SU2 configuration file with default incompressible settings


