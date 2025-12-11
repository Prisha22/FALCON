## Framework for Airfoil CFD and Analysis Optimization (FALCON)
<a name="top"></a>

[![OS](https://img.shields.io/badge/OS-linux%2C%20windows%2C%20macOS-0078D4)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

:star: Star us on GitHub - it is what keeps us going!

### About

FALCON is an open-source Python-based GUI framework designed to facilitate airfoil design, analysis, and optimization. It integrates airfoil parametrization techniques (CST and PARSEC), low-fidelity aerodynamic analysis using XFOIL, automated meshing with GMSH, and high-fidelity CFD simulations via SU2 (with automated post processing). The framework aims to streamline the airfoil design process by providing a user-friendly interface and automating various stages of analysis and optimization.

### Features

* User-friendly and easily accesible GUI.

* Airfoil parametrization using CST and PARSEC techniques.

* Embedded XFOIL for low-fidelity aerodynamic analysis.

* Automated meshing using GMSH

* CFD analyses via SU2 with automated post-processing.

* Validated solver settings are pre-chosen on the basis of the given initial conditions.

* Real-time residual monitoring

### Pre-requisites

* SU2 $\rightarrow$ https://su2code.github.io/ $\rightarrow$ Can be installed with or without MPI

* XFoil $\rightarrow$ https://web.mit.edu/drela/Public/web/xfoil/

* Python $\rightarrow$ https://www.python.org/downloads/

* Python specific packages $\rightarrow$

    - numpy
    - matplotlib
    - scipy
    - tkinter
    - pandas
    - gmsh

### Installation

```bash
python -m pip install -r requirements.txt
```

### Usage

```bash
python3 main.py
```

### References
1. Drela, M. (1989). XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils. In: Mueller, T.J. (eds) Low Reynolds Number Aerodynamics. Lecture Notes in Engineering, vol 54. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-84010-4_1

2. SU2: An Open-Source Suite for Multiphysics Simulation and Design Thomas D. Economon, Francisco Palacios, Sean R. Copeland, Trent W. Lukaczyk, and Juan J. Alonso, AIAA Journal 2016 54:3, 828-846. https://doi.org/10.2514/1.J053813

2. Kulfan, B.M. Universal parametric geometry representation method. J. Aircr. 2008, 45, 142–158. https://doi.org/10.2514/1.29958

3. Sobieczky, H., “Parametric Airfoils and Wings,” Recent Development of Aerodynamic Design Methodologies, Springer, New York, 1999, pp. 71–87. https://doi.org/10.1007/978-3-322-89952-1_4

4. Geuzaine, Christophe & Remacle, Jean-François. (2009). Gmsh: A 3-D Finite Element Mesh Generator with Built-in Pre- and Post-Processing Facilities. International Journal for Numerical Methods in Engineering. 79. 1309 - 1331. 10.1002/nme.2579. 