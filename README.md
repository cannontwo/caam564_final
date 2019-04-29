# caam564_final
Final project for CAAM 564. Local system identification of nonlinear systems using nonlinear optimization.

# Installation / Running
In order to run the code, you will need Python 3 and the following Python
packages:

* Pyomo
* Matplotlib
* Numpy
* Control
* Sklearn
* Scipy

The code in this repository is very much research code, and so there's no real
installation process. That being said, it does rely on IPOPT to do the actual
nonlinear optimization. You will need to modify the path to the ipopt
executable at the top of `matrix_opt.py` in order to run the test code.

# Files
* `matrix_opt.py` contains functions which interface with Pyomo to define the
  optimization problems that we care about in this project, then solve them.
  Note that there are a number of utility functions defining matrix arithmetic
  in this file because Pyomo does not natively support optimization over
  variables that define a matrix. This file contains both the linear system
  fitting code that is used to generate the results in my report and quadratic
  system fitting code that was not used for this project, but which will be
  used for my future research.
* `test_matrix_opt.py` contains a few simple tests of the functions in
  `matrix_opt.py`.
* `linear_model.py` contains a class representing a discrete time linear
  dynamical system model (i.e., a model defined by a state transition matrix A
  and a control matrix B).
* `test_linear_model.py` contains a few simple tests of the `linear_model.py` code.
* `one_one_d.py` contains code for defining and manipulating what I refer to as
  "1-1D" systems, which is to say discrete time, single-input single-output
  (SISO) dynamical systems.
* `test_one_one_d.py` contains simple tests of the code in `one_one_d.py`.
* `local_controllers.py` contains code defining different kinds of controllers
  that might be applied to a linear dynamical system. For the purposes of this
  project, I am only making use of the `DiscreteLQRController` class, which
  approximately solves the Discrete Time Algebraic Ricatti Equation for a given
  linear system. 
* `viz_1_1d.py` contains utilities for plotting 1-1D systems defined by the
  code in `one_one_d.py`, and is used to produce the figures in my report.
* `test_linear_fitting.py` contains the code defining the final testing that I
  did for my project. It essentially stitches together the functionality
  provided by all the rest of the files in this repository. This was the script
  that I ran to produce the figures in my report. By changing the integer
  variable `n` at the top of the script, you can test my code on different
  orders of polynomial 1-1D system.
