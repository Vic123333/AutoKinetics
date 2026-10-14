# distutils: language = c++
# cython: language_level=3
"""Single compiled extension for AutoKinetics.

The module is assembled from include files so that numerics (integrators,
linear algebra) and kinetics (thermo, rates, reactors) share one C++ binary and
can pass ``OdeSystem*`` pointers to each other directly.

* ``_numerics.pxi``  - integrators, Newton/linear solvers, eigenvalues,
  ``OdeSystemHandle`` base class, Python-callback systems.
* ``_kinetics.pxi``  - compiled mechanisms, thermo, rates, reactor systems
  (subclasses of ``OdeSystemHandle``).
"""

cimport numpy as cnp
import numpy as np

cnp.import_array()

include "_numerics.pxi"
include "_kinetics.pxi"
