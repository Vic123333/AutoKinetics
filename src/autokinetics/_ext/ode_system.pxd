# Cython declaration of the shared C++ OdeSystem interface.
cdef extern from "autokinetics/ode_system.hpp" namespace "ak":
    cdef cppclass OdeSystem:
        size_t size()
        void rhs(double t, const double* y, double* dydt) except +
        bint has_jacobian()
        void jacobian(double t, const double* y, double* J) except +
        const unsigned char* nonnegative_mask()
