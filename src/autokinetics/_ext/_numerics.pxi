# Numerics bindings (owned by the numerics work package).
from autokinetics._ext.ode_system cimport OdeSystem


cdef class OdeSystemHandle:
    """Owns or borrows a C++ ``ak::OdeSystem``; base class of all systems."""
    cdef OdeSystem* ptr

    def __cinit__(self):
        self.ptr = NULL

    @property
    def size(self):
        if self.ptr == NULL:
            raise RuntimeError("OdeSystemHandle is not initialised")
        return self.ptr.size()
