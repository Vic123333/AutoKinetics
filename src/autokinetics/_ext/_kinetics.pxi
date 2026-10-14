# Kinetics bindings: compiled mechanisms and reactor ODE systems.
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector


cdef extern from "autokinetics/kinetics/mechanism.hpp" namespace "ak":
    cdef cppclass MechanismData:
        MechanismData()
        int n_species, n_reactions, phase
        vector[int] fwd_ptr, fwd_idx, rev_ptr, rev_idx, nu_ptr, nu_idx, rtype
        vector[double] fwd_ord, rev_ord, nu_val, dnu, A, b, Ea_R
        vector[int] tb_rxn, fo_rxn, fo_kind, plog_rxn, plog_ptr, rev_mode, th_ptr
        vector[double] tb_eff, fo_A, fo_b, fo_Ea_R, fo_params
        vector[double] plog_lnP, plog_A, plog_b, plog_Ea_R
        vector[double] rev_A, rev_b, rev_Ea_R, K_ref, K_Tref, K_dH_R
        vector[unsigned char] has_thermo
        vector[double] th_Tlo, th_Thi, th_coef, dH_rxn

    cdef cppclass Kinetics:
        Kinetics(MechanismData) except +
        int n_species()
        int n_reactions()
        int phase()
        cbool all_thermo()
        void species_thermo(double T, double* cp, double* h, double* s) except +
        void equilibrium_constants(double T, double* Kc) except +
        void rate_constants(double T, double P, const double* c, double* kf, double* kr, double* M) except +
        void rates_of_progress(double T, double P, const double* c, double* qf, double* qr, double* q) except +
        void production_rates(double T, double P, const double* c, double* wdot, double* q) except +
        void jacobian_conc(double T, double P, const double* c, double* J) except +
        void reaction_enthalpies(double T, double* dH) except +


cdef extern from "autokinetics/kinetics/reactor.hpp" namespace "ak":
    cdef cppclass ReactorConfig:
        ReactorConfig()
        int energy, constraint, flow
        double P0, UA_V, rho_cp, tau, T_in
        vector[double] T_prog_t, T_prog_T, T_env_t, T_env_T, c_in

    cdef cppclass ReactorSystem(OdeSystem):
        ReactorSystem(shared_ptr[Kinetics], ReactorConfig) except +
        void observe(double t, const double* y, double* c, double* T, double* P, double* V) except +
        long n_rhs()
        long n_jac()


cdef vector[int] _ivec(a):
    cdef vector[int] v
    for x in np.asarray(a, dtype=np.int64).ravel():
        v.push_back(<int>x)
    return v


cdef vector[double] _dvec(a):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())
    cdef vector[double] v
    v.reserve(arr.shape[0])
    cdef Py_ssize_t i
    for i in range(arr.shape[0]):
        v.push_back(arr[i])
    return v


cdef inline cnp.ndarray _as_c(a, Py_ssize_t n, str what):
    cdef cnp.ndarray arr = np.ascontiguousarray(a, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{what}: Länge {n} erwartet, erhalten Form {np.shape(a)}")
    return arr


cdef class KineticsModel:
    """Compiled C++ kinetics for a :class:`CompiledMechanism` (SI units)."""
    cdef shared_ptr[Kinetics] kin
    cdef readonly int n_species
    cdef readonly int n_reactions

    def __cinit__(self, cm):
        cdef MechanismData d
        d.n_species = cm.n_species
        d.n_reactions = cm.n_reactions
        d.phase = cm.phase
        d.fwd_ptr = _ivec(cm.fwd_ptr); d.fwd_idx = _ivec(cm.fwd_idx); d.fwd_ord = _dvec(cm.fwd_ord)
        d.rev_ptr = _ivec(cm.rev_ptr); d.rev_idx = _ivec(cm.rev_idx); d.rev_ord = _dvec(cm.rev_ord)
        d.nu_ptr = _ivec(cm.nu_ptr); d.nu_idx = _ivec(cm.nu_idx); d.nu_val = _dvec(cm.nu_val)
        d.dnu = _dvec(cm.dnu); d.rtype = _ivec(cm.rtype)
        d.A = _dvec(cm.A); d.b = _dvec(cm.b); d.Ea_R = _dvec(cm.Ea_R)
        d.tb_rxn = _ivec(cm.tb_rxn); d.tb_eff = _dvec(cm.tb_eff)
        d.fo_rxn = _ivec(cm.fo_rxn); d.fo_kind = _ivec(cm.fo_kind)
        d.fo_A = _dvec(cm.fo_A); d.fo_b = _dvec(cm.fo_b); d.fo_Ea_R = _dvec(cm.fo_Ea_R)
        d.fo_params = _dvec(cm.fo_params)
        d.plog_rxn = _ivec(cm.plog_rxn); d.plog_ptr = _ivec(cm.plog_ptr)
        d.plog_lnP = _dvec(cm.plog_lnP); d.plog_A = _dvec(cm.plog_A); d.plog_b = _dvec(cm.plog_b)
        d.plog_Ea_R = _dvec(cm.plog_Ea_R)
        d.rev_mode = _ivec(cm.rev_mode); d.rev_A = _dvec(cm.rev_A); d.rev_b = _dvec(cm.rev_b)
        d.rev_Ea_R = _dvec(cm.rev_Ea_R); d.K_ref = _dvec(cm.K_ref); d.K_Tref = _dvec(cm.K_Tref)
        d.K_dH_R = _dvec(cm.K_dH_R)
        for x in np.asarray(cm.has_thermo, dtype=np.uint8):
            d.has_thermo.push_back(<unsigned char>x)
        d.th_ptr = _ivec(cm.th_ptr); d.th_Tlo = _dvec(cm.th_Tlo); d.th_Thi = _dvec(cm.th_Thi)
        d.th_coef = _dvec(cm.th_coef); d.dH_rxn = _dvec(cm.dH_rxn)
        self.kin = make_shared[Kinetics](d)
        self.n_species = cm.n_species
        self.n_reactions = cm.n_reactions

    def species_thermo(self, double T):
        """(cp, h, s) molar, SI; NaN for species without thermo data."""
        cdef cnp.ndarray[cnp.float64_t] cp = np.empty(self.n_species)
        cdef cnp.ndarray[cnp.float64_t] h = np.empty(self.n_species)
        cdef cnp.ndarray[cnp.float64_t] s = np.empty(self.n_species)
        self.kin.get().species_thermo(T, &cp[0], &h[0], &s[0])
        return cp, h, s

    def equilibrium_constants(self, double T):
        cdef cnp.ndarray[cnp.float64_t] K = np.empty(max(self.n_reactions, 1))
        if self.n_reactions:
            self.kin.get().equilibrium_constants(T, &K[0])
        return K[:self.n_reactions]

    def rate_constants(self, double T, double P, c):
        cdef cnp.ndarray[cnp.float64_t] cc = _as_c(c, self.n_species, "c")
        cdef Py_ssize_t nr = max(self.n_reactions, 1)
        cdef cnp.ndarray[cnp.float64_t] kf = np.zeros(nr)
        cdef cnp.ndarray[cnp.float64_t] kr = np.zeros(nr)
        cdef cnp.ndarray[cnp.float64_t] M = np.ones(nr)
        if self.n_reactions:
            self.kin.get().rate_constants(T, P, &cc[0], &kf[0], &kr[0], &M[0])
        return kf[:self.n_reactions], kr[:self.n_reactions], M[:self.n_reactions]

    def rates_of_progress(self, double T, double P, c):
        cdef cnp.ndarray[cnp.float64_t] cc = _as_c(c, self.n_species, "c")
        cdef Py_ssize_t nr = max(self.n_reactions, 1)
        cdef cnp.ndarray[cnp.float64_t] qf = np.zeros(nr)
        cdef cnp.ndarray[cnp.float64_t] qr = np.zeros(nr)
        cdef cnp.ndarray[cnp.float64_t] q = np.zeros(nr)
        if self.n_reactions:
            self.kin.get().rates_of_progress(T, P, &cc[0], &qf[0], &qr[0], &q[0])
        return qf[:self.n_reactions], qr[:self.n_reactions], q[:self.n_reactions]

    def production_rates(self, double T, double P, c):
        cdef cnp.ndarray[cnp.float64_t] cc = _as_c(c, self.n_species, "c")
        cdef cnp.ndarray[cnp.float64_t] w = np.zeros(self.n_species)
        cdef cnp.ndarray[cnp.float64_t] q = np.zeros(max(self.n_reactions, 1))
        self.kin.get().production_rates(T, P, &cc[0], &w[0], &q[0])
        return w

    def jacobian_conc(self, double T, double P, c):
        cdef cnp.ndarray[cnp.float64_t] cc = _as_c(c, self.n_species, "c")
        cdef cnp.ndarray[cnp.float64_t, ndim=2] J = np.zeros((self.n_species, self.n_species))
        self.kin.get().jacobian_conc(T, P, &cc[0], &J[0, 0])
        return J

    def reaction_enthalpies(self, double T):
        cdef cnp.ndarray[cnp.float64_t] dH = np.zeros(max(self.n_reactions, 1))
        if self.n_reactions:
            self.kin.get().reaction_enthalpies(T, &dH[0])
        return dH[:self.n_reactions]


cdef class Reactor(OdeSystemHandle):
    """Reactor model as a C++ ODE system (see cpp/include/autokinetics/kinetics/reactor.hpp)."""
    cdef ReactorSystem* reactor
    cdef readonly KineticsModel kinetics
    cdef readonly int n_species
    cdef readonly int n_state

    def __cinit__(self, KineticsModel kinetics, *, int energy=0, int constraint=0, int flow=0,
                  double P0=1e5, T_program=None, double UA_V=0.0, T_env_program=None,
                  double rho_cp=4.18e6, double tau=0.0, c_in=None, double T_in=300.0):
        cdef ReactorConfig cfg
        cfg.energy = energy
        cfg.constraint = constraint
        cfg.flow = flow
        cfg.P0 = P0
        cfg.UA_V = UA_V
        cfg.rho_cp = rho_cp
        cfg.tau = tau
        cfg.T_in = T_in
        if T_program is not None:
            tp = np.asarray(T_program, dtype=float).reshape(-1, 2)
            cfg.T_prog_t = _dvec(tp[:, 0]); cfg.T_prog_T = _dvec(tp[:, 1])
        if T_env_program is not None:
            te = np.asarray(T_env_program, dtype=float).reshape(-1, 2)
            cfg.T_env_t = _dvec(te[:, 0]); cfg.T_env_T = _dvec(te[:, 1])
        if c_in is not None:
            cfg.c_in = _dvec(c_in)
        self.kinetics = kinetics
        self.reactor = new ReactorSystem(kinetics.kin, cfg)
        self.ptr = self.reactor
        self.n_species = kinetics.n_species
        self.n_state = self.reactor.size()

    def __dealloc__(self):
        if self.reactor != NULL:
            del self.reactor
            self.reactor = NULL
        self.ptr = NULL

    def eval_rhs(self, double t, y):
        cdef cnp.ndarray[cnp.float64_t] yy = _as_c(y, self.n_state, "y")
        cdef cnp.ndarray[cnp.float64_t] f = np.empty(self.n_state)
        self.reactor.rhs(t, &yy[0], &f[0])
        return f

    def eval_jacobian(self, double t, y):
        cdef cnp.ndarray[cnp.float64_t] yy = _as_c(y, self.n_state, "y")
        cdef cnp.ndarray[cnp.float64_t, ndim=2] J = np.empty((self.n_state, self.n_state))
        self.reactor.jacobian(t, &yy[0], &J[0, 0])
        return J

    def observe(self, t, Y):
        """Map states (m x n_state) at times t (m,) to (c[m, ns], T[m], P[m], V[m])."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] tt = np.ascontiguousarray(np.atleast_1d(t), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] YY = np.ascontiguousarray(np.atleast_2d(Y), dtype=np.float64)
        if YY.shape[1] != self.n_state or YY.shape[0] != tt.shape[0]:
            raise ValueError("observe: Form von t/Y passt nicht zum Reaktorzustand")
        cdef Py_ssize_t m = YY.shape[0], k
        cdef cnp.ndarray[cnp.float64_t, ndim=2] C = np.empty((m, self.n_species))
        cdef cnp.ndarray[cnp.float64_t] T = np.empty(m)
        cdef cnp.ndarray[cnp.float64_t] P = np.empty(m)
        cdef cnp.ndarray[cnp.float64_t] V = np.empty(m)
        for k in range(m):
            self.reactor.observe(tt[k], &YY[k, 0], &C[k, 0], &T[k], &P[k], &V[k])
        return C, T, P, V

    @property
    def counters(self):
        return {"rhs": self.reactor.n_rhs(), "jacobian": self.reactor.n_jac()}
