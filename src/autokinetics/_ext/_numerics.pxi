# Numerics bindings (owned by the numerics work package).
#
# Python-visible names defined here:
#   OdeSystemHandle             base class of every compiled ODE system
#   PythonOdeSystem             wraps a Python object with rhs(t, y) [/ jacobian(t, y)]
#   LinearOdeSystem, RobertsonSystem, VanDerPolSystem   built-in test problems
#   num_*                       low-level functions used by autokinetics.numerics
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.vector cimport vector

from autokinetics._ext.ode_system cimport OdeSystem
from autokinetics._ext cimport num_core as nc


# ---------------------------------------------------------------------------
# conversion helpers
# ---------------------------------------------------------------------------
cdef nc.Matrix _num_matrix(object A, str what="matrix") except *:
    cdef cnp.ndarray arr = np.ascontiguousarray(A, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{what} must be two-dimensional, got shape {np.shape(arr)}")
    cdef nc.Matrix M = nc.Matrix(arr.shape[0], arr.shape[1], 0.0)
    if arr.size > 0:
        memcpy(M.data.data(), cnp.PyArray_DATA(arr), arr.size * sizeof(double))
    return M


cdef object _num_from_matrix(const nc.Matrix& M):
    cdef cnp.ndarray out = np.empty((M.rows, M.cols), dtype=np.float64)
    if M.rows * M.cols > 0:
        memcpy(cnp.PyArray_DATA(out), M.data.data(), M.rows * M.cols * sizeof(double))
    return out


cdef vector[double] _num_vec(object x, str what="vector") except *:
    cdef cnp.ndarray arr = np.ascontiguousarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional, got shape {np.shape(arr)}")
    cdef vector[double] v
    v.resize(arr.shape[0])
    if arr.shape[0] > 0:
        memcpy(v.data(), cnp.PyArray_DATA(arr), arr.shape[0] * sizeof(double))
    return v


cdef object _num_from_vec(const vector[double]& v):
    cdef cnp.ndarray out = np.empty(v.size(), dtype=np.float64)
    if v.size() > 0:
        memcpy(cnp.PyArray_DATA(out), v.data(), v.size() * sizeof(double))
    return out


cdef str _num_str(const string& s):
    return s.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# ODE systems
# ---------------------------------------------------------------------------
cdef class OdeSystemHandle:
    """Owns or borrows a C++ ``ak::OdeSystem``; base class of all compiled systems.

    Subclasses (e.g. reactor models in ``_kinetics.pxi``) set ``self.ptr`` in
    ``__cinit__``/``__init__``.  If ``owns_ptr`` is true the object is deleted
    when the handle is collected; otherwise it is borrowed and ``_keepalive``
    should reference whatever Python object keeps it alive.
    """
    cdef OdeSystem* ptr
    cdef bint owns_ptr
    cdef object _keepalive
    cdef bint _releases_gil
    cdef object _py_error

    def __cinit__(self):
        self.ptr = NULL
        self.owns_ptr = False
        self._keepalive = None
        self._releases_gil = True
        self._py_error = None

    def __dealloc__(self):
        if self.owns_ptr and self.ptr != NULL:
            del self.ptr
        self.ptr = NULL

    cdef OdeSystem* _require(self) except NULL:
        if self.ptr == NULL:
            raise RuntimeError(f"{type(self).__name__} is not initialised (no C++ system attached)")
        return self.ptr

    cdef int _reraise_callback_error(self) except -1:
        cdef object err = self._py_error
        if err is not None:
            self._py_error = None
            raise err
        return 0

    @property
    def size(self):
        """Dimension n of the state vector."""
        return self._require().size()

    @property
    def has_jacobian(self):
        """True if the system provides an analytic Jacobian."""
        return self._require().has_jacobian()

    @property
    def nonnegative_mask(self):
        """Boolean array marking physically non-negative components, or None."""
        cdef OdeSystem* p = self._require()
        cdef const unsigned char* m = p.nonnegative_mask()
        if m == NULL:
            return None
        return np.array([m[i] != 0 for i in range(p.size())], dtype=bool)

    def rhs(self, double t, y):
        """f(t, y) as a new array."""
        cdef OdeSystem* p = self._require()
        cdef size_t n = p.size()
        cdef vector[double] yv = _num_vec(y, "y")
        if yv.size() != n:
            raise ValueError(f"y has length {yv.size()}, system has dimension {n}")
        cdef cnp.ndarray out = np.empty(n, dtype=np.float64)
        self._py_error = None
        try:
            p.rhs(t, yv.data(), <double*>cnp.PyArray_DATA(out))
        except Exception:
            self._reraise_callback_error()
            raise
        return out

    def jacobian(self, double t, y):
        """Dense Jacobian df/dy (n x n): analytic if available, else finite differences."""
        cdef OdeSystem* p = self._require()
        cdef size_t n = p.size()
        cdef vector[double] yv = _num_vec(y, "y")
        if yv.size() != n:
            raise ValueError(f"y has length {yv.size()}, system has dimension {n}")
        cdef cnp.ndarray out = np.empty((n, n), dtype=np.float64)
        self._py_error = None
        try:
            nc.evaluate_jacobian(p[0], t, yv.data(), NULL, <double*>cnp.PyArray_DATA(out))
        except Exception:
            self._reraise_callback_error()
            raise
        return out


cdef int _num_cb_rhs(void* ctx, double t, const double* y, double* out, size_t n) noexcept with gil:
    cdef PythonOdeSystem s = <PythonOdeSystem>ctx
    cdef cnp.ndarray yarr
    cdef cnp.ndarray r
    try:
        yarr = np.empty(n, dtype=np.float64)
        memcpy(cnp.PyArray_DATA(yarr), y, n * sizeof(double))
        r = np.ascontiguousarray(s.pyobj.rhs(t, yarr), dtype=np.float64)
        if r.ndim != 1 or <size_t>r.shape[0] != n:
            raise ValueError(f"rhs(t, y) returned shape {np.shape(r)}, expected ({n},)")
        memcpy(out, cnp.PyArray_DATA(r), n * sizeof(double))
        return 0
    except BaseException as exc:
        s._py_error = exc
        return 1


cdef int _num_cb_jac(void* ctx, double t, const double* y, double* out, size_t n) noexcept with gil:
    cdef PythonOdeSystem s = <PythonOdeSystem>ctx
    cdef cnp.ndarray yarr
    cdef cnp.ndarray r
    try:
        yarr = np.empty(n, dtype=np.float64)
        memcpy(cnp.PyArray_DATA(yarr), y, n * sizeof(double))
        r = np.ascontiguousarray(s.pyobj.jacobian(t, yarr), dtype=np.float64)
        if r.ndim != 2 or <size_t>r.shape[0] != n or <size_t>r.shape[1] != n:
            raise ValueError(f"jacobian(t, y) returned shape {np.shape(r)}, expected ({n}, {n})")
        memcpy(out, cnp.PyArray_DATA(r), n * n * sizeof(double))
        return 0
    except BaseException as exc:
        s._py_error = exc
        return 1


cdef class PythonOdeSystem(OdeSystemHandle):
    """C++ system calling back into a Python object with ``rhs(t, y)`` and
    optionally ``jacobian(t, y)``.  Exceptions raised in the callbacks abort
    the integration and are re-raised unchanged."""
    cdef readonly object pyobj

    def __init__(self, obj, Py_ssize_t n, nonnegative_mask=None):
        if n <= 0:
            raise ValueError("dimension n must be positive")
        if not callable(getattr(obj, "rhs", None)):
            raise TypeError("system object must provide a callable rhs(t, y)")
        jac = getattr(obj, "jacobian", None)
        self.pyobj = obj
        cdef nc.ode_callback_fn jac_fn = NULL
        if callable(jac):
            jac_fn = _num_cb_jac
        cdef nc.CallbackSystem* cb = new nc.CallbackSystem(n, <void*>self, _num_cb_rhs, jac_fn)
        self.ptr = cb
        self.owns_ptr = True
        self._releases_gil = False
        cdef vector[unsigned char] mask
        if nonnegative_mask is not None:
            m = np.asarray(nonnegative_mask, dtype=bool).ravel()
            for v in m:
                mask.push_back(1 if v else 0)
            cb.set_nonnegative_mask(mask)


cdef class LinearOdeSystem(OdeSystemHandle):
    """y' = A y with constant matrix A (analytic Jacobian A)."""
    cdef readonly object A

    def __init__(self, A):
        self.A = np.array(A, dtype=np.float64)
        self.ptr = new nc.LinearSystem(_num_matrix(self.A, "A"))
        self.owns_ptr = True


cdef class RobertsonSystem(OdeSystemHandle):
    """Robertson's stiff chemical kinetics problem (analytic Jacobian,
    all components non-negative)."""
    cdef readonly double k1, k2, k3

    def __init__(self, double k1=0.04, double k2=1e4, double k3=3e7):
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.ptr = new nc.RobertsonSystem(k1, k2, k3)
        self.owns_ptr = True


cdef class VanDerPolSystem(OdeSystemHandle):
    """Van der Pol oscillator y1' = y2, y2' = mu((1 - y1^2) y2 - y1) (Hairer-Wanner
    scaling with mu = 1/eps: stiff for large mu, one relaxation cycle per O(1) time)."""
    cdef readonly double mu

    def __init__(self, double mu=1.0):
        self.mu = mu
        self.ptr = new nc.VanDerPolSystem(mu)
        self.owns_ptr = True


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------
cdef nc.ButcherTableau _num_tableau_from_dict(dict d) except *:
    cdef nc.ButcherTableau tab
    A = np.asarray(d["A"], dtype=np.float64)
    s = A.shape[0]
    if A.ndim != 2 or A.shape[1] != s:
        raise ValueError("tableau A must be a square matrix")
    tab.name = str(d.get("name", "custom")).encode()
    tab.s = s
    tab.A = _num_vec(A.ravel(), "A")
    tab.b = _num_vec(d["b"], "b")
    tab.c = _num_vec(d["c"], "c")
    bhat = d.get("bhat")
    if bhat is not None:
        tab.bhat = _num_vec(bhat, "bhat")
        tab.order_hat = int(d["order_hat"])
    tab.order = int(d["order"])
    tab.fsal = bool(d.get("fsal", False))
    return tab


def num_tableau(str name):
    """Built-in Butcher tableau as a dict (A, b, c, bhat, order, order_hat, fsal)."""
    cdef nc.ButcherTableau t = nc.builtin_tableau(name.encode())
    s = <Py_ssize_t>t.s
    return {
        "name": _num_str(t.name),
        "A": _num_from_vec(t.A).reshape(s, s),
        "b": _num_from_vec(t.b),
        "c": _num_from_vec(t.c),
        "bhat": _num_from_vec(t.bhat) if t.bhat.size() > 0 else None,
        "order": t.order,
        "order_hat": t.order_hat if t.bhat.size() > 0 else None,
        "fsal": bool(t.fsal),
        "explicit": bool(t.is_explicit()),
    }


def num_tableau_names():
    return [_num_str(s) for s in nc.builtin_tableau_names()]


def num_integrate(OdeSystemHandle system, str method, double t0, double tf, y0, *,
                  dict tableau=None, double rtol=1e-6, atol=1e-12, double h0=0.0,
                  double h_min=0.0, double h_max=np.inf, double fixed_step=0.0,
                  long max_steps=100000, double max_wall_time=0.0, t_eval=None, bint exact_t_eval=False,
                  bint nonnegative=False, str nonlinear_solver="simplified_newton",
                  str linear_solver="lu_pivot", double newton_tol=0.03, int max_newton_iter=0,
                  double linear_tol=1e-12, int linear_max_iter=0):
    """Runs a C++ integrator; returns a dict with t, y, status, message, stats."""
    cdef OdeSystem* sp = system._require()
    cdef nc.Method m
    if tableau is not None:
        m = nc.method_from_tableau(_num_tableau_from_dict(tableau))
    else:
        m = nc.builtin_method(method.encode())
    cdef nc.Options opt
    opt.rtol = rtol
    opt.atol = _num_vec(np.atleast_1d(np.asarray(atol, dtype=np.float64)), "atol")
    opt.h0 = h0
    opt.h_min = h_min
    opt.h_max = h_max
    opt.fixed_step = fixed_step
    opt.max_steps = max_steps
    opt.max_wall_time = max_wall_time
    if t_eval is not None:
        opt.t_eval = _num_vec(t_eval, "t_eval")
    opt.exact_t_eval = exact_t_eval
    opt.nonnegative = nonnegative
    opt.nonlinear = nc.nonlinear_solver_from_string(nonlinear_solver.encode())
    opt.linear = nc.linear_solver_from_string(linear_solver.encode())
    opt.newton_tol = newton_tol
    opt.max_newton_iter = max_newton_iter
    opt.linear_tol = linear_tol
    opt.linear_max_iter = linear_max_iter
    cdef vector[double] y0v = _num_vec(y0, "y0")
    cdef nc.Result res
    system._py_error = None
    try:
        if system._releases_gil:
            with nogil:
                res = nc.integrate(sp[0], m, t0, tf, y0v, opt)
        else:
            res = nc.integrate(sp[0], m, t0, tf, y0v, opt)
    except Exception:
        system._reraise_callback_error()
        raise
    cdef Py_ssize_t npts = res.t.size()
    cdef nc.Stats st = res.stats
    stats = {
        "n_steps": st.n_steps, "n_accepted": st.n_accepted, "n_rejected": st.n_rejected,
        "n_error_rejections": st.n_error_rejections, "n_solver_rejections": st.n_solver_rejections,
        "n_nonneg_rejections": st.n_nonneg_rejections, "n_rhs": st.n_rhs, "n_jac": st.n_jac,
        "n_jac_fd": st.n_jac_fd, "n_factorizations": st.n_factorizations,
        "n_linear_solves": st.n_linear_solves, "n_linear_iterations": st.n_linear_iterations,
        "n_linear_failures": st.n_linear_failures, "n_newton_iterations": st.n_newton_iterations,
        "n_newton_failures": st.n_newton_failures,
    }
    stats["theta_max"] = st.theta_max
    stats["theta_mean"] = st.theta_sum / st.n_theta if st.n_theta > 0 else 0.0
    stats["h_min"] = st.h_min_used
    stats["h_max"] = st.h_max_used
    stats["h_mean"] = st.h_sum / st.n_accepted if st.n_accepted > 0 else 0.0
    return {
        "t": _num_from_vec(res.t),
        "y": _num_from_vec(res.y).reshape(npts, res.n),
        "status": _num_str(nc.status_to_string(res.status)),
        "message": _num_str(res.message),
        "stats": stats,
        "method": _num_str(m.name),
        "order": m.order,
    }


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------
def num_lu_factor(A, bint pivoting=True, double pivot_tol=1e-12):
    """Returns (L, U, perm, ok, message, min_pivot, growth)."""
    cdef nc.LUFactorization f = nc.lu_factor(_num_matrix(A, "A"), pivoting, pivot_tol)
    packed = _num_from_matrix(f.lu)
    n = packed.shape[0]
    L = np.tril(packed, -1) + np.eye(n)
    U = np.triu(packed)
    perm = np.array([f.perm[i] for i in range(f.perm.size())], dtype=np.int64)
    return L, U, perm, bool(f.ok), _num_str(f.message), f.min_pivot, f.growth


def num_lu_solve(A, b, bint pivoting=True, double pivot_tol=1e-12):
    """Solves A x = b via LU; returns (x or None, message). b may be (n,) or (n, k)."""
    cdef nc.LUFactorization f = nc.lu_factor(_num_matrix(A, "A"), pivoting, pivot_tol)
    if not f.ok:
        return None, _num_str(f.message)
    B = np.array(b, dtype=np.float64, order="F", copy=True)
    vec = B.ndim == 1
    if vec:
        B = B.reshape(-1, 1, order="F")
    if B.shape[0] != f.lu.rows:
        raise ValueError("right-hand side has wrong length")
    cdef vector[double] col
    for j in range(B.shape[1]):
        col = _num_vec(B[:, j], "b")
        nc.lu_solve_inplace(f, col.data())
        B[:, j] = _num_from_vec(col)
    return (B[:, 0] if vec else np.ascontiguousarray(B)), ""


def num_det(A):
    cdef nc.LUFactorization f = nc.lu_factor(_num_matrix(A, "A"), True, 0.0)
    if not f.ok:
        return 0.0
    return nc.lu_determinant(f)


def num_inv(A):
    cdef nc.LUFactorization f = nc.lu_factor(_num_matrix(A, "A"), True, 1e-14)
    if not f.ok:
        return None, _num_str(f.message)
    return _num_from_matrix(nc.lu_inverse(f)), ""


def num_cholesky(A):
    cdef nc.CholeskyFactorization f = nc.cholesky(_num_matrix(A, "A"))
    return (_num_from_matrix(f.L) if f.ok else None), bool(f.ok), _num_str(f.message)


def num_qr_householder(A):
    """Thin QR: (Q (m x n), R (n x n))."""
    cdef nc.QRFactorization f = nc.qr_householder(_num_matrix(A, "A"))
    return _num_from_matrix(nc.qr_q(f)), _num_from_matrix(f.R)


def num_qr_givens(A):
    """Full QR by Givens rotations: (Q (m x m), R (m x n))."""
    cdef nc.Matrix Q
    cdef nc.Matrix R
    nc.qr_givens(_num_matrix(A, "A"), Q, R)
    return _num_from_matrix(Q), _num_from_matrix(R)


def num_lstsq(A, b, str method="householder", double rank_tol=1e-12):
    """Least squares; returns (x or None, residual_norm, ok, message)."""
    cdef nc.LeastSquaresResult r
    if method == "householder":
        r = nc.lstsq_householder(_num_matrix(A, "A"), _num_vec(b, "b"), rank_tol)
    elif method == "givens":
        r = nc.lstsq_givens(_num_matrix(A, "A"), _num_vec(b, "b"), rank_tol)
    else:
        raise ValueError(f"unknown least-squares method {method!r}")
    return (_num_from_vec(r.x) if r.ok else None), r.residual_norm, bool(r.ok), _num_str(r.message)


def num_iterative(str kind, A, b, x0=None, double tol=1e-10, int max_iter=1000):
    """Iterative solve; returns dict(x, iterations, residual, converged, message, history)."""
    cdef nc.Matrix M = _num_matrix(A, "A")
    cdef vector[double] bv = _num_vec(b, "b")
    cdef vector[double] xv
    if x0 is not None:
        xv = _num_vec(x0, "x0")
    cdef nc.IterativeResult r
    if kind == "jacobi":
        r = nc.jacobi(M, bv, xv, tol, max_iter)
    elif kind == "gauss_seidel":
        r = nc.gauss_seidel(M, bv, xv, tol, max_iter)
    elif kind == "cg":
        r = nc.conjugate_gradient(M, bv, xv, tol, max_iter)
    elif kind == "cgnr":
        r = nc.cgnr(M, bv, xv, tol, max_iter)
    else:
        raise ValueError(f"unknown iterative solver {kind!r}")
    return {
        "x": _num_from_vec(r.x),
        "iterations": r.iterations,
        "residual": r.residual,
        "converged": bool(r.converged),
        "message": _num_str(r.message),
        "history": _num_from_vec(r.history),
    }


def num_cond(A, p):
    cdef nc.Matrix M = _num_matrix(A, "A")
    if p == 1:
        return nc.cond_1(M)
    if p == 2:
        return nc.cond_2(M)
    if p == np.inf or p == "inf":
        return nc.cond_inf(M)
    raise ValueError("p must be 1, 2 or inf")


def num_matrix_norm(A, p):
    cdef nc.Matrix M = _num_matrix(A, "A")
    if p == 1:
        return nc.matrix_norm_1(M)
    if p == np.inf or p == "inf":
        return nc.matrix_norm_inf(M)
    if p == "fro":
        return nc.matrix_norm_frobenius(M)
    raise ValueError("p must be 1, inf or 'fro'")


# ---------------------------------------------------------------------------
# eigenvalues
# ---------------------------------------------------------------------------
def num_eigvals(A, bint balance=True):
    """Returns (complex eigenvalues, ok, message, qr_iterations)."""
    cdef nc.EigenResult r = nc.eigenvalues(_num_matrix(A, "A"), balance)
    ev = _num_from_vec(r.re) + 1j * _num_from_vec(r.im)
    return ev, bool(r.ok), _num_str(r.message), r.iterations


def num_hessenberg(A):
    return _num_from_matrix(nc.hessenberg(_num_matrix(A, "A")))


def num_balance(A):
    return _num_from_matrix(nc.balance(_num_matrix(A, "A")))


def num_power_iteration(A, x0=None, double tol=1e-10, int max_iter=1000):
    cdef vector[double] xv
    if x0 is not None:
        xv = _num_vec(x0, "x0")
    cdef nc.PowerIterationResult r = nc.power_iteration(_num_matrix(A, "A"), xv, tol, max_iter)
    return {"eigenvalue": r.eigenvalue, "vector": _num_from_vec(r.eigenvector), "iterations": r.iterations,
            "converged": bool(r.converged), "message": _num_str(r.message)}


def num_inverse_iteration(A, double sigma, x0=None, double tol=1e-10, int max_iter=1000):
    cdef vector[double] xv
    if x0 is not None:
        xv = _num_vec(x0, "x0")
    cdef nc.PowerIterationResult r = nc.inverse_iteration(_num_matrix(A, "A"), sigma, xv, tol, max_iter)
    return {"eigenvalue": r.eigenvalue, "vector": _num_from_vec(r.eigenvector), "iterations": r.iterations,
            "converged": bool(r.converged), "message": _num_str(r.message)}


def num_gershgorin(A):
    cdef vector[nc.GershgorinDisc] d = nc.gershgorin_discs(_num_matrix(A, "A"))
    centers = np.array([d[i].center for i in range(d.size())], dtype=np.float64)
    radii = np.array([d[i].radius for i in range(d.size())], dtype=np.float64)
    return centers, radii
