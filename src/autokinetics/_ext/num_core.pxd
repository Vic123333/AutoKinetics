# Cython declarations of the C++ numerics library (namespace ak::num).
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from autokinetics._ext.ode_system cimport OdeSystem


cdef extern from "autokinetics/numerics/matrix.hpp" namespace "ak::num":
    cdef cppclass Matrix:
        size_t rows
        size_t cols
        vector[double] data
        Matrix()
        Matrix(size_t, size_t, double)

    double matrix_norm_1(const Matrix&)
    double matrix_norm_inf(const Matrix&)
    double matrix_norm_frobenius(const Matrix&)


cdef extern from "autokinetics/numerics/linalg.hpp" namespace "ak::num":
    cdef cppclass LUFactorization:
        Matrix lu
        vector[size_t] perm
        bool pivoting
        int swaps
        bool ok
        string message
        double min_pivot
        double growth

    LUFactorization lu_factor(const Matrix&, bool, double) except +
    void lu_solve_inplace(const LUFactorization&, double*) except +
    double lu_determinant(const LUFactorization&) except +
    Matrix lu_inverse(const LUFactorization&) except +

    cdef cppclass CholeskyFactorization:
        Matrix L
        bool ok
        string message

    CholeskyFactorization cholesky(const Matrix&) except +

    cdef cppclass QRFactorization:
        size_t m
        size_t n
        Matrix V
        Matrix R

    QRFactorization qr_householder(const Matrix&) except +
    Matrix qr_q(const QRFactorization&) except +
    void qr_givens(const Matrix&, Matrix&, Matrix&) except +

    cdef cppclass LeastSquaresResult:
        vector[double] x
        double residual_norm
        bool ok
        string message

    LeastSquaresResult lstsq_householder(const Matrix&, const vector[double]&, double) except +
    LeastSquaresResult lstsq_givens(const Matrix&, const vector[double]&, double) except +

    cdef cppclass IterativeResult:
        vector[double] x
        int iterations
        double residual
        bool converged
        string message
        vector[double] history

    IterativeResult jacobi(const Matrix&, const vector[double]&, const vector[double]&, double, int) except +
    IterativeResult gauss_seidel(const Matrix&, const vector[double]&, const vector[double]&, double, int) except +
    IterativeResult conjugate_gradient(const Matrix&, const vector[double]&, const vector[double]&, double, int) except +
    IterativeResult cgnr(const Matrix&, const vector[double]&, const vector[double]&, double, int) except +

    double cond_1(const Matrix&) except +
    double cond_inf(const Matrix&) except +
    double cond_2(const Matrix&) except +


cdef extern from "autokinetics/numerics/eigen.hpp" namespace "ak::num":
    Matrix balance(const Matrix&) except +
    Matrix hessenberg(const Matrix&) except +

    cdef cppclass EigenResult:
        vector[double] re
        vector[double] im
        bool ok
        int iterations
        string message

    EigenResult eigenvalues(const Matrix&, bool) except +

    cdef cppclass PowerIterationResult:
        double eigenvalue
        vector[double] eigenvector
        int iterations
        bool converged
        string message

    PowerIterationResult power_iteration(const Matrix&, const vector[double]&, double, int) except +
    PowerIterationResult inverse_iteration(const Matrix&, double, const vector[double]&, double, int) except +

    cdef cppclass GershgorinDisc:
        double center
        double radius

    vector[GershgorinDisc] gershgorin_discs(const Matrix&) except +


cdef extern from "autokinetics/numerics/linear_solver.hpp" namespace "ak::num":
    cdef enum class LinearSolverKind "ak::num::LinearSolverKind":
        LUPivot
        LUNoPivot
        QRHouseholder
        GaussSeidel
        Jacobi
        CGNR

    LinearSolverKind linear_solver_from_string(const string&) except +


cdef extern from "autokinetics/numerics/integrator.hpp" namespace "ak::num":
    cdef cppclass ButcherTableau:
        string name
        size_t s
        vector[double] A
        vector[double] b
        vector[double] c
        vector[double] bhat
        int order
        int order_hat
        bool fsal
        bool is_explicit()

    ButcherTableau builtin_tableau(const string&) except +
    vector[string] builtin_tableau_names()

    cdef cppclass Method:
        string name
        int order

    Method builtin_method(const string&) except +
    Method method_from_tableau(const ButcherTableau&) except +

    cdef enum class NonlinearSolverKind "ak::num::NonlinearSolverKind":
        FullNewton
        SimplifiedNewton
        FixedPoint

    NonlinearSolverKind nonlinear_solver_from_string(const string&) except +

    cdef cppclass Options:
        double rtol
        vector[double] atol
        double h0
        double h_min
        double h_max
        double fixed_step
        long max_steps
        double max_wall_time
        vector[double] t_eval
        bool exact_t_eval
        bool nonnegative
        NonlinearSolverKind nonlinear
        LinearSolverKind linear
        double newton_tol
        int max_newton_iter
        double linear_tol
        int linear_max_iter
        double safety
        double facmin
        double facmax

    cdef cppclass Stats:
        long n_steps
        long n_accepted
        long n_rejected
        long n_error_rejections
        long n_solver_rejections
        long n_nonneg_rejections
        long n_rhs
        long n_jac
        long n_jac_fd
        long n_factorizations
        long n_linear_solves
        long n_linear_iterations
        long n_linear_failures
        long n_newton_iterations
        long n_newton_failures
        double theta_max
        double theta_sum
        long n_theta
        double h_min_used
        double h_max_used
        double h_sum

    cdef enum class Status "ak::num::Status":
        Success
        MaxSteps
        WallTime
        StepSizeTooSmall
        NonlinearSolverFailure
        NonFinite

    string status_to_string "ak::num::to_string"(Status)

    cdef cppclass Result:
        vector[double] t
        vector[double] y
        size_t n
        Status status
        string message
        Stats stats
        bool success()

    Result integrate(OdeSystem&, const Method&, double, double, const vector[double]&, const Options&) except + nogil
    long evaluate_jacobian(OdeSystem&, double, const double*, const double*, double*) except +


cdef extern from "autokinetics/numerics/test_systems.hpp" namespace "ak::num":
    ctypedef int (*ode_callback_fn)(void*, double, const double*, double*, size_t) noexcept

    cdef cppclass LinearSystem(OdeSystem):
        LinearSystem(Matrix) except +

    cdef cppclass RobertsonSystem(OdeSystem):
        RobertsonSystem(double, double, double) except +

    cdef cppclass VanDerPolSystem(OdeSystem):
        VanDerPolSystem(double) except +

    cdef cppclass CallbackSystem(OdeSystem):
        CallbackSystem(size_t, void*, ode_callback_fn, ode_callback_fn) except +
        void set_nonnegative_mask(const vector[unsigned char]&) except +
