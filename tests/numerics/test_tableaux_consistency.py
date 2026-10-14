import numpy as np
import pytest
from num_problems import Logistic

from autokinetics.numerics import METHODS, ButcherTableau
from autokinetics.numerics.consistency import (
    algebraic_order,
    fit_order,
    lmm_order,
    local_error_study,
    order_conditions,
    root_condition,
    rooted_trees,
)
from autokinetics.numerics.stability import BDF2_ALPHA, BDF2_BETA
from autokinetics.numerics.tableaux import (
    builtin_tableau_names,
    gauss_legendre_tableau,
    lobatto_iiia_tableau,
    radau_iia_tableau,
)


def test_rooted_tree_counts():
    assert [len(rooted_trees(k)) for k in range(1, 8)] == [1, 1, 2, 4, 9, 20, 48]
    conds = order_conditions(np.zeros((1, 1)), [1.0], max_order=4)
    assert [c["tree"] for c in conds[:4]] == ["•", "[•]", "[•,•]", "[[•]]"]
    assert [c["gamma"] for c in conds[:4]] == [1, 2, 3, 6]


@pytest.mark.parametrize("name", builtin_tableau_names())
def test_tableaux_satisfy_order_conditions_exactly(name):
    tab = ButcherTableau.builtin(name)
    res = algebraic_order(tab.A, tab.b, tab.c, max_order=6)
    assert res["order"] == tab.order
    assert res["row_sum_condition"]  # c = A 1 (Lemma 3.3)
    # all conditions up to the order hold to round-off, the next order fails
    for c in res["conditions"]:
        if c["order"] <= tab.order:
            assert abs(c["residual"]) < 1e-14
    assert any(not c["satisfied"] for c in res["conditions"] if c["order"] == tab.order + 1)
    assert METHODS[name].order == tab.order


def test_embedded_dopri_weights():
    tab = ButcherTableau.builtin("dopri5")
    assert algebraic_order(tab.A, tab.bhat, max_order=6)["order"] == 4
    assert tab.fsal and np.allclose(tab.A[-1], tab.b)
    heun = ButcherTableau.builtin("heun")
    assert algebraic_order(heun.A, heun.bhat)["order"] == 1


@pytest.mark.parametrize(
    ("generated", "builtin"),
    [
        (gauss_legendre_tableau(1), "implicit_midpoint"),
        (gauss_legendre_tableau(2), "gauss4"),
        (radau_iia_tableau(1), "implicit_euler"),
        (radau_iia_tableau(2), "radau3"),
        (radau_iia_tableau(3), "radau5"),
        (lobatto_iiia_tableau(2), "trapezoidal"),
    ],
)
def test_collocation_tableaux_match_closed_forms(generated, builtin):
    tab = ButcherTableau.builtin(builtin)
    np.testing.assert_allclose(generated.A, tab.A, atol=1e-14)
    np.testing.assert_allclose(generated.b, tab.b, atol=1e-14)
    np.testing.assert_allclose(generated.c, tab.c, atol=1e-14)
    assert algebraic_order(generated.A, generated.b, max_order=6)["order"] == tab.order


def test_closed_form_values_from_script():
    g = ButcherTableau.builtin("gauss4")
    r = np.sqrt(3) / 6
    np.testing.assert_allclose(g.A, [[0.25, 0.25 - r], [0.25 + r, 0.25]])
    r5 = ButcherTableau.builtin("radau5")
    np.testing.assert_allclose(r5.c, [(4 - np.sqrt(6)) / 10, (4 + np.sqrt(6)) / 10, 1.0])
    np.testing.assert_allclose(
        r5.A[0], [0.19681547722366, -0.06553542585020, 0.02377097434822], atol=1e-13
    )
    assert r5.stiffly_accurate
    # higher collocation methods have the predicted orders
    assert algebraic_order(*_ab(gauss_legendre_tableau(3)), max_order=7)["order"] == 6
    assert algebraic_order(*_ab(lobatto_iiia_tableau(3)), max_order=6)["order"] == 4


def _ab(t):
    return t.A, t.b


def test_bdf2_multistep_order_and_zero_stability():
    res = lmm_order(BDF2_ALPHA, BDF2_BETA)
    assert res["order"] == 2
    assert res["error_constant"] == pytest.approx(-1.0 / 3.0)  # C_3 / (3! sigma(1)), normalised
    assert root_condition(BDF2_ALPHA)["satisfied"]
    # Example 7.13: consistent of order 3 but violates the root condition
    alpha = np.array([-5.0, 4.0, 1.0])
    beta = np.array([2.0, 4.0, 0.0])
    assert lmm_order(alpha, beta)["order"] == 3
    assert not root_condition(alpha)["satisfied"]


def test_fit_order_excludes_noise_and_instability():
    h = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    e = 3.0 * h**2
    e[0] = np.inf
    e[-1] = 1e-20
    s = fit_order(h, e, floor=1e-15)
    assert s.slope == pytest.approx(2.0)
    assert list(s.used) == [False, True, True, True, False]
    assert len(s.reason_excluded) == 2


@pytest.mark.parametrize(
    "method",
    [
        "euler",
        "heun",
        "rk4",
        "dopri5",
        "implicit_euler",
        "trapezoidal",
        "gauss4",
        "radau3",
        "radau5",
        "sdirk2",
        "sdirk3",
        "ros2",
    ],
)
def test_empirical_consistency_order(method):
    sys = Logistic()
    p = METHODS[method].order
    hs = (0.1 if p <= 3 else 0.8) * 0.5 ** np.arange(5)
    study = local_error_study(sys, 0.0, [sys.y0], method, hs, exact=sys.exact)
    assert study.slope - 1.0 == pytest.approx(p, abs=0.3)
