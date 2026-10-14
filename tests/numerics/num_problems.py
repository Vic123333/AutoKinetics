"""Test problems with known solutions (shared by the numerics tests)."""

import numpy as np


class Logistic:
    """y' = y (1 - y), y(0) = y0: y(t) = 1 / (1 + (1/y0 - 1) e^{-t})."""

    y0 = 0.1

    def rhs(self, t, y):
        return y * (1.0 - y)

    def jacobian(self, t, y):
        return np.array([[1.0 - 2.0 * y[0]]])

    def exact(self, t):
        return np.array([1.0 / (1.0 + (1.0 / self.y0 - 1.0) * np.exp(-t))])


class Forced:
    """Non-autonomous y' = -y + cos t, y(0) = 1: y(t) = (cos t + sin t)/2 + e^{-t}/2."""

    def rhs(self, t, y):
        return -y + np.cos(t)

    def exact(self, t):
        return np.array([0.5 * (np.cos(t) + np.sin(t)) + 0.5 * np.exp(-t)])


class Oscillator:
    def rhs(self, t, y):
        return np.array([y[1], -y[0]])

    def jacobian(self, t, y):
        return np.array([[0.0, 1.0], [-1.0, 0.0]])
