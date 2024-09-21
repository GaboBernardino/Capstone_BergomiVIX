"""
Gabo Bernardino, Niccolò Fabbri - 09/05/2024

Utility functions for Capstone project
"""

import numpy as np
from scipy.special import gamma, hyp2f1
from scipy.integrate import dblquad, quad


DELTA = 30. / 252.  # number of days


def c_h(hurst: float) -> float:
    """
    Compute the constant C_H for a given Hurst exponent `hurst`
    """
    num = 2 * hurst * gamma(2.5 - hurst)
    den = gamma(hurst + .5) * gamma(2 - 2 * hurst)
    return (num / den) ** .5


def function_F(u: float, hurst: float) -> float:
    """
    Function (2.6) with hypergeometric
    """
    if u > 0:
        u *= -1
    return hyp2f1(-(hurst - 0.5), hurst + 0.5, 1.5 + hurst, u)


def sigma_lognormal(exp_vix: float, exp_vix_squared: float) -> float:
    """
    Compute the **variance** of the lognormal approximation of the VIX
    """
    return - 2 * np.log(exp_vix) + np.log(exp_vix_squared)


def mu_lognormal(exp_vix: float, exp_vix_squared: float) -> float:
    """
    Compute the mean of the lognormal approximation of the VIX
    """
    return np.log(exp_vix) - sigma_lognormal(exp_vix, exp_vix_squared) / 2


def theta(u: float,t: float,
          volvol: float, hurst: float,
          T: float = 1.) -> float:
    out = 2 * volvol**2 * c_h(hurst)**2
    h2 = 2 * hurst
    frac1 = (u**h2 - (u - T)**h2 + t**h2 - (t - T)**h2) / (h2)
    inner = t**(hurst+.5) * function_F(-t / (u-t), hurst)\
            - (t - T)**(hurst+.5) * function_F((T - t) / (u - t), hurst)
    return out * (frac1 + 2 * (u - t) ** (hurst-.5) / (hurst+0.5) * inner)


def theta_bar(u: float,t: float,
              volvol: float, hurst: float,
              T: float = 1.) -> float:
    return 0 if u == t else theta(max(u, t), min(u, t), volvol, hurst, T)


class VarianceCurve(object):
    """
    Class for a function object representing
    the forward variance curve xi_0(t)
    (basically a wrapper around a simple function of time t)
    """
    def __init__(self, fun):
        self._curve_function = fun

    def __call__(self, t):
        return self._curve_function(t)

    def get_curve(self):
        return self._curve_function


def expected_vix_sq(curve: VarianceCurve, delta: float = DELTA, T: float = 1.):
    xi = curve.get_curve()
    return quad(xi, T, delta)[0]


def moment2_vix_sq(curve: VarianceCurve, delta: float,
                   volvol: float, hurst: float, T: float = 1) -> float:
    xi = curve.get_curve()
    def integrand(u, t):
        h2 = 2 * hurst
        acf = (u - T) ** h2 + (t - T) ** h2 - u ** h2 - t ** h2
        exp = np.exp(volvol**2 * c_h(hurst)**2 * (acf) / hurst)
        exp_theta = np.exp(theta_bar(u, t, volvol, hurst, T))
        return xi(u) * xi(t) * exp * exp_theta
    return dblquad(integrand, T, delta, T, delta)[0]
