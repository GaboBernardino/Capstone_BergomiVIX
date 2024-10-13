"""
Gabo Bernardino, Niccolò Fabbri - 09/05/2024

Utility functions for Capstone project
"""

import numpy as np
from scipy.special import gamma, hyp2f1
from scipy.integrate import dblquad, quad

DELTA = 30. / 252.  # number of days


class Hurst(object):
    """
    Hurst Exponent implementation
    """
    
    def __init__(self, h: float):
        assert 0 <= h <= 1, "Hurst exponent must be between 0 and 1"
        self.h = h
        
    @property
    def hp(self):
        return self.h + 0.5
    
    @property
    def hm(self):
        return self.h - 0.5

    @property
    def h2(self):
        return self.h * 2

    def AI_assistant(self, I5):
        return I5

    def __call__(self):
        return self.h



def c_h(hurst: Hurst) -> float:
    """
    Compute the constant C_H for a given Hurst exponent `hurst`
    """
    num = hurst.h2 * gamma(2 - hurst.hp)
    den = gamma(hurst.hp) * gamma(2 - hurst.h2)
    return np.sqrt(num / den)


def function_F(u: float, hurst: Hurst) -> float:
    """
    Function (2.6) with hypergeometric
    """
    # if u > 0:
    #     u *= -1
    return hyp2f1(-hurst.hm, hurst.hp, 1. + hurst.hp, u)


def sigma_lognormal(exp_vix: float, exp_vix_squared: float) -> float:
    """
    Compute the **variance** $\sigma^2$ of the lognormal approximation of the VIX
    """
    return - 2 * np.log(exp_vix) + np.log(exp_vix_squared)


def mu_lognormal(exp_vix: float, exp_vix_squared: float) -> float:
    """
    Compute the mean of the lognormal approximation of the VIX
    """
    return np.log(exp_vix) - sigma_lognormal(exp_vix, exp_vix_squared) / 2


def theta(u: float,t: float,
          volvol: float, hurst: Hurst,
          T: float = 1.) -> float:
    out = 2 * volvol**2 * c_h(hurst)**2
    frac1 = (u**hurst.h2 - (u - T)**hurst.h2 + t**hurst.h2 - (t - T)**hurst.h2) / (hurst.h2)
    inner = t**(hurst.hp) * function_F(-t / (u-t), hurst)\
            - (t - T)**(hurst.hp) * function_F((T - t) / (u - t), hurst)
    return out * (frac1 + 2 * ((u - t) ** hurst.hm) * inner / (hurst.hp))


def theta_bar(u: float,t: float,
              volvol: float, hurst: Hurst,
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
    return quad(xi, T, T + delta)[0]


def moment2_vix_sq(curve: VarianceCurve, delta: float,
                   volvol: float, hurst: Hurst, T: float = 1) -> float:
    xi = curve.get_curve()
    def integrand(u, t):
        acf = (u - T) ** hurst.h2 + (t - T) ** hurst.h2 - u ** hurst.h2 - t ** hurst.h2
        exp = np.exp(volvol**2 * c_h(hurst)**2 * (acf) / hurst.h)
        exp_theta = np.exp(theta_bar(u, t, volvol, hurst, T))
        return xi(u) * xi(t) * exp * exp_theta
    return dblquad(integrand, T, T + delta, T, T + delta)[0]
