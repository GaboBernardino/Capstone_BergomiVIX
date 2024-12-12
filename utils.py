"""
Gabo Bernardino, Niccolò Fabbri - 09/05/2024

Utility functions for Capstone project
"""

import numpy as np
from scipy.special import gamma, hyp2f1
from scipy.integrate import dblquad, quad
from scipy.stats import norm
import cvxpy as cp

DELTA = 1. / 12.  # number of days


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

    def __add__(self, other):
        if isinstance(other, Hurst):
            return self.h + other.h
        else:
            return self.h + other

    def __mul__(self, other):
        if isinstance(other, Hurst):
            return self.h * other.h
        else:
            return self.h * other

    def __rmul__(self, other):
        return self * other


def c_h(hurst: Hurst) -> float:
    r"""
    Compute the constant $C_H$ for a given Hurst exponent `hurst`
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
    r"""
    Compute the **variance** $\sigma^2$ of the lognormal approximation of the VIX
    """
    return - 2 * np.log(exp_vix) + np.log(exp_vix_squared)


def sigma_lognormal_jim(volvol: float, hurst: Hurst, T: float = 1., delta: float = DELTA) -> float:
    r"""
    Compute the **variance** $\tilde{sigma}^2$ of the lognormal approximation of the VIX,
    using approximation 3.16
    """
    factor = 4 * volvol**2 * c_h(hurst)**2 / (delta**2 * hurst.hp**2)
    def integrand(s):
        inner = (T - s + delta)**hurst.hp - (T - s)**hurst.hp
        return inner**2
    return quad(integrand, 0, T)[0]


def mu_lognormal(exp_vix: float, exp_vix_squared: float) -> float:
    """
    Compute the mean of the lognormal approximation of the VIX
    """
    return np.log(exp_vix) - sigma_lognormal(exp_vix, exp_vix_squared) / 2


def mu_lognormal_jim(exp_vix: float, volvol: float, hurst: Hurst, T: float = 1., delta: float = DELTA) -> float:
    """
    Compute the mean of the lognormal approximation of the VIX
    based on approximation 3.16
    """
    return np.log(exp_vix) - sigma_lognormal_jim(volvol, hurst, T, delta) / 2


def theta(u: float,t: float,
          volvol: float, hurst: Hurst,
          T: float = 1.) -> float:
    out = 2 * volvol**2 * c_h(hurst)**2
    frac1 = (u**hurst.h2 - (u - T)**hurst.h2 + t**hurst.h2 - (t - T)**hurst.h2) / (hurst.h2)
    inner = t**(hurst.hp) * function_F(-t / (u-t), hurst)\
            - (t - T)**(hurst.hp) * function_F((T - t) / (u - t), hurst)
    return out * (frac1 + 2 * ((u - t) ** hurst.hm) * inner / (hurst.hp))


def theta_bar(u: float, t: float,
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


def VIX_price(
        curve: VarianceCurve,
        exp_vix: float,
        exp_vix_squared: float,
        delta: float,
        T: float = 1.
) -> float:
    xi = curve.get_curve()
    integral = quad(xi, T, T + delta)[0]
    sig = sigma_lognormal(exp_vix, exp_vix_squared)
    return np.sqrt(integral / delta) * np.exp(-sig / 8.)


def VIX_price_jim(
        curve: VarianceCurve,
        volvol: float,
        hurst: Hurst,
        delta: float,
        T: float = 1.
) -> float:
    xi = curve.get_curve()
    integral = quad(xi, T, T + delta)[0]
    sig = sigma_lognormal_jim(volvol, hurst, T, delta)
    return np.sqrt(integral / delta) * np.exp(-sig / 8.)


def VIX_price_jim_2(
        curve: VarianceCurve,
        eta: float,
        hurst: Hurst,
        delta: float,
        T: float = 1.
) -> float:
    xi = curve.get_curve()
    integral = quad(xi, T, T + delta)[0]
    convexity = eta**2 * T**hurst.h2 * f_supH(delta / T, hurst)
    return np.sqrt(integral / delta) * np.exp(-convexity / 8.)


def f_supH(theta: float, hurst: Hurst, T: float = 1.) -> float:
    """
    Function defined in Jim's paper for VIX future approximation
    """
    dh = np.sqrt(2 * hurst.h) / (hurst.h + 0.5)
    factor = dh**2 / theta**2
    def integrand(x):
        part1 = (1 + theta - x) ** hurst.hp
        part2 = (1 - x) ** hurst.hp
        return (part1 - part2) ** 2
    return factor * quad(integrand, 0, T)[0]


def cvxpy_exponential(a: float, x: cp.Variable):
    """
    Cvxpy-friendly exponentiation
    """
    return cp.exp(cp.multiply(cp.log(a), x))


def BSFormula(S, K, t, r, vol, callPutFlag: int):
    """Black-Scholes formula for option pricing"""
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    if callPutFlag == 1:  # Call option
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    else:  # Put option
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def stineman_interp(xi: np.array, yi: np.array, x: np.array):
    """
    Perform Stineman interpolation for 1D data.
    """
    n = len(xi)
    if n != len(yi):
        raise ValueError("xi and yi must have the same length")
    
    # Sort the input data
    idx = np.argsort(xi)
    xi = xi[idx]
    yi = yi[idx]
    
    # Compute slopes
    dx = np.diff(xi)
    dy = np.diff(yi)
    m = dy / dx
    
    # Compute slopes at each point
    s = np.zeros(n)
    s[1:-1] = (m[:-1] * dx[1:] + m[1:] * dx[:-1]) / (dx[:-1] + dx[1:])
    s[0] = m[0]
    s[-1] = m[-1]
    
    # Perform the interpolation
    y = np.interp(x, xi, yi)
    for i, xi_val in enumerate(x):
        # Find the interval
        if xi_val <= xi[0]:
            i1 = 0
            i2 = 1
        elif xi_val >= xi[-1]:
            i1 = n - 2
            i2 = n - 1
        else:
            i1 = np.searchsorted(xi, xi_val) - 1
            i2 = i1 + 1
        
        h = xi[i2] - xi[i1]
        t = (xi_val - xi[i1]) / h
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t ** 2 * (3 - 2 * t)
        h11 = t ** 2 * (t - 1)
        y[i] = h00 * yi[i1] + h10 * h * s[i1] + h01 * yi[i2] + h11 * h * s[i2]
    
    return y

