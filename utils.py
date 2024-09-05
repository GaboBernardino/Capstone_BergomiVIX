"""
Gabo Bernardino, Niccolò Fabbri - 09/05/2024

Utility functions for Capstone project
"""

import numpy as np
from scipy.special import gamma


DELTA = 30. / 252.  # number of days


def c_h(hurst: float) -> float:
    """
    Compute the constant C_H for a given Hurst exponent `hurst`
    """
    num = 2 * hurst * gamma(2.5 - hurst)
    den = gamma(hurst + .5) * gamma(2 - 2 * hurst)
    return (num / den) ** .5


def sigma_lognormal(exp_vix: float, exp_vix_squared: float) -> float:
    """
    Compute the *variance* of the lognormal approximation of the VIX
    """
    return - 2 * np.log(exp_vix) + np.log(exp_vix_squared)


def mu_lognormal(exp_vix: float, exp_vix_squared: float) -> float:
    """
    Compute the mean of the lognormal approximation of the VIX
    """
    return np.log(exp_vix) - sigma_lognormal(exp_vix, exp_vix_squared) / 2
