"""
Gabo Bernardino, Niccolò Fabbri

Functions for model calibration and parameter estimation
"""


import pandas as pd
import numpy as np
import utils
from scipy.integrate import quad
from typing import Tuple


# Integrand functions for calls and puts
def cTilde(y: float, interp_func, texp: float, multiple: int):
    K = np.exp(y)
    vol = interp_func(y)
    price = utils.BSFormula(S=1., K=K, t=texp, r=0, vol=vol, callPutFlag=1)
    return np.exp(y * multiple) * price


def pTilde(y: float, interp_func, texp: float, multiple: int):
    K = np.exp(y)
    vol = interp_func(y)
    price = utils.BSFormula(S=1., K=K, t=texp, r=0, vol=vol, callPutFlag=0)
    return np.exp(y * multiple) * price


def get_power_strip(
        df: pd.DataFrame,
        texp: float,
        degree: int
) -> Tuple[float, float]:
    """
    Get necessary data for computing the strip replication of
    a power payoff

    Parameters
    ----------
    df: pd.DataFrame
        Implied vol data for the expiration in question
    texp: float
        Time to maturity in years
    degree: int
        Degree of the power payoff
    """
    midVol = 0.5 * (df['Ask'] + df['Bid'])
    fwd = df['Fwd'].values[0]  # Forward price
    strikes = df['Strike']
    k = np.log(strikes / fwd).values  # Log-moneyness
    kmin, kmax = min(k), max(k)
    minvo, maxvo = midVol[k == kmin].values[0], midVol[k == kmax].values[0]

    def volInterp(kout):
        """
        Interpolate vol between strikes;
        set constant outside strike range, use Stineman inside
        """
        if not isinstance(kout, np.ndarray):
            kout = np.array([kout])
        return np.where(
            kout < kmin, minvo,
            np.where(
                kout > kmax, maxvo,
                utils.stineman_interp(k, midVol.values, kout)
            )
        )

    # Compute the integrals
    callIntegral, _ = quad(
        cTilde, 0, 10, args=(volInterp, texp, degree-1)
    )
    putIntegral, _ = quad(
        pTilde, -10, 0, args=(volInterp, texp, degree-1)
    )
    return callIntegral, putIntegral


def vix_power(ivol_data: pd.DataFrame, expiry: int, degree: int) -> float:
    r"""
    Estimate expectation of a VIX power via strip replication

    Parameters
    ----------
    ivol_data: pd.DataFrame
        Implied vol data as provided by OptionMetrics
    expiry: int
        Number representing the expiration date (as YYYYMMDD)
    degree: int
        Degree of the power payoff

    Returns
    -------
    float: estimation of $\mathbb{E}[\text{VIX}^p_T]$
    """
    df = ivol_data[ivol_data['Expiry'] == expiry]
    texp = df['Texp'].values[0]
    mask = ~df['Bid'].isna()
    df = df.loc[mask]

    # compute integrals for strip
    call_integral, put_integral = get_power_strip(df, texp, degree)
    # Calculate the result
    fwd = df['Fwd'].values[0]  # Forward price
    res = fwd ** degree * (1 + degree * (degree-1) * (call_integral + put_integral))
    return res


# functions to compute the convexity adjustment based on model or market
def sigma_market(ivol_data: pd.DataFrame, expiry: int) -> np.array:
    r"""$\sigma^2$ extracted from the market, based on [i dont remember lol]"""
    exp_vix2_mkt = vix_power(ivol_data, expiry, 2) * utils.DELTA / 10**4
    exp_vix4_mkt = vix_power(ivol_data, expiry, 4) * utils.DELTA**2 / 10**8
    return utils.sigma_lognormal(exp_vix2_mkt, exp_vix4_mkt)


def sigma_jim(texp: float, eta: float, hurst: utils.Hurst):
    r"""$\sigma^2$ estimated with Jim's formula"""
    return (eta**2) * (texp**hurst.h2) * utils.f_supH(utils.DELTA / texp, hurst)


def sigma_jim_cvxpy(texp: float, eta: float, hurst: float):
    r"""$\sigma^2$ estimated with Jim's formula"""
    return (eta**2) * utils.cvxpy_exponential(texp, (2*hurst)) * utils.f_supH_floats(utils.DELTA / texp, hurst)
