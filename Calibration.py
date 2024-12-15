"""
Gabo Bernardino, Niccolò Fabbri

Functions for model calibration and parameter estimation
"""


import pandas as pd
import numpy as np
from scipy.optimize import minimize
import utils
from scipy.integrate import quad
from typing import Tuple


# Functions for fitting SVI
def raw_svi(
    k: float, a: float, b: float, rho: float, m: float, sig: float
) -> float:
    """Return the total variance in the raw SVI parametrization"""
    root = (k-m)**2 + sig**2
    fac = rho * (k - m) + np.sqrt(root)
    return a + b * fac

svi_bounds = [
    (None, None), (0, None), (-1, 1), (None, None), (0, None)
]

init_guess = (.04, .4, .1, -.4, .1)

def svi_constraint(
    k: float, a: float, b: float, rho: float, m: float, sig: float
):
    """
    To ensure total variance is non-negative;
    compatible with the API of scipy.optimize.minimize
    """
    return a + b * sig * np.sqrt(1 - rho**2)

def fit_svi(k: float, mids: float, texp: float) -> float:
    """
    Given a volatility smile, fit an SVI parametrization
    """
    w = mids**2 * texp
    
    obj = lambda x: np.sum((w - raw_svi(k, *x))**2)
    constraint = {'type': 'ineq', 'fun': lambda x: svi_constraint(*x)}
    
    opt = minimize(
        obj, init_guess, method='L-BFGS-B',
        constraints=constraint, bounds=svi_bounds,
    )
    if opt.success:
        params = opt.x
        svi_w = raw_svi(k, *params)
        return params, svi_w
    else:
        return None, None


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


def get_power_strip_svi(
        df: pd.DataFrame,
        texp: float,
        degree: int
) -> Tuple[float, float]:
    """
    Get necessary data for computing the strip replication of
    a power payoff; fit a simple SVI to extrapolate
    outside observed moneyness level

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
    
    svi_params, _ = fit_svi(k, midVol.values, texp)
    
    smile = lambda x: raw_svi(x, *svi_params)

    # Compute the integrals
    callIntegral, _ = quad(
        cTilde, 0, 10, args=(smile, texp, degree-1)
    )
    putIntegral, _ = quad(
        pTilde, -10, 0, args=(smile, texp, degree-1)
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


def vix_power_svi(ivol_data: pd.DataFrame, expiry: int, degree: int) -> float:
    r"""
    Estimate expectation of a VIX power via strip replication;
    fit a simple SVI to extrapolate outside observed moneyness level

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
    call_integral, put_integral = get_power_strip_svi(df, texp, degree)
    # Calculate the result
    fwd = df['Fwd'].values[0]  # Forward price
    res = fwd ** degree * (1 + degree * (degree-1) * (call_integral + put_integral))
    return res


# functions to compute the convexity adjustment based on model or market
def sigma_market(ivol_data: pd.DataFrame, expiry: int, use_svi: bool = False) -> np.array:
    r"""$\sigma^2$ extracted from the market, based on [i dont remember lol]"""
    power_func = vix_power_svi if use_svi else vix_power
    exp_vix2_mkt = power_func(ivol_data, expiry, 2) * utils.DELTA / 10**4
    exp_vix4_mkt = power_func(ivol_data, expiry, 4) * utils.DELTA**2 / 10**8
    return utils.sigma_lognormal(exp_vix2_mkt, exp_vix4_mkt)


def sigma_jim(texp: float, eta: float, hurst: utils.Hurst):
    r"""$\sigma^2$ estimated with Jim's formula"""
    return (eta**2) * (texp**hurst.h2) * utils.f_supH(utils.DELTA / texp, hurst)


def sigma_jim_cvxpy(texp: float, eta: float, hurst: float):
    r"""$\sigma^2$ estimated with Jim's formula"""
    return (eta**2) * utils.cvxpy_exponential(texp, (2*hurst)) * utils.f_supH_floats(utils.DELTA / texp, hurst)
