"""
models.py
---------
Defines the sigmoid model used to fit fluorescence kinetics and functions
to derive kinetic parameters from fitted model coefficients.
"""

import numpy as np


def sigmoid_model(t, k_prime, k, K, n):
    """
    Four-parameter sigmoid function for modeling fluorescence kinetics.

    Parameters
    ----------
    t : array-like
        Time points (hours).
    k_prime : float
        Basal fluorescence level (offset).
    k : float
        Maximal increase in fluorescence above k_prime.
    K : float
        Half-maximal time (hours).
    n : float
        Hill coefficient (controls steepness of the sigmoid).

    Returns
    -------
    array-like
        Modeled fluorescence values at each time point.

    References
    ----------
    Karzbrun et al. (2011) https://doi.org/10.1038/msb.2009.50
    Blanken et al. (2019), Doerr et al. (2019)
    """
    return k_prime + (k * t**n) / (t**n + K**n)


def calculate_plateau_time(K, n, max_time):
    """
    Estimate the time at which the reaction reaches its fluorescence plateau.

    The plateau time is estimated from the sigmoid parameters as:
        plateau_time = K + (2 * K) / n

    If the estimated plateau time exceeds the observation window, 0 is returned
    to indicate that no plateau was reached within the measurement period.

    Parameters
    ----------
    K : float
        Half-maximal time from the sigmoid fit (hours).
    n : float
        Hill coefficient from the sigmoid fit.
    max_time : float
        Duration of the experiment (hours).

    Returns
    -------
    float
        Estimated plateau time in hours, or 0 if beyond the observation window.
    """
    plateau_time = K + (2 * K) / n
    return plateau_time if plateau_time <= max_time else 0


def calculate_translation_rate(k, n, K):
    """
    Estimate the translation rate as a heuristic kinetic metric.

    Calculated as the maximum slope of the sigmoid at its inflection point:
        translation_rate = (k * n) / (4 * K)

    This is intended for relative comparison between conditions, not as an
    absolute biochemical rate. Non-positive values are returned as NaN to
    prevent propagation of non-physical estimates into downstream analyses.

    Parameters
    ----------
    k : float
        Amplitude of fluorescence increase from the sigmoid fit (RFU).
    n : float
        Hill coefficient from the sigmoid fit.
    K : float
        Half-maximal time from the sigmoid fit (hours).

    Returns
    -------
    float
        Estimated translation rate in RFU/hour, or np.nan if non-physical.
    """
    translation_rate = (k * n) / (4 * K)
    return translation_rate if translation_rate > 0 else np.nan
