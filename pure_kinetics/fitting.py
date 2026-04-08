"""
fitting.py
----------
Provides functions for generating initial parameter guesses and fitting
the sigmoid model to fluorescence time-course data using a multi-retry
strategy to avoid local minima.
"""

import numpy as np
from scipy.optimize import curve_fit
from .models import sigmoid_model


def generate_initial_guesses(fluorescence_values):
    """
    Generate heuristic initial parameter guesses for sigmoid fitting.

    Estimates are derived from the data itself to provide physically
    reasonable starting points:
      - k_prime : minimum observed fluorescence (basal offset)
      - k       : observed fluorescence range (amplitude)
      - K       : median fluorescence value (half-max time proxy)
      - n       : random draw from Uniform(1, 4) (Hill coefficient)

    Parameters
    ----------
    fluorescence_values : array-like
        Fluorescence measurements for a single replicate.

    Returns
    -------
    list of float
        Initial guesses [k_prime, k, K, n].
    """
    k_prime = np.min(fluorescence_values)
    k = np.max(fluorescence_values) - k_prime
    K = np.median(fluorescence_values)
    n = np.random.uniform(1, 4)
    return [k_prime, k, K, n]


def fit_with_retries(time_points, fluorescence_values, max_retries=120):
    """
    Fit the sigmoid model to fluorescence data with multiple random restarts.

    Retries up to `max_retries` times with freshly randomized initial guesses
    to reduce sensitivity to starting conditions and avoid local minima.
    No explicit parameter bounds are enforced, as fluorescence values in RFU
    can vary substantially between instruments and experimental conditions.

    Parameters
    ----------
    time_points : array-like
        Time points corresponding to fluorescence measurements (hours).
    fluorescence_values : array-like
        Fluorescence measurements for a single replicate (RFU).
    max_retries : int, optional
        Maximum number of fitting attempts (default: 120).

    Returns
    -------
    popt : array or None
        Optimal parameters [k_prime, k, K, n], or None if all attempts failed.
    pcov : array or None
        Estimated covariance of popt, or None if all attempts failed.
    """
    for _ in range(max_retries):
        initial_guesses = generate_initial_guesses(fluorescence_values)
        try:
            popt, pcov = curve_fit(
                sigmoid_model,
                time_points,
                fluorescence_values,
                p0=initial_guesses,
                maxfev=10000
            )
            return popt, pcov
        except RuntimeError:
            pass
    return None, None
