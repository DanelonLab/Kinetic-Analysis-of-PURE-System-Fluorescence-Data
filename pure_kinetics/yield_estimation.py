"""
yield_estimation.py
-------------------
Provides sliding-window-based yield estimation for fluorescence time courses.

This approach is used instead of reading the raw endpoint value because
fluorescence signals in PURE system experiments may exhibit a gradual decline
after reaching their maximum (e.g., due to evaporation or condensation),
making the endpoint an unreliable measure of true yield. The sliding window
is robust to transient noise spikes, as no single outlier time point can
dominate the average.

Note: This implementation assumes equally spaced time points. Results may
not be meaningful for datasets with missing or irregularly spaced measurements.
"""

import numpy as np


def get_sliding_window_max_yield(fluorescence_values, window_size=20):
    """
    Estimate fluorescence yield as the maximum average within a sliding window.

    Slides a window of `window_size` consecutive time points across the full
    fluorescence time course and returns the highest window-average value found.
    This provides a robust yield estimate that is independent of the sigmoid fit
    and is unaffected by curve fitting failures.

    Parameters
    ----------
    fluorescence_values : array-like
        Fluorescence measurements for a single replicate (RFU).
    window_size : int, optional
        Number of consecutive time points per window (default: 20).
        Assumes equally spaced time points.

    Returns
    -------
    float
        Maximum sliding-window average fluorescence value (RFU).

    Raises
    ------
    ValueError
        If `window_size` exceeds the length of `fluorescence_values`.
    """
    fluorescence_values = np.asarray(fluorescence_values)
    if window_size > len(fluorescence_values):
        raise ValueError(
            f"window_size ({window_size}) exceeds the number of time points "
            f"({len(fluorescence_values)})."
        )

    max_average = -np.inf
    for i in range(len(fluorescence_values) - window_size + 1):
        window_average = np.mean(fluorescence_values[i:i + window_size])
        if window_average > max_average:
            max_average = window_average

    return max_average
