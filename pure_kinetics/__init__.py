"""
pure_kinetics
=============
A Python library for kinetic analysis of PURE system fluorescence data.

Modules
-------
models          : Sigmoid model definition and derived parameter calculations
fitting         : Curve fitting with multi-retry strategy
yield_estimation: Sliding-window yield estimation
plotting        : Visualization of raw and fitted curves
io              : Data loading and Excel export
"""

from .models import sigmoid_model, calculate_plateau_time, calculate_translation_rate
from .fitting import generate_initial_guesses, fit_with_retries
from .yield_estimation import get_sliding_window_max_yield
from .plotting import plot_condition
from .io import load_fluorescence_data, save_parameters_to_excel

__version__ = "1.0.0"
__all__ = [
    "sigmoid_model",
    "calculate_plateau_time",
    "calculate_translation_rate",
    "generate_initial_guesses",
    "fit_with_retries",
    "get_sliding_window_max_yield",
    "plot_condition",
    "load_fluorescence_data",
    "save_parameters_to_excel",
]
