"""
plotting.py
-----------
Visualization functions for PURE system fluorescence kinetics.

Generates overlay plots of raw fluorescence data and sigmoid fits for each
condition. These plots serve as the primary quality-control step: visual
inspection allows identification of low-confidence fits and conditions where
extracted kinetic parameters may not be meaningful.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from .models import sigmoid_model


# Predefined color schemes per fluorescent protein
_COLOR_SCHEMES = {
    'mVenus':  ['#adf719', '#91d10f', '#78a60c'],
    'eYFP':    ['#e6fa0c', '#f0db0c', '#f5b90c'],
    'mCherry': ['#E02B48', '#C02E3D', '#9F2A35'],
}
_DEFAULT_COLORS = ['#000000', '#333333', '#666666']


def clean_condition_name(condition_name):
    """
    Remove characters that are illegal in filenames from a condition name.

    Parameters
    ----------
    condition_name : str
        Raw condition name (may include special characters).

    Returns
    -------
    str
        Sanitized condition name safe for use as a filename.
    """
    return re.sub(r'[<>:"/\\|?*]', '', condition_name)


def get_replicate_colors(fluorescent_protein):
    """
    Return the color palette for a given fluorescent protein.

    Parameters
    ----------
    fluorescent_protein : str
        One of 'mVenus', 'eYFP', or 'mCherry'.

    Returns
    -------
    list of str
        List of hex color codes for plotting replicates.
    """
    return _COLOR_SCHEMES.get(fluorescent_protein, _DEFAULT_COLORS)


def plot_condition(
    time_points,
    replicate_data,
    fitted_params,
    condition_name,
    fluorescent_protein,
    save_dir
):
    """
    Generate and save an overlay plot of raw data and sigmoid fits for one condition.

    Raw replicate traces are plotted in the color scheme of the fluorescent protein.
    Sigmoid fits are overlaid in black. If fitting failed for a replicate (i.e., its
    entry in `fitted_params` is None), only the raw data is plotted for that replicate.
    Visual inspection of these plots is the primary quality-control step.

    Parameters
    ----------
    time_points : array-like
        Time points (hours).
    replicate_data : dict
        Mapping of replicate name -> fluorescence array (RFU).
    fitted_params : dict
        Mapping of replicate name -> popt array or None if fitting failed.
    condition_name : str
        Condition label used as the plot title and filename.
    fluorescent_protein : str
        One of 'mVenus', 'eYFP', or 'mCherry'.
    save_dir : str
        Directory where the PNG plot will be saved.

    Returns
    -------
    str
        Full path to the saved PNG file.
    """
    colors = get_replicate_colors(fluorescent_protein)
    fit_lines = []

    plt.figure()
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"

    for idx, (replicate_name, fluorescence_values) in enumerate(replicate_data.items()):
        color = colors[idx % len(colors)]
        plt.plot(time_points, fluorescence_values, linewidth=8,
                 label=f'{replicate_name} (Data)', color=color)

        popt = fitted_params.get(replicate_name)
        if popt is not None:
            fitted_values = sigmoid_model(np.asarray(time_points), *popt)
            fit_lines.append(fitted_values)

    for fitted_values in fit_lines:
        plt.plot(time_points, fitted_values, 'k-', linewidth=3)

    plt.plot([], [], 'k-', linewidth=1, label='Fit')
    plt.xlabel('Time (hours)', size=14)
    plt.ylabel(f'{fluorescent_protein} Fluorescence (RFU)', size=14)
    plt.title(clean_condition_name(condition_name), size=20)
    plt.grid(False)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{clean_condition_name(condition_name)}.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path
