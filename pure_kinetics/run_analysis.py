"""
run_analysis.py
---------------
Example script showing how to use the pure_kinetics library to run a full
kinetic analysis of PURE system fluorescence data.

Usage
-----
1. Edit the paths and settings in the configuration block below.
2. Run from your terminal:
       python run_analysis.py
"""

import os
import numpy as np
from pure_kinetics import (
    load_fluorescence_data,
    fit_with_retries,
    get_sliding_window_max_yield,
    calculate_plateau_time,
    calculate_translation_rate,
    plot_condition,
    save_parameters_to_excel,
)

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE  = "path/to/your/input.xlsx"
OUTPUT_FILE = "path/to/save/Kinetic_Parameters.xlsx"
SAVE_DIR    = "path/to/save/plots"
mCherry     = False   # Set to True if analyzing mCherry fluorescence
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

# Load and organize data
time_points, grouped_data = load_fluorescence_data(INPUT_FILE)
max_time = max(time_points)

all_parameters = []

for condition_name, replicate_data in grouped_data.items():

    # Determine fluorescent protein from condition name
    if mCherry:
        fluorescent_protein = "mCherry"
    elif "T7" in condition_name:
        fluorescent_protein = "mVenus"
    else:
        fluorescent_protein = "eYFP"

    fitted_params = {}

    for replicate_name, fluorescence_values in replicate_data.items():

        # Estimate yield independently of the sigmoid fit
        max_yield = get_sliding_window_max_yield(fluorescence_values)

        # Fit sigmoid model
        popt, pcov = fit_with_retries(time_points, fluorescence_values)

        if popt is not None:
            k_prime, k, K, n = popt
            plateau_time     = calculate_plateau_time(K, n, max_time)
            translation_rate = calculate_translation_rate(k, n, K)
            fitted_params[replicate_name] = popt

            print(f"{condition_name} | {replicate_name}")
            print(f"  k'={k_prime:.3f}  k={k:.3f}  K={K:.3f}  n={n:.3f}")
            print(f"  Yield={max_yield:.3f}  Plateau={plateau_time:.3f}h  "
                  f"TranslationRate={translation_rate}")

            all_parameters.append([
                condition_name, replicate_name,
                k_prime, k, K, n,
                max_yield, plateau_time, translation_rate
            ])
        else:
            print(f"WARNING: Fitting failed for {condition_name} | {replicate_name}. "
                  "Raw data will be plotted without a fitted curve.")
            fitted_params[replicate_name] = None

    # Generate and save overlay plot for this condition
    plot_condition(
        time_points=time_points,
        replicate_data=replicate_data,
        fitted_params=fitted_params,
        condition_name=condition_name,
        fluorescent_protein=fluorescent_protein,
        save_dir=SAVE_DIR,
    )

# Export all parameters to Excel
save_parameters_to_excel(all_parameters, OUTPUT_FILE)
print(f"\nFitted parameters saved to {OUTPUT_FILE}")
