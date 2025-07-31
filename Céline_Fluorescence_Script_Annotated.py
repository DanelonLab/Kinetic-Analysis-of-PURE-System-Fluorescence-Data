# --- Import required libraries ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
from openpyxl import load_workbook


# --- Define the sigmoid model used to fit fluorescence data ---
def sigmoid_model(t, k_prime, k, K, n):
    """
    Sigmoid function used to model fluorescence kinetics.
    Parameters:
        t       : Time (hours)
        k_prime : Basal fluorescence level (offset)
        k       : Maximal increase above k_prime
        K       : Time at half-maximal fluorescence
        n       : Hill coefficient (steepness)
    """
    return k_prime + (k * t**n) / (t**n + K**n)


# --- Estimate time to reach fluorescence plateau ---
def calculate_plateau_time(K, n, max_time):
    """
    Estimate the time when the reaction plateaus based on sigmoid parameters.
    """
    plateau_time = ((2 * K) / n) + K
    return plateau_time


# --- Estimate translation rate from fit parameters ---
def calculate_translation_rate(k, n, K, max_yield):
    """
    Calculate translation rate from the curve parameters.
    """
    translation_rate = (k * n) / (4 * K)
    return translation_rate if translation_rate > 0 else np.nan


# --- Clean up names for safe file saving ---
def clean_condition_name(condition_name):
    """
    Remove illegal characters from condition names for safe file saving.
    """
    return re.sub(r'[<>:"/\\|?*]', '', condition_name)


# --- Generate initial guesses for curve fitting ---
def generate_initial_guesses(fluorescence_values):
    """
    Heuristically generate initial guesses for the sigmoid fitting algorithm.
    """
    k_prime = min(fluorescence_values)
    k = max(fluorescence_values) - k_prime
    K = np.median(fluorescence_values)
    n = np.random.uniform(1, 4)
    return [k_prime, k, K, n]


# --- Try fitting curve with multiple retries to avoid local minima ---
def fit_with_retries(time_points, fluorescence_values, max_retries=120):
    """
    Fit sigmoid curve to data, retrying with different guesses if needed.
    """
    for _ in range(max_retries):
        initial_guesses = generate_initial_guesses(fluorescence_values)
        try:
            popt, pcov = curve_fit(sigmoid_model, time_points, fluorescence_values,
                                   p0=initial_guesses, maxfev=10000)
            return popt, pcov
        except RuntimeError:
            pass
    return None, None


# --- Assign colors to replicates based on fluorescent protein ---
def get_replicate_colors(fluorescent_protein):
    """
    Return predefined color schemes depending on which FP is used.
    """
    color_schemes = {
        'mVenus': ['#adf719', '#91d10f', '#78a60c'],
        'eYFP': ['#e6fa0c', '#f0db0c', '#f5b90c'],
        'mCherry': ['#E02B48', '#C02E3D', '#9F2A35']
    }
    return color_schemes.get(fluorescent_protein, ['#000000', '#333333', '#666666'])  # default: grayscale


# --- Calculate maximum average signal in a sliding window ---
def get_sliding_window_max_yield(fluorescence_values, window_size=20):
    """
    Return the highest average value found within a sliding window of the signal.
    Useful for estimating the yield more robustly.
    """
    max_average = -np.inf
    best_window = None

    for i in range(len(fluorescence_values) - window_size + 1):
        window = fluorescence_values[i:i + window_size]
        window_average = np.mean(window)
        if window_average > max_average:
            max_average = window_average
            best_window = window

    return max_average


# --- Fit and plot replicates for a given condition ---
def fit_and_plot_replicates(time_points, replicate_data, save_dir, condition_name, mCherry):
    """
    Fit all replicates of a condition and generate plots with raw and fitted data.
    Returns list of parameters for each replicate.
    """
    fluorescent_protein = "mCherry" if mCherry else ("mVenus" if "T7" in condition_name else "eYFP")
    colors = get_replicate_colors(fluorescent_protein)

    plt.figure()
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"

    all_parameters = []
    fit_lines = []
    max_time = max(time_points)

    for idx, (replicate_name, fluorescence_values) in enumerate(replicate_data.items()):
        color = colors[idx % len(colors)]
        max_yield = get_sliding_window_max_yield(fluorescence_values)
        popt, pcov = fit_with_retries(time_points, fluorescence_values)

        if popt is not None:
            k_prime, k, K, n = popt
            plateau_time = calculate_plateau_time(K, n, max_time)

            if plateau_time > max_time:
                plateau_time = 0

            translation_rate = calculate_translation_rate(k, n, K, max_yield)

            print(f"Parameters for {condition_name} ({replicate_name}):")
            print("k_prime (k'):", k_prime, "k:", k, "K:", K, "n:", n)
            print("Yield:", max_yield, "Plateau Time:", plateau_time, "Translation Rate:", translation_rate)

            fitted_values = sigmoid_model(time_points, *popt)
            plt.plot(time_points, fluorescence_values, linewidth=8, label=f'{replicate_name} (Data)', color=color)
            fit_lines.append((time_points, fitted_values))

            all_parameters.append([
                replicate_name, k_prime, k, K, n, max_yield, plateau_time, translation_rate
            ])
        else:
            print(f"Failed to fit {replicate_name}. Plotting only experimental data.")
            plt.plot(time_points, fluorescence_values, linewidth=8, label=f'{replicate_name} (Data)', color=color)

    # Plot all fits in black
    for time_points, fitted_values in fit_lines:
        plt.plot(time_points, fitted_values, 'k-', linewidth=3)

    # Legend and axis setup
    plt.plot([], [], 'k-', linewidth=1, label='Fit')
    plt.xlabel('Time (hours)', size=14)
    plt.ylabel(f'{fluorescent_protein} Fluorescence (RFU)', size=14)
    plt.title(clean_condition_name(condition_name), size=20)
    plt.grid(False)

    # Save the figure
    plot_name = os.path.join(save_dir, f"{clean_condition_name(condition_name)}.png")
    plt.savefig(plot_name)
    plt.close()

    return all_parameters


# --- Format column widths uniformly in the Excel output ---
def set_uniform_column_width(ws, padding=1):
    """
    Set all Excel columns to the same width based on max content length.
    """
    max_length = 0
    for col in ws.columns:
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
    adjusted_width = max(max_length + padding, 10)
    for col in ws.columns:
        column = col[0].column_letter
        ws.column_dimensions[column].width = adjusted_width


# --- Main analysis workflow ---
def main():
    mCherry = False  # Change to True if you're analyzing mCherry fluorescence

    # Define input/output paths
    input_file = "u:\\Thèse\\Data\\Delft\\Ellen\\PURE_Syn_Chrom_mVenus.xlsx"
    output_file = "u:\\Thèse\\Data\\Delft\\Ellen\\PURE_Syn_Chrom_mVenusMaxYield\\Kinetic_Parameters.xlsx"
    save_dir = "u:\\Thèse\\Data\\Delft\\Ellen\\PURE_Syn_Chrom_mVenusMaxYield"
    os.makedirs(save_dir, exist_ok=True)

    # Read time series fluorescence data
    df = pd.read_excel(input_file, header=None)
    time_points = df.iloc[0, 2:].values.astype(float)
    polymerases = df.iloc[1:, 0].values
    conditions = df.iloc[1:, 1].values
    fluorescence_data = df.iloc[1:, 2:].values.astype(float)

    # Organize replicates by condition
    grouped_data = {}
    for polymerase, condition_name, fluorescence_values in zip(polymerases, conditions, fluorescence_data):
        base_condition_name = f"{polymerase} - {condition_name.split('rep')[0].strip()}"
        if base_condition_name not in grouped_data:
            grouped_data[base_condition_name] = {}
        replicate_name = condition_name.split('rep')[-1].strip()
        grouped_data[base_condition_name][f"rep {replicate_name}"] = fluorescence_values

    # Fit and collect parameters for all conditions and replicates
    all_parameters = []
    for condition_name, replicate_data in grouped_data.items():
        parameters = fit_and_plot_replicates(time_points, replicate_data, save_dir, condition_name, mCherry)
        for replicate_params in parameters:
            all_parameters.append([condition_name] + replicate_params)

    # Define output table headers
    parameter_columns = [
        "Condition", "Replicate", "k'", "k (RFU)", "K (hours)", "n",
        "Yield (RFU)", "Plateau Time (hours)", "Translation Rate (RFU/hours)"
    ]

    # Create output DataFrame and save to Excel
    parameter_df = pd.DataFrame(all_parameters, columns=parameter_columns)
    parameter_df.to_excel(output_file, index=False)

    # Adjust Excel formatting
    wb = load_workbook(output_file)
    ws = wb.active
    set_uniform_column_width(ws)
    wb.save(output_file)

    print(f"Fitted parameters saved to {output_file}")


# --- Entry point ---
if __name__ == "__main__":
    main()
