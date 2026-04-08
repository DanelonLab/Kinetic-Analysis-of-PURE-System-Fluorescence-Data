"""
io.py
-----
Data loading and Excel export functions for PURE system fluorescence analysis.

Expected input format
---------------------
An Excel file where:
  - Row 1        : Time points (from column C onward), in hours
  - Column A     : Polymerase name (e.g., T7, SP6)
  - Column B     : Condition and replicate label (e.g., "Cond1 rep 1")
  - Columns C+   : Fluorescence values over time (RFU)

Note: The script assumes equally spaced time points. The sliding-window yield
estimation relies on consistent temporal sampling and may not be appropriate
for datasets with missing or irregularly spaced measurements.
"""

import pandas as pd
from openpyxl import load_workbook


def load_fluorescence_data(input_file):
    """
    Load and organize fluorescence time-course data from an Excel file.

    Reads the input spreadsheet and groups replicates by condition, combining
    the polymerase name and base condition label into a single condition key.

    Parameters
    ----------
    input_file : str
        Path to the input Excel file.

    Returns
    -------
    time_points : np.ndarray
        Array of time points (hours).
    grouped_data : dict
        Nested dict: {condition_name: {replicate_name: fluorescence_array}}.
    """
    df = pd.read_excel(input_file, header=None)
    time_points = df.iloc[0, 2:].values.astype(float)
    polymerases = df.iloc[1:, 0].values
    conditions = df.iloc[1:, 1].values
    fluorescence_data = df.iloc[1:, 2:].values.astype(float)

    grouped_data = {}
    for polymerase, condition_name, fluorescence_values in zip(polymerases, conditions, fluorescence_data):
        base_condition = f"{polymerase} - {condition_name.split('rep')[0].strip()}"
        replicate_name = f"rep {condition_name.split('rep')[-1].strip()}"
        grouped_data.setdefault(base_condition, {})[replicate_name] = fluorescence_values

    return time_points, grouped_data


def save_parameters_to_excel(all_parameters, output_file):
    """
    Save fitted kinetic parameters to a formatted Excel file.

    Each row corresponds to one replicate. Columns are uniformly sized for
    readability.

    Parameters
    ----------
    all_parameters : list of list
        Each entry: [condition, replicate, k', k, K, n, yield, plateau_time,
                     translation_rate]
    output_file : str
        Path to the output Excel file.
    """
    columns = [
        "Condition", "Replicate", "k'", "k (RFU)", "K (hours)", "n",
        "Yield (RFU)", "Plateau Time (hours)", "Translation Rate (RFU/hours)"
    ]
    df = pd.DataFrame(all_parameters, columns=columns)
    df.to_excel(output_file, index=False)

    # Uniform column width formatting
    wb = load_workbook(output_file)
    ws = wb.active
    max_length = max(
        (len(str(cell.value)) for col in ws.columns for cell in col if cell.value),
        default=10
    )
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = max_length + 2
    wb.save(output_file)
