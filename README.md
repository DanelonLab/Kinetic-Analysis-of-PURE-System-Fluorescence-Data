# Kinetic Analysis of PURE System Fluorescence Data

## 📌 Overview

This repository provides tools for automated kinetic analysis of time-course fluorescence data (mVenus, eYFP, or mCherry) from PURE system experiments. A four-parameter sigmoid model is fitted to each replicate of each condition, and key kinetic parameters are extracted. Outputs include per-condition plots and an Excel file with all calculated metrics.

The sigmoid model is used as described in Blanken et al. 2019 *Phys. Biol.* 16 045002 ([DOI 10.1088/1478-3975/ab0c62](https://doi.org/10.1088/1478-3975/ab0c62)) and in Doerr et al. 2019 Phys. Biol. 16 025001 ([DOI 10.1088/1478-3975/aaf33d](https://doi.org/10.1088/1478-3975/aaf33d)).

Two versions of the tool are available in this repository:

| Version | File | Description |
|---|---|---|
| **Packaged library** *(recommended)* | `pure_kinetics/` + `run_analysis.py` | Modular Python package; install with pip |
| **Legacy script** | `Kinetic_Analysis_Fluorescence_Script_Annotated.py` | Original single-file annotated script |

---

## ⚙️ Features

- Fits a **4-parameter sigmoid model** to time-course fluorescence data.
- Handles **multiple replicates** per condition.
- Calculates:
  - `k'`: Basal fluorescence (offset)
  - `k`: Amplitude of fluorescence increase
  - `K`: Half-maximal time (hours)
  - `n`: Hill coefficient (curve steepness)
  - **Yield**: Maximum sliding-window average signal (robust to noise and end-plateau decline)
  - **Plateau time**: Estimated time at which signal stabilizes
  - **Translation rate**: Heuristic kinetic metric for relative comparison between conditions, estimated as `(k · n) / (4 · K)`
- Saves:
  - One PNG plot per condition (raw data + sigmoid fits overlaid)
  - An Excel file with all replicate-level parameters

---

## 🛠️ Installation

### Requirements

- Python ≥ 3.8
- Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`

### Install directly from GitHub

```bash
pip install git+https://github.com/DanelonLab/Kinetic-Analysis-of-PURE-System-Fluorescence-Data.git
```

This installs the `pure_kinetics` package and all its dependencies automatically.

### Alternative: clone and install locally

```bash
git clone https://github.com/DanelonLab/Kinetic-Analysis-of-PURE-System-Fluorescence-Data.git
cd Kinetic-Analysis-of-PURE-System-Fluorescence-Data
pip install .
```

---

## 🚀 How to Run an Analysis

### 1. Edit the configuration block in `run_analysis.py`

Open `run_analysis.py` and set the paths and options at the top of the file:

```python
INPUT_FILE  = "path/to/your/input.xlsx"
OUTPUT_FILE = "path/to/save/Kinetic_Parameters.xlsx"
SAVE_DIR    = "path/to/save/plots"
mCherry     = False   # Set to True if analyzing mCherry fluorescence
```

### 2. Run from your terminal

```bash
python run_analysis.py
```

### 3. Review the outputs

- **PNG plots** saved to `SAVE_DIR` — one file per condition, showing raw replicates and sigmoid fits overlaid. Visual inspection of these plots is the primary quality-control step.
- **Excel file** saved to `OUTPUT_FILE` — one row per replicate with all kinetic parameters (see Output section below).

---

## 🧪 Quick Start with the Example Dataset

An example dataset (`MSG1_1_mCherry.xlsx`) is included in the repository. It contains mCherry fluorescence time-course data from a PURE system experiment. To run the analysis on this dataset:

```python
INPUT_FILE  = "MSG1_1_mCherry.xlsx"
OUTPUT_FILE = "results/MSG1_1_mCherry_Kinetic_Parameters.xlsx"
SAVE_DIR    = "results/plots"
mCherry     = True    # This dataset uses mCherry fluorescence
```

Then run:

```bash
python run_analysis.py
```

A `results/` folder will be created containing the plots and the Excel parameter file.

---

## 📁 Input Format

The script expects an Excel file structured as follows:

| (Row 1 → time points) | | t=0 | t=0.5 | t=1 | ... |
|---|---|---|---|---|---|
| T7 | Cond1 rep 1 | ... | ... | ... | |
| T7 | Cond1 rep 2 | ... | ... | ... | |
| SP6 | Cond2 rep 1 | ... | ... | ... | |

- **Row 1**: Time points in hours, starting from column C
- **Column A**: Polymerase name (e.g., `T7`, `SP6`)
- **Column B**: Condition and replicate label (e.g., `Cond1 rep 1`)
- **Columns C onward**: Fluorescence values over time (RFU)

> **Note:** The script assumes **equally spaced time points**. The sliding-window yield estimation relies on consistent temporal sampling and may not be appropriate for datasets with missing or irregularly spaced measurements.

---

## 📦 Output

### Excel file

One row per replicate, with the following columns:

| Condition | Replicate | k′ | k (RFU) | K (hours) | n | Yield (RFU) | Plateau Time (hours) | Translation Rate (RFU/h) |
|---|---|---|---|---|---|---|---|---|
| T7 - Cond1 | rep 1 | ... | ... | ... | ... | ... | ... | ... |

- `Condition`: Polymerase + base condition name
- `Replicate`: Replicate identifier (e.g., `rep 1`)
- `k′`: Basal fluorescence offset
- `k`: Amplitude of fluorescence increase
- `K`: Half-maximal time (hours)
- `n`: Hill coefficient (controls steepness)
- `Yield`: Maximum sliding-window average (window = 20 time points by default)
- `Plateau Time`: Estimated time at which the signal plateaus; returns `0` if the plateau is not reached within the experiment duration
- `Translation Rate`: Estimated as `(k · n) / (4 · K)` — a heuristic metric for relative comparison between conditions; non-physical values are reported as `NaN`

### Plots

One PNG per condition, displaying raw replicate traces in the fluorescent protein color scheme (mVenus: green, eYFP: yellow, mCherry: red) with sigmoid fits overlaid in black.

---

## 🧩 Using `pure_kinetics` as a Library

Individual functions can be imported and used directly in your own scripts:

```python
from pure_kinetics import (
    load_fluorescence_data,
    fit_with_retries,
    get_sliding_window_max_yield,
    calculate_plateau_time,
    calculate_translation_rate,
    plot_condition,
    save_parameters_to_excel,
)
```

| Function | Module | Description |
|---|---|---|
| `load_fluorescence_data` | `io` | Load and group replicates from the input Excel file |
| `fit_with_retries` | `fitting` | Fit the sigmoid model with multi-restart strategy |
| `get_sliding_window_max_yield` | `yield_estimation` | Robust yield estimate via sliding-window average |
| `calculate_plateau_time` | `models` | Estimate plateau time from fit parameters |
| `calculate_translation_rate` | `models` | Estimate translation rate from fit parameters |
| `plot_condition` | `plotting` | Generate and save overlay plot for one condition |
| `save_parameters_to_excel` | `io` | Export all fitted parameters to a formatted Excel file |

---

## 📝 Notes and Limitations

- If fitting fails for a replicate (after 120 random restarts), only the raw data is plotted and no parameters are saved for that replicate.
- Yield is estimated from the **sliding window average** (default window = 20 time points) rather than the raw endpoint value, because fluorescence signals in PURE system experiments can gradually decline after reaching their maximum (e.g., due to evaporation or condensation). This approach is independent of the sigmoid fit.
- No explicit parameter bounds are enforced during fitting, as RFU values vary substantially between instruments. Initial guesses are derived heuristically from the data.
- The sigmoid model assumes a smooth, monotonically increasing fluorescence curve. Curves with multiple phases, pronounced lag, or post-maximum decay may yield poor or misleading fits. Visual inspection of the output plots is strongly recommended.
- Baseline subtraction is not automated; background fluorescence is assumed to be minimal.
- The current implementation does not export formal goodness-of-fit metrics (e.g., R², residuals). The generated plots serve as the primary quality-control step.

| Feature | Current implementation | Possible extension |
|---|---|---|
| Model | 4-parameter sigmoid | Extend via `sigmoid_model()` for alternative kinetic shapes |
| Packaging | Python library (`pure_kinetics`) | Upload library to PyPI |
| Time-point handling | Equally spaced time points required | Interpolation or adaptive windowing for uneven sampling |
| Fit diagnostics | Visual inspection via plots | Export R², residuals, or confidence intervals |
| Parameter bounds | Unconstrained (heuristic initial guesses) | User-defined bounds for specific instruments or conditions |

---

## 🧬 About

This tool was developed for automated kinetic analysis of protein expression using the **PURE system**. It helps characterize transcription-translation dynamics under varying conditions by modeling fluorescence profiles with a phenomenological sigmoid model.

Contributions and suggestions are welcome.

---

## Developer

Yannick Bernard-Lapeyre  
Danelon Lab ([www.danelonlab.com](http://www.danelonlab.com))
