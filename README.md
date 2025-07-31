# Kinetic Analysis of PURE System Fluorescence Data

## 📌 Overview

This script analyzes time-course fluorescence data (mVenus, eYFP, or mCherry) from PURE system experiments. It fits a four-parameter sigmoid model to each replicate of each condition and extracts key kinetic parameters. Outputs include plots of the fits and an Excel file with all calculated metrics.

The sigmoid model is used as described in Blanken et al 2019 Phys. Biol. 16 045002 (DOI 10.1088/1478-3975/ab0c62).

---

## ⚙️ Features

- Fits a **sigmoid model** to time-course fluorescence data.
- Handles **multiple replicates** per condition.
- Calculates:
  - `k'`: Basal fluorescence
  - `k`: Amplitude of increase
  - `K`: Half-max time
  - `n`: Hill coefficient
  - **Yield** (max average over sliding window)
  - **Plateau time**
  - **Translation rate**
- Saves:
  - A PNG plot per condition
  - An Excel file with all replicate parameters


---

## 📁 Input Format

The script expects an Excel file like this:

| Polymerase | Condition      | t0  | t1  | t2  | ... |
|------------|----------------|-----|-----|-----|-----|
| T7         | Cond1 rep 1    | ... | ... | ... |     |
| T7         | Cond1 rep 2    | ... | ... | ... |     |
| SP6        | Cond2 rep 1    | ... | ... | ... |     |

- **Row 1**: Time points (from column C onward)
- **Column A**: Polymerase name (e.g., T7, SP6)
- **Column B**: Condition and replicate (e.g., `Cond1 rep 1`)
- **Columns C+**: Fluorescence values over time

---

## 🚀 How to Use

1. **Edit the `main()` function** in the script:

   ```python
   input_file = "path/to/your/input.xlsx"
   output_file = "path/to/save/results/Kinetic_Parameters.xlsx"
   save_dir = "path/to/save/plots"
   mCherry = False  # Set to True if analyzing mCherry fluorescence

2. **Run the script** from your terminal

3. Review outputs:

- PNG plots saved to save_dir
- Excel file with fitted parameters saved to output_file

## 📦 Output

### 🧾 Excel File Format

After fitting, the script generates an Excel file with one row per replicate. Each row contains the kinetic parameters extracted from the sigmoid fit.

| Condition    | Replicate | k′   | k (RFU) | K (hours) | n   | Yield (RFU) | Plateau Time (hours) | Translation Rate (RFU/h) |
|--------------|-----------|------|---------|-----------|-----|--------------|------------------------|---------------------------|
| T7 - Cond1   | rep 1     | ...  | ...     | ...       | ... | ...          | ...                    | ...                       |

- `Condition`: Polymerase + base condition name
- `Replicate`: Replicate ID (e.g., rep 1)
- `k′`: Basal fluorescence (offset)
- `k`: Amplitude of increase
- `K`: Half-maximal time
- `n`: Hill coefficient (curve sharpness)
- `Yield`: Max average signal from sliding window
- `Plateau Time`: Time when signal is considered stabilized
- `Translation Rate`: Estimated as Translation Rate= (4⋅K)/(k⋅n)
​
## 📝 Notes

- If fitting fails for a replicate, only the raw data will be plotted (no fitted curve).
- Yield is computed using a **sliding window average** (default window = 20 timepoints) to reduce noise.
- Plots use predefined color schemes depending on the fluorescent protein (mVenus, eYFP, or mCherry).

---

## 🧬 About

This script was developed for automated kinetic analysis of protein expression using the **PURE system**. It helps characterize transcription-translation dynamics under varying conditions by modeling fluorescence profiles with a phenomenological model.

If you'd like to adapt this tool for other applications or improve its fitting logic, feel free to extend the code. Contributions or suggestions are welcome.

---

## Developer

Yannick Bernard-Lapeyre

Danelon Lab (www.danelonlab.com)

---
