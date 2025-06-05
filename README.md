Interactive Spectroscopy – Analysis of Experimental and Simulated Spectra

Author: Laura Lerza  
Technologies Used: Python, PySimpleGUI, NumPy, Plotly  

**Description**

This interactive tool enables the analysis and comparison of experimental optical spectra (multi-frame) acquired from spectrometers with simulated spectra based on theoretical atomic line data (NIST database). It is designed primarily for plasma diagnostics.

The GUI-based application supports:
- Time-resolved visualization of experimental spectra
- Simulation of atomic spectra using Gaussian convolution
- Combined analysis of experimental and simulated data


---

**Features**

- Interactive GUI built with PySimpleGUI
- Plotly-based visualization with real-time sliders
- Support for multiple atomic species: Argon, Fluorine, Helium, Hydrogen
- Gaussian convolution of theoretical lines with adjustable σ
- Three vertical scale modes:
  - Autoscale
  - Normalize (0–1 range)
  - Offset-correct (baseline subtraction)
- Combined modes for comparing experimental and simulated spectra
- Terminal interaction for selecting simulation datasets

---

**Required File Structure**

- Experimental files (`.txt` or `.csv`) can be stored in any user-selected folder (in this case folder = `Ocean_Optics`).
- Atomic line files must be placed in:

  ```bash
  Path(__file__).resolve().parent / "Atomic_Lines_NIST"
  ```

  Example files:
  - `Argon_I.csv`
  - `Hydrogen_I.csv`
  - `H_F_He.csv` (for combined simulations)

- Include the logo image `Logo_Laura_resized.png` in the same folder as the Python script.

---

**How to Run**

- Prerequisites

  - Install required packages:

    ```bash
    pip install PySimpleGUI numpy plotly
    ```

  - Run the Script

    From terminal:

    ```bash
    python Spectral_GUI.py
    ```

---

**User Guide**

GUI Overview

1. Select Folder: Choose the directory containing experimental `.txt` or `.csv` files.
2. Select File: Click on a file to load it.
3. Choose Vertical Scale: `Autoscale`, `Normalize`, or `Offset-correct`.
4. Buttons:
   - `Experimental Spectrum`: View with time slider
   - `Simulated Spectrum`: Terminal input to simulate atomic lines
   - `Simulated + Experimental Spectrum`: Launch comparison window
   - `Exit`: Close the program

*Terminal Inputs (for Simulated Spectra)*

After selecting "Simulated Spectrum", use the terminal to enter:

- `trace1` → Argon I  
- `trace2` → Argon II  
- `trace3` → Fluorine I  
- `trace4` → Helium I  
- `trace5` → Hydrogen I  
- `trace6` → H + F + He  

Use the slider to adjust the Gaussian width (σ).  
Enter `stop` to exit this mode.

---

*Combined Mode Features*

Upon selecting "Simulated + Experimental Spectrum", a second window opens:

1. Fix a frame: Selects a single experimental frame (typically with peak intensity) and overlays a simulated spectrum.
2. Sum all counts: Aggregates intensity across time and compares it to simulated data.
3. Save options (reserved for future extensions):
   - `Save calibrated spectrum`
   - `Save fit data`

---

**Suggested Test Cases**

 *Experimental Tests*

| Shot      | Fuel Type                          | Use Case                                           |
|-----------|------------------------------------|----------------------------------------------------|
| 994       | Hydrogen                           | Inglis-Teller density comparison                   |
| 1022      | Hydrogen_normalized_1022           | Calibrated vs. uncalibrated comparison             |
| 1171      | Argon_II                           | Heavy gas recognition                              |
| 1228      | Hydrogen_I_normalized              | Impurity analysis and simulation comparison        |
| 2089–2183 | Helium                             | Density/temperature and line ratio tests           |
| 2183      | Helium (full spectrum over time)   | Full-frame analysis                                |

*Simulated Spectra*

- Pure Hydrogen
- Hydrogen + Fluorine
- Combination (H+F+He)

*Combined (Experimental + Simulated)*

- 1228: Direct experimental/simulated overlay
- 994: Theoretical density comparison
- 1022: Calibration verification
- 2089–2176: Line ratio simulations (Helium I)

---

**Notes**

- Automatic wavelength calibration:
  - `Ocean Optics` → +1.17 nm

- Code includes configurable thresholds for filtering and directories.

---

**Contact**

For questions or suggestions, please contact: Laura Lerza->mail:laura.lerza@studio.unibo.it
