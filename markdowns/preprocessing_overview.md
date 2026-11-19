# Gannet Pre-Processing Overview

Gannet is an established analysis pipeline for Magnetic Resonance Spectroscopy (MRS) data, particularly tuned for J-difference edited data (e.g., MEGA-PRESS, HERMES) targeting metabolites like GABA, GSH, and Lactate. 

The `GannetLoad` module is primarily responsible for the pre-processing of the data. Below is a detailed overview of each step involved in Gannet's pre-processing workflow, generally following the execution order in `GannetLoad.m`.

## 1. Pre-Initialisation (`GannetPreInitialise` / `CheckTargets`)
**Overview:** This step sets up the processing environment and configuration.
**How Gannet processes data:** Gannet reads in a configuration structure (often modified by `GannetPreInitialise.m`). It sets global variables, the target metabolites to quantify, vendor details, and user-defined pre-processing options (e.g., enabling/disabling water removal, line broadening, or choosing the alignment algorithm). It also validates the requested target metabolites.

## 2. File Parsing and Data Loading (`DICOMRead`, `GERead`, `SiemensTWIXRead`, etc.)
**Overview:** This step identifies the format of the MRS data based on the vendor and loads the raw FIDs (Free Induction Decays) into memory.
**How Gannet processes data:** Gannet calls vendor-specific reading routines (like `SiemensTWIXRead.m` or `PhilipsRead.m`) to load the raw time-domain signal. It extracts necessary metadata from the headers, such as spectral width, Larmor frequency, echo time (TE), repetition time (TR), and number of points. The raw data is stored in a structured array (`MRS_struct.fids.data`). If an unsuppressed water reference is provided, it is loaded simultaneously.

## 3. Parallel Reconstruction (`senseRecon`)
**Overview:** If the data originates from a multi-voxel or parallel transmit acquisition (e.g., PRIAM), the coil sensitivity maps must be applied to reconstruct the spatial origins of the signal.
**How Gannet processes data:** For PRIAM datasets, Gannet reads the coil reference scan and calculates the SENSE (Sensitivity Encoding) reconstruction matrix. It then multiplies the uncombined data by the SENSE matrix to reconstruct the individual voxels from the combined coil signals, resolving the spatial localization.

## 4. Zero-Filling
**Overview:** Zero-filling increases the digital resolution of the resulting frequency-domain spectrum.
**How Gannet processes data:** Gannet calculates the number of points required to reach a nominal spectral resolution of 0.061 Hz/point (e.g., zero-filling to 32768 points for typical bandwidths). The raw time-domain FIDs are padded with zeros up to this length before the Fourier Transform.

## 5. Eddy Current Correction (`EddyCurrentCorrection`)
**Overview:** Time-varying magnetic field gradients induce eddy currents in the scanner hardware, causing unwanted phase and frequency shifts in the MRS signal.
**How Gannet processes data:** If enabled, Gannet uses the unsuppressed water reference signal to correct the metabolite data. It calculates the phase of the unsuppressed water FID at each time point and subtracts this phase from the corresponding points in the water-suppressed metabolite FIDs. This compensates for phase shifts caused by eddy currents.

## 6. Apodization / Line Broadening
**Overview:** Apodization reduces the truncation artifact (sinc wiggles) and suppresses high-frequency noise at the tail of the FID, at the cost of slight broadening of the spectral peaks.
**How Gannet processes data:** Gannet applies an exponential filter to the time-domain data. The signal is multiplied by $e^{-t \times LB \times \pi}$, where $LB$ is the line-broadening factor (typically 3 Hz) and $t$ is the time vector. 

## 7. Fast Fourier Transform (FFT)
**Overview:** The time-domain FIDs are converted into the frequency-domain spectra.
**How Gannet processes data:** A Fast Fourier Transform (`fft`) is applied to the time-domain data, coupled with `fftshift` to center the zero frequency component. The frequency axis is mapped to parts-per-million (ppm) based on the scanner's Larmor frequency. 

## 8. Global Zero-Order Phase Correction (`PhaseCorrection`)
**Overview:** A zero-order phase error causes the spectral peaks to become asymmetrical (a mix of absorptive and dispersive line shapes).
**How Gannet processes data:** Gannet estimates and applies a bulk zero-order phase shift to all spectra to maximize the positive, real part of the spectrum. This is typically achieved by optimizing a phase parameter to maximize the integral of the real part of the spectrum.

## 9. Spectral Alignment / Registration (`SpectralRegistration` / `AlignUsingPeak`)
**Overview:** Due to subject motion or scanner field drift ($B_0$ drift), individual transients (scans) can shift in frequency and phase over the duration of the acquisition. Spectral alignment corrects these shifts frame-by-frame.
**How Gannet processes data:** Gannet aligns the individual frames before they are averaged. Depending on the settings, it uses one of several methods:
- **Robust Spectral Registration:** Time-domain frequency and phase correction routine that aligns each frame to a template (usually the median or a running average) by minimizing the difference between them.
- **Peak Alignment:** Fits a simple model to a prominent reference peak (e.g., Creatine at 3.02 ppm) frame-by-frame and applies a frequency shift to align them all.

## 10. Outlier Rejection and Signal Averaging (`SignalAveraging`)
**Overview:** Individual scans are averaged together to boost the Signal-to-Noise Ratio (SNR). Before averaging, heavily corrupted scans (e.g., from motion or lipid contamination) can be identified and rejected or down-weighted.
**How Gannet processes data:**
- Gannet categorizes the aligned frames into their respective sub-experiments (e.g., 'ON' and 'OFF' edit pulses).
- **Arithmetic Averaging:** Simple mean of all accepted frames.
- **Weighted Averaging:** Assigns weights to each frame based on its similarity to the median spectrum, effectively down-weighting outliers and motion-corrupted frames.
- Gannet then subtracts the averaged 'OFF' spectrum from the 'ON' spectrum to generate the **DIFF (Difference)** spectrum, which reveals the J-difference edited metabolite (e.g., GABA), and adds them to create the **SUM** spectrum.

## 11. Residual Water Removal (`WaterRemovalHSVD`)
**Overview:** Even with water suppression, a large residual water peak often remains at 4.7 ppm, which can cause baseline distortions that affect the quantification of nearby metabolites.
**How Gannet processes data:** Gannet uses the Hankel Singular Value Decomposition (HSVD) algorithm. It transforms the averaged DIFF and SUM spectra back to the time domain, models the low-frequency components corresponding to the water resonance (usually between 4.6 and 4.8 ppm), and subtracts this modeled water signal from the data. The data is then transformed back to the frequency domain.

## 12. Baseline Correction
**Overview:** A final, simple adjustment to ensure the spectral baseline sits at zero.
**How Gannet processes data:** Gannet calculates the mean signal amplitude in a noise-only region (usually far downfield, e.g., between 7 and 8 ppm) and subtracts this constant value from the entire spectrum.

## 13. Output Generation and Quality Control
**Overview:** The pre-processing pipeline concludes by packaging the processed data and generating visual quality control reports.
**How Gannet processes data:** The final spectra (ON, OFF, SUM, DIFF) are stored in the `MRS_struct`. Gannet generates a PDF report showing pre- and post-alignment heatmaps of the spectra (to visualize frequency drift correction) and plots the frequency offset of the reference peak across averages. This allows the user to visually inspect the success of the pre-processing.
