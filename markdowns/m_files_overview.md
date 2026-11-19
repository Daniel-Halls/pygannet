# Gannet `.m` Files Overview

This document provides a brief overview of the main MATLAB (`.m`) files in the root directory of the Gannet repository. They are broadly grouped by their functionality in the pipeline.

## Main Pipeline Modules
These files constitute the main user-facing pipeline functions that carry out the core processing steps.

*   **`GannetPreInitialise.m`**: Sets up global parameters, configurations, and options for the Gannet workflow before loading the data (e.g., target metabolites, vendor, file paths).
*   **`GannetLoad.m`**: The primary entry point for loading raw MRS data. Handles reading, zero-filling, eddy current correction, phase correction, frequency alignment, signal averaging, and water removal.
*   **`GannetFit.m`**: Performs nonlinear least-squares curve fitting on the frequency-domain spectra (e.g., modeling GABA, Glx, GSH peaks using multi-Gaussian models) to extract peak areas.
*   **`GannetFitPhantom.m`**: A specialized version of the fitting routine tailored for phantom (in vitro) data.
*   **`GannetCoRegister.m`**: Coregisters the MRS voxel to a high-resolution structural MRI image (usually T1-weighted) to visualize the voxel placement and prepare for segmentation.
*   **`GannetSegment.m`**: Interfaces with SPM (Statistical Parametric Mapping) to segment the structural MRI into Gray Matter (GM), White Matter (WM), and Cerebrospinal Fluid (CSF), determining tissue fractions within the MRS voxel.
*   **`GannetQuantify.m`**: Quantifies the metabolite concentrations using the fitted peak areas from `GannetFit` and the tissue fractions from `GannetSegment`.

## Data Reading and Parsing
Routines to interpret vendor-specific or generic data formats and headers.

*   **`DICOMRead.m` / `SiemensDICOMRead.m`**: Load raw FIDs from generic or Siemens DICOM files.
*   **`GERead.m`**: Loads GE P-file raw data.
*   **`NIfTIMRSRead.m`**: Reads standard NIfTI-MRS format files.
*   **`PhilipsRead.m` / `PhilipsDataRead.m` / `PhilipsDataRead_7T.m` / `PhilipsRawRead.m`**: Load Philips `.SDAT`/`.SPAR` or `.data`/`.list` formats.
*   **`SiemensRead.m` / `SiemensTWIXRead.m`**: Load Siemens `.rda` (Siemens Read) or `.dat` (Siemens TWIX raw data) files.
*   **`SDATread.m` / `SDATreadMEGA.m`**: Extracts data arrays from Philips `.SDAT` files.
*   **`mapVBVD_Gannet.m` / `read_twix_hdr_Gannet.m` / `twix_map_obj_Gannet.m`**: Utilities for parsing complex Siemens TWIX raw data formats.
*   **`read_dcm_header.m` / `dicom_open.m` / `dicom_move.m` / `dicom_get_spectrum_siemens.m`**: Helper functions to traverse and read metadata/FIDs from DICOM files.

## Alignment and Averaging
Functions responsible for correcting frequency/phase drift and generating final spectra.

*   **`AlignUsingPeak.m`**: Aligns frames by fitting a prominent peak (e.g., NAA or Creatine) and calculating its frequency shift.
*   **`AlignUsingH2O.m`**: Aligns frames based on the residual unsuppressed water peak.
*   **`SpectralRegistration.m` / `SpectralRegistrationHERMES.m` / `RobustSpectralRegistration.m`**: Perform time-domain frequency and phase corrections (Spectral Registration) to align the transients, with robust or HERMES-specific variants.
*   **`SignalAveraging.m`**: Combines aligned transient frames (ON and OFF) via arithmetic or weighted averaging to produce SUM and DIFF spectra.
*   **`AlignSubSpectra.m` / `AlignSubSpectra_PreAlignRef.m`**: Support routines for frame alignment.

## Core Modeling and Fitting Functions
Functions that define the mathematical models used for peak fitting.

*   **`GaussModel.m` / `DoubleGaussModel.m` / `FiveGaussModel.m` / `SixGaussModel.m`**: Various single or multi-Gaussian models used to fit specific multiplet patterns (e.g., GABA, Glx, GSH).
*   **`LorentzModel.m` / `TwoLorentzModel.m` / `ThreeLorentzModel.m`**: Lorentzian lineshape models.
*   **`LorentzGaussModel.m` / `LorentzGaussModelP.m` / `VoigtModel.m`**: Mixed lineshape models (pseudo-Voigt or Voigt).
*   **`BaselineModel.m`**: Simple polynomial/baseline model used in fitting.
*   **`GABAGlxModel.m` / `EtOHModel.m` / `LacModel.m`**: Specific combined models for fitting GABA+Glx, Ethanol, or Lactate.
*   **`FitCr.m` / `FitChoCr.m` / `FitPeaksByFrames.m`**: Helpers for fitting reference peaks (Creatine/Choline) across frames, often for alignment purposes.

## Signal Processing and Correction
Low-level operations on the FIDs and spectra.

*   **`PhaseCorrection.m`**: Applies zero-order global phase correction to maximize the real component of the spectrum.
*   **`EddyCurrentCorrection.m`**: Uses an unsuppressed water signal to correct eddy current-induced phase distortions in the metabolite data.
*   **`WaterRemovalHSVD.m`**: Removes the residual water peak using Hankel Singular Value Decomposition (HSVD).
*   **`SignalFilter.m`**: Provides smoothing or filtering of the spectra for baseline recognition.
*   **`CalcNoise.m`**: Estimates the standard deviation of noise from a signal-free region of the spectrum.
*   **`SpecifyOnOffOrder.m`**: Determines the temporal order of edit pulses (ON/OFF) from the raw data.
*   **`FreqPhaseShiftNest.m`**: Applies a specified frequency and phase shift to time-domain data.

## Masking and Voxel Co-registration
Functions to create binary masks representing the MRS voxel in anatomical space.

*   **`GannetMask_GE.m` / `GannetMask_GE_nii.m`**: Extract spatial localization parameters from GE files to build a mask.
*   **`GannetMask_NIfTI.m`**: Extracts localization parameters for NIfTI-MRS.
*   **`GannetMask_Philips.m`**: Extracts localization parameters for Philips.
*   **`GannetMask_SiemensRDA.m` / `GannetMask_SiemensTWIX.m`**: Extracts localization parameters for Siemens formats.
*   **`NormalizeVoxelMask.m`**: Normalizes binary voxel masks into standard MNI space.
*   **`VoxelMaskOverlap.m`**: Computes overlap of multiple voxel masks.
*   **`voxel2world_space.m`**: Helper to convert voxel coordinates into scanner world coordinates.

## Utilities and De-Identification
Miscellaneous helper scripts and tools.

*   **`GannetVersion.m` / `VersionCheck.m` / `UpdateGannet.m`**: Checks current Gannet version, pings online repositories, and updates the software if needed.
*   **`ToolboxCheck.m`**: Checks if required MATLAB toolboxes (e.g., Image Processing Toolbox, Optimization Toolbox) are installed.
*   **`GetFullPath.m`**: Resolves an absolute, canonical file path.
*   **`ExportToCSV.m`**: Exports Gannet output variables (e.g., concentration, fit errors) to a CSV spreadsheet.
*   **`DICOMDeIdentify.m` / `GEDeIdentify.m` / `PhilipsDeIdentify.m` / `TWIXDeIdentify.m`**: A suite of tools to strip Protected Health Information (PHI) from raw MRS headers.
*   **`DeIdentifyDisclaimer.m`**: Displays warnings regarding data privacy before running de-identification tools.
*   **`DiscernDataType.m`**: Deduces the vendor and format of the input file by its extension or header.
*   **`PlotPrePostAlign.m` / `PlotPrePostAlign2.m` / `PaperPlot.m` / `SavePDF.m`**: Visualization tools used internally by Gannet to output PDF reports of the processed data.
*   **`mergestructs.m` / `sort_nat.m` / `write_xxx.m`**: General data structure manipulation and natural string sorting routines.
*   **`CalcIU.m`**: Calculates the concentration in Institutional Units (pseudo-concentration).

## VAX Conversion Functions
Utilities for reading older proprietary formats that store numbers in DEC VAX binary representation.

*   **`VAXD_to_uint64le.m` / `fwriteVAXD.m`**: Conversion for VAXD double precision format.
*   **`VAXF_to_uint32le.m` / `uint32le_to_VAXF.m`**: Conversion for VAXF single precision format.
*   **`VAXG_to_uint64le.m` / `uint64le_to_VAXG.m` / `freadVAXG.m`**: Conversion for VAXG double precision format.
