"""GannetLoad module: data loading, coil/ECC/phase corrections, and averaging."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gannet.config import GannetConfig, MRSStruct, check_targets, gannet_pre_initialise
from gannet.io.bids import bids_query, bids_resolve_uri
from gannet.io.dicom import dicom_read, siemens_dicom_read
from gannet.io.ge import ge_read
from gannet.io.nifti_mrs import nifti_mrs_read
from gannet.io.philips import philips_read
from gannet.io.philips_data import philips_data_read
from gannet.io.philips_raw import philips_raw_read
from gannet.io.siemens_rda import siemens_read
from gannet.io.siemens_twix import siemens_twix_read
from gannet.io.utils import discern_data_type
from gannet.priam.sense import sense_recon
from gannet.processing.alignment import (
    align_using_h2o,
    align_using_peak,
    robust_spectral_registration,
    spectral_registration,
    spectral_registration_hermes,
)
from gannet.processing.averaging import signal_averaging
from gannet.processing.corrections import eddy_current_correction, phase_correction
from gannet.processing.hsvd import water_removal_hsvd
from gannet.processing.on_off import specify_on_off_order
from gannet.utils.export import export_to_html, save_mrs_mat


def _read_scan_pair(
    mrs_struct: MRSStruct,
    metab_path: str,
    water_path: str | None,
) -> MRSStruct:
    """Dispatch MRS reader function matching the detected file vendor/format.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` with ``p.vendor`` and ``ii`` set.
    metab_path: str
        Path to metabolite MRS file.
    water_path: str | None
        Optional path to unsuppressed water file.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` containing raw time-domain FIDs and scan params.
    """
    vendor = str(mrs_struct["p"]["vendor"])
    if vendor == "Philips":
        return philips_read(mrs_struct, metab_path, water_path)
    if vendor == "Philips_data":
        return philips_data_read(mrs_struct, metab_path, water_path)
    if vendor == "Philips_raw":
        return philips_raw_read(mrs_struct, metab_path)
    if vendor == "Siemens_rda":
        return siemens_read(mrs_struct, metab_path, water_path)
    if vendor == "Siemens_twix":
        return siemens_twix_read(mrs_struct, metab_path, water_path)
    if vendor == "Siemens_dicom":
        return siemens_dicom_read(mrs_struct, metab_path, water_path)
    if vendor == "DICOM":
        return dicom_read(mrs_struct, metab_path, water_path)
    if vendor == "GE":
        return ge_read(mrs_struct, metab_path)
    if vendor == "NIfTI":
        return nifti_mrs_read(mrs_struct, metab_path, water_path)
    raise ValueError(f"Unsupported vendor format: {vendor}")


def _resolve_input_files(
    metabfile: str | list[str],
    waterfile: str | list[str] | None,
) -> tuple[list[str], list[str] | None]:
    """Normalize input file paths or BIDS directory queries into explicit file lists.

    Parameters
    ------------
    metabfile: str | list[str]
        Single file path, directory (BIDS/folder), or list of metabolite files.
    waterfile: str | list[str] | None
        Optional matching water reference path(s).

    Returns
    --------
        tuple[list[str], list[str] | None]:
            Tuple ``(metab_list, water_list)``.
    """
    if isinstance(metabfile, str):
        if metabfile.startswith("bids:"):
            resolved = bids_resolve_uri(metabfile, os.getcwd())
            metab_list = [resolved]
        elif os.path.isdir(metabfile):
            metab_list = bids_query(metabfile, suffix="mrs")
            if not metab_list:
                raise FileNotFoundError(f"No MRS files found in directory: {metabfile}")
        else:
            metab_list = [metabfile]
    else:
        metab_list = list(metabfile)

    if waterfile is None:
        return (metab_list, None)
    if isinstance(waterfile, str):
        return (
            metab_list,
            [waterfile] if len(metab_list) == 1 else [waterfile] * len(metab_list),
        )
    return (metab_list, list(waterfile))


def _trim_averages(
    fids: NDArray[np.complex128],
    trim_avgs: int,
    is_hermes: bool,
) -> NDArray[np.complex128]:
    """Trim transient count symmetrically to ``trim_avgs`` matching ``GannetLoad.m``.

    Parameters
    ------------
    fids: NDArray[np.complex128]
        Complex FID matrix shaped ``(npoints, navg)``.
    trim_avgs: int
        Target number of averages to retain (0 means no trimming).
    is_hermes: bool
        Whether data uses 4-subspectrum HERMES/HERCULES interleaving.

    Returns
    --------
        NDArray[np.complex128]:
            Trimmed FID matrix.
    """
    if trim_avgs <= 0 or fids.shape[1] <= trim_avgs:
        return fids
    block = 4 if is_hermes else 2
    keep = max(block, (trim_avgs // block) * block)
    return fids[:, : min(keep, fids.shape[1])]


def _run_frequency_phase_alignment(
    all_frames_ft: NDArray[np.complex128],
    mrs_struct: MRSStruct,
) -> tuple[NDArray[np.complex128], MRSStruct]:
    """Dispatch frequency and phase alignment algorithm configured in ``p.alignment``.

    Parameters
    ------------
    all_frames_ft: NDArray[np.complex128]
        Frequency-domain frames shaped ``(zf_pts, navg)``.
    mrs_struct: MRSStruct
        Active ``MRSStruct``.

    Returns
    --------
        tuple[NDArray[np.complex128], MRSStruct]:
            Aligned spectral frames and updated ``MRSStruct``.
    """
    p = mrs_struct["p"]
    method = str(p.get("alignment", "RobustSpecReg"))
    if p.get("HERMES"):
        return spectral_registration_hermes(mrs_struct)
    if method == "RobustSpecReg":
        return robust_spectral_registration(mrs_struct)
    if method == "SpecReg":
        return spectral_registration(mrs_struct, on_what=0, dual=0)
    if method == "SpecRegDual":
        return spectral_registration(mrs_struct, on_what=0, dual=1)
    if method in ("Cr", "Cho", "NAA"):
        return align_using_peak(all_frames_ft, mrs_struct)
    if method == "H2O":
        return align_using_h2o(all_frames_ft, mrs_struct)
    # "none" or unrecognized fallback
    ii = int(mrs_struct["ii"])
    n_avg = all_frames_ft.shape[1]
    mrs_struct["out"]["reject"][ii] = np.zeros(n_avg, dtype=int)
    mrs_struct["fids"]["data_align"] = mrs_struct["fids"]["data"]
    return (all_frames_ft.copy(), mrs_struct)


def gannet_load(
    metabfile: str | list[str],
    waterfile: str | list[str] | None = None,
    trim_avgs: int = 0,
    metabfile_join: str | list[str] | None = None,
    waterfile_join: str | list[str] | None = None,
    config: GannetConfig | None = None,
    save_html: bool = False,
    out_dir: str | None = None,
) -> MRSStruct:
    """Load, preprocess, align, and average MRS data (``GannetLoad.m``).

    Parameters
    ------------
    metabfile: str | list[str]
        Path(s) to metabolite MRS file(s) or BIDS directory.
    waterfile: str | list[str] | None
        Optional water reference file path(s).
    trim_avgs: int
        Optional cap on number of transients to retain per scan.
    metabfile_join: str | list[str] | None
        Optional secondary metabolite file(s) to concatenate with ``metabfile``.
    waterfile_join: str | list[str] | None
        Optional secondary water reference file(s) for join mode.
    config: GannetConfig | None
        Optional acquisition/processing parameter configuration override.
    save_html: bool
        Whether to automatically write a standalone HTML QC report after loading.
    out_dir: str | None
        Optional output directory for saved HTML/MAT artifacts.

    Returns
    --------
        MRSStruct:
            Populated ``MRSStruct`` containing aligned spectra, ``spec.freq``,
            processed FIDs, and acquisition/QC metadata.
    """
    metab_list, water_list = _resolve_input_files(metabfile, waterfile)
    mrs_struct = gannet_pre_initialise(config=config)
    mrs_struct["metabfile"] = metab_list
    if water_list is not None:
        mrs_struct["waterfile"] = water_list
        mrs_struct["p"]["Reference_compound"] = "H2O"

    check_targets(mrs_struct)
    n_scans = len(metab_list)
    mrs_struct["p"]["numScans"] = n_scans
    mrs_struct = discern_data_type(metab_list[0], mrs_struct)

    p = mrs_struct["p"]
    # Pre-allocate per-scan numpy arrays matching Gannet reader helpers
    p["npoints"] = np.zeros(n_scans, dtype=int)
    p["nrows"] = np.zeros(n_scans, dtype=int)
    p["Navg"] = np.zeros(n_scans, dtype=int)
    p["Nwateravg"] = np.zeros(n_scans, dtype=int)
    p["LarmorFreq"] = np.zeros(n_scans, dtype=float)
    p["sw"] = np.zeros(n_scans, dtype=float)
    p["TR"] = np.zeros(n_scans, dtype=float)
    p["TE"] = np.zeros(n_scans, dtype=float)
    p["ZeroFillTo"] = np.zeros(n_scans, dtype=int)
    p["voxdim"] = np.zeros((n_scans, 3), dtype=float)
    p["voxoff"] = np.zeros((n_scans, 3), dtype=float)
    p["voxang"] = np.zeros((n_scans, 3), dtype=float)
    p["tlhc"] = np.zeros((n_scans, 3), dtype=float)
    p["trhc"] = np.zeros((n_scans, 3), dtype=float)
    p["brhc"] = np.zeros((n_scans, 3), dtype=float)
    p["TablePosition"] = np.zeros((n_scans, 3), dtype=float)
    p["NormSag"] = np.zeros(n_scans, dtype=float)
    p["NormCor"] = np.zeros(n_scans, dtype=float)
    p["NormTra"] = np.ones(n_scans, dtype=float)
    p["VoI_InPlaneRot"] = np.zeros(n_scans, dtype=float)
    p["nrows_water"] = np.zeros(n_scans, dtype=int)

    mrs_struct["spec"]["F0freq"] = [None] * n_scans
    mrs_struct["spec"]["F0freq2"] = [None] * n_scans
    mrs_struct["spec"]["AvgDeltaF0"] = [0.0] * n_scans
    mrs_struct["out"]["reject"] = [np.empty(0, dtype=int)] * n_scans

    for ii in range(n_scans):
        mrs_struct["ii"] = ii
        m_path = metab_list[ii]
        w_path = water_list[ii] if (water_list and ii < len(water_list)) else None

        mrs_struct = _read_scan_pair(mrs_struct, m_path, w_path)
        if metabfile_join is not None:
            join_list = (
                [metabfile_join]
                if isinstance(metabfile_join, str)
                else list(metabfile_join)
            )
            if ii < len(join_list):
                tmp_struct = gannet_pre_initialise(config=config)
                tmp_struct["ii"] = 0
                tmp_struct["p"]["vendor"] = p["vendor"]
                tmp_struct["p"]["npoints"] = np.zeros(1, dtype=int)
                tmp_struct["p"]["nrows"] = np.zeros(1, dtype=int)
                tmp_struct["p"]["Navg"] = np.zeros(1, dtype=int)
                tmp_struct["p"]["Nwateravg"] = np.zeros(1, dtype=int)
                tmp_struct["p"]["LarmorFreq"] = np.zeros(1, dtype=float)
                tmp_struct["p"]["sw"] = np.zeros(1, dtype=float)
                tmp_struct["p"]["TR"] = np.zeros(1, dtype=float)
                tmp_struct["p"]["TE"] = np.zeros(1, dtype=float)
                tmp_struct["p"]["voxdim"] = np.zeros((1, 3), dtype=float)
                tmp_struct["p"]["voxoff"] = np.zeros((1, 3), dtype=float)
                tmp_struct["p"]["voxang"] = np.zeros((1, 3), dtype=float)
                tmp_struct["p"]["tlhc"] = np.zeros((1, 3), dtype=float)
                tmp_struct["p"]["trhc"] = np.zeros((1, 3), dtype=float)
                tmp_struct["p"]["brhc"] = np.zeros((1, 3), dtype=float)
                tmp_struct["p"]["TablePosition"] = np.zeros((1, 3), dtype=float)
                tmp_struct["p"]["NormSag"] = np.zeros(1, dtype=float)
                tmp_struct["p"]["NormCor"] = np.zeros(1, dtype=float)
                tmp_struct["p"]["NormTra"] = np.ones(1, dtype=float)
                tmp_struct["p"]["VoI_InPlaneRot"] = np.zeros(1, dtype=float)
                w_j = (
                    waterfile_join[ii]
                    if isinstance(waterfile_join, list) and ii < len(waterfile_join)
                    else (waterfile_join if isinstance(waterfile_join, str) else None)
                )
                tmp_struct = _read_scan_pair(tmp_struct, join_list[ii], w_j)
                mrs_struct["fids"]["data"] = np.concatenate(
                    [mrs_struct["fids"]["data"], tmp_struct["fids"]["data"]], axis=1
                )
                p["Navg"][ii] = mrs_struct["fids"]["data"].shape[1]

        if trim_avgs > 0:
            mrs_struct["fids"]["data"] = _trim_averages(
                mrs_struct["fids"]["data"], trim_avgs, bool(p.get("HERMES"))
            )
            p["Navg"][ii] = mrs_struct["fids"]["data"].shape[1]

        if bool(p.get("PRIAM")):
            mrs_struct = sense_recon(mrs_struct, ii)

        mrs_struct = specify_on_off_order(mrs_struct)

        sw = float(p["sw"][ii]) or 2000.0
        npts = int(p["npoints"][ii])
        larmor = float(p["LarmorFreq"][ii]) or 127.7
        zf = round(32768.0 / 2000.0 * sw)
        p["ZeroFillTo"][ii] = zf

        vox_names = p.get("Vox", ["vox1"])
        n_vox = int(p["numVox"])
        for kk in range(n_vox):
            vox = vox_names[kk] if kk < len(vox_names) else f"vox{kk + 1}"
            mrs_struct.ensure_substruct("spec", vox)

            data_fids = np.asarray(mrs_struct["fids"]["data"], dtype=np.complex128)
            if data_fids.ndim == 1:
                data_fids = data_fids[:, np.newaxis]

            # Water processing (ECC, LB, FFT)
            has_water = (
                "data_water" in mrs_struct["fids"]
                and mrs_struct["fids"]["data_water"] is not None
            )
            if has_water:
                water_fid = np.asarray(
                    mrs_struct["fids"]["data_water"], dtype=np.complex128
                )
                if water_fid.ndim > 1:
                    water_fid = np.mean(water_fid, axis=1)
                if bool(p.get("metab_ECC", False)):
                    data_fids = eddy_current_correction(data_fids, water_fid)
                if bool(p.get("water_ECC", True)):
                    water_fid = eddy_current_correction(
                        water_fid[:, np.newaxis], water_fid
                    )[:, 0]

                time_w = np.arange(water_fid.size, dtype=np.float64) / sw
                lb_w = 3.0
                water_lb = water_fid * np.exp(-time_w * lb_w * np.pi)
                water_zf = np.zeros(zf, dtype=np.complex128)
                water_zf[: min(zf, water_lb.size)] = water_lb[: min(zf, water_lb.size)]
                water_ft = np.fft.fftshift(np.fft.fft(water_zf))
                if p.get("vendor") == "Siemens_rda":
                    water_ft = -water_ft
                mrs_struct["spec"][vox]["water"] = _set_scan_row(
                    mrs_struct["spec"][vox].get("water"), ii, water_ft
                )

            if bool(p.get("phase_correction", True)):
                data_fids = phase_correction(data_fids, mrs_struct)

            mrs_struct["fids"]["data"] = data_fids
            time = np.arange(npts, dtype=np.float64) / sw
            lb_m = float(p.get("LB", 3.0))
            decay = np.exp(-time * lb_m * np.pi)[:, np.newaxis]
            data_lb = data_fids * decay
            all_frames_ft = np.fft.fftshift(np.fft.fft(data_lb, n=zf, axis=0), axes=0)

            if bool(p.get("phantom", False)):
                f0_ref = 4.8
            else:
                f0_ref = 4.68
            freq_range = sw / larmor
            # Standard Gannet freq axis runs high-to-low ppm centered on f0_ref
            freq = np.linspace(f0_ref + freq_range / 2.0, f0_ref - freq_range / 2.0, zf)
            mrs_struct["spec"]["freq"] = freq

            # Water frequency tracking across frames before alignment
            w_idx = (freq >= 4.2) & (freq <= 5.2)
            if np.any(w_idx):
                peak_bins = np.argmax(np.abs(all_frames_ft[w_idx, :]), axis=0)
                f0_curve = freq[w_idx][peak_bins]
                mrs_struct["spec"]["F0freq"][ii] = f0_curve
                mrs_struct["spec"]["AvgDeltaF0"][ii] = float(
                    np.mean(np.abs(f0_curve - f0_ref))
                )

            # Cr frequency tracking before alignment (for RobustSpecReg init)
            cr_idx = (freq >= 2.9) & (freq <= 3.15)
            if np.any(cr_idx):
                cr_bins = np.argmax(np.real(all_frames_ft[cr_idx, :]), axis=0)
                mrs_struct["spec"]["F0freq2"][ii] = freq[cr_idx][cr_bins]
            else:
                mrs_struct["spec"]["F0freq2"][ii] = np.full(
                    all_frames_ft.shape[1], 3.02
                )

            all_frames_ft_realign, mrs_struct = _run_frequency_phase_alignment(
                all_frames_ft, mrs_struct
            )
            mrs_struct["spec"]["AllFramesFT"] = all_frames_ft
            mrs_struct["spec"]["AllFramesFTrealign"] = all_frames_ft_realign

            # Refresh Cr frequency stability post-alignment
            if np.any(cr_idx):
                cr_bins = np.argmax(np.real(all_frames_ft_realign[cr_idx, :]), axis=0)
                mrs_struct["spec"]["F0freq2"][ii] = freq[cr_idx][cr_bins]

            mrs_struct = signal_averaging(
                mrs_struct, all_frames_ft, all_frames_ft_realign, ii, kk, vox_names
            )

            # Set diff / diff_noalign (with optional HSVD residual water removal)
            targets = list(p.get("target", ["GABA"]))
            for targ in targets:
                if targ not in mrs_struct["spec"][vox]:
                    continue
                t_spec = mrs_struct["spec"][vox][targ]
                diff_u = np.asarray(t_spec["diff_unfilt"], dtype=np.complex128)
                diff_na = np.asarray(t_spec["diff_unfilt_noalign"], dtype=np.complex128)
                diff_spec = (diff_u[ii] if diff_u.ndim == 2 else diff_u).copy()
                diff_na_spec = (diff_na[ii] if diff_na.ndim == 2 else diff_na).copy()

                if bool(p.get("water_removal", True)):
                    diff_fid = np.fft.ifft(np.fft.ifftshift(diff_spec))
                    cleaned_fid = water_removal_hsvd(
                        diff_fid,
                        sw_khz=sw / 1000.0,
                        use_points=min(2048, diff_fid.size),
                    )
                    cleaned_spec = np.fft.fftshift(np.fft.fft(cleaned_fid))
                    bl_mask = (freq >= 7.0) & (freq <= 8.0)
                    if np.any(bl_mask):
                        cleaned_spec = cleaned_spec - np.mean(
                            np.real(cleaned_spec[bl_mask])
                        )
                    diff_spec = cleaned_spec

                    diff_na_fid = np.fft.ifft(np.fft.ifftshift(diff_na_spec))
                    cleaned_na_fid = water_removal_hsvd(
                        diff_na_fid,
                        sw_khz=sw / 1000.0,
                        use_points=min(2048, diff_na_fid.size),
                    )
                    cleaned_na_spec = np.fft.fftshift(np.fft.fft(cleaned_na_fid))
                    if np.any(bl_mask):
                        cleaned_na_spec = cleaned_na_spec - np.mean(
                            np.real(cleaned_na_spec[bl_mask])
                        )
                    diff_na_spec = cleaned_na_spec

                t_spec["diff"] = _set_scan_row(t_spec.get("diff"), ii, diff_spec)
                t_spec["diff_noalign"] = _set_scan_row(
                    t_spec.get("diff_noalign"), ii, diff_na_spec
                )

    dest_dir = out_dir or os.getcwd()
    if bool(p.get("mat", False)):
        save_mrs_mat(mrs_struct, os.path.join(dest_dir, "MRS_struct.mat"))
    if save_html:
        export_to_html(
            mrs_struct,
            out_path=os.path.join(dest_dir, "GannetLoad_report.html"),
            title="GannetLoad Quality Control Report",
        )
    return mrs_struct


def _set_scan_row(
    existing: Any, ii: int, row_vec: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Store 1D spectrum ``row_vec`` at row ``ii`` of a 2D ``(n_scans, zf)`` array.

    Parameters
    ------------
    existing: Any
        Existing 2D array or ``None``.
    ii: int
        0-based scan row index.
    row_vec: NDArray[np.complex128]
        1D spectrum vector.

    Returns
    --------
        NDArray[np.complex128]:
            Updated 2D complex array.
    """
    n_pts = row_vec.size
    if existing is None or np.asarray(existing).size == 0:
        arr = np.zeros((ii + 1, n_pts), dtype=np.complex128)
    else:
        arr = np.asarray(existing, dtype=np.complex128)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[0] <= ii:
            pad = np.zeros((ii + 1 - arr.shape[0], arr.shape[1]), dtype=np.complex128)
            arr = np.vstack([arr, pad])
    arr[ii, : min(n_pts, arr.shape[1])] = row_vec[: arr.shape[1]]
    return arr
