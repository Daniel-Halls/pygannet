"""Philips .raw/.lab/.sin (PRIAM and single-voxel) MRS reader for Gannet."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct
from gannet.priam.cpx import get_sin_info, load_ref_scan, read_lab_raw_labels
from gannet.priam.sense import calc_unfolding_matrix


def _downsample_philips_fir(
    fids: NDArray[np.complex128],
    noversamples: int,
    npoints: int,
) -> NDArray[np.complex128]:
    """Apply Philips two-stage decimation + 12-tap FIR downsampling filter.

    Parameters
    ------------
    fids: NDArray[np.complex128]
        Oversampled coil FID array shaped ``(n_coils, noversamples, n_mixes, n_avg)``.
    noversamples: int
        Number of oversampled samples per FID.
    npoints: int
        Target downsampled sample count.

    Returns
    --------
        NDArray[np.complex128]:
            Downsampled array shaped ``(n_coils, npoints, n_mixes, n_avg)``.
    """
    if noversamples <= npoints or npoints <= 0:
        return fids[:, :npoints, :, :] if npoints > 0 else fids

    coeffs = np.array(
        [
            -2.74622659936243e-2,
            -3.16256052372452e-2,
            6.00081666037503e-3,
            9.10571641138698e-2,
            1.94603313644247e-1,
            2.66010793957549e-1,
            2.66010793957549e-1,
            1.94603313644247e-1,
            9.10571641138698e-2,
            6.00081666037503e-3,
            -3.16256052372452e-2,
            -2.74622659936243e-2,
        ],
        dtype=np.float64,
    )
    r_fir = 4
    r_hdf = max(1, round(noversamples / npoints / r_fir))
    n_mid = noversamples // r_hdf
    n_coils, _, n_mixes, n_avg = fids.shape
    trimmed = fids[:, : n_mid * r_hdf, :, :]
    h_out = np.sum(
        trimmed.reshape(n_coils, r_hdf, n_mid, n_mixes, n_avg, order="F"),
        axis=1,
    )
    out = np.zeros((n_coils, npoints, n_mixes, n_avg), dtype=np.complex128)
    n_tabs = len(coeffs)
    n_delay = n_tabs // 2
    n_arr = np.arange(npoints)
    n1 = n_arr * r_fir + n_delay
    for k in range(n_tabs):
        n2 = n1 - k
        valid = (n2 >= 0) & (n2 < n_mid)
        out[:, n_arr[valid], :, :] += coeffs[k] * h_out[:, n2[valid], :, :]
    return out


def _read_lab_raw_fids(raw_path: str) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Read normal STD FID records from a Philips ``.lab/.raw`` file pair.

    Parameters
    ------------
    raw_path: str
        Path to ``.raw`` or ``.lab`` file.

    Returns
    --------
        tuple[NDArray[np.complex128], dict[str, Any]]:
            Complex array ``(n_coils, n_pts, n_mixes, n_avg)`` and decoded
            label metadata dict.
    """
    info = read_lab_raw_labels(raw_path)
    n_idx = info["normal_idx"]
    if len(n_idx) == 0:
        raise ValueError(f"No NORMAL_DATA labels found in {raw_path}")

    n_coils = max(1, int(info["n_coils"]))
    max_size = int(np.max(info["data_size"][n_idx]))
    n_kx = max_size // (4 * n_coils)

    mix_vals = sorted({int(v) for v in info["mix"][n_idx]})
    row_vals = sorted({int(v) for v in info["row"][n_idx]})
    meas_vals = sorted({int(v) for v in info["meas"][n_idx]})
    mix_map = {v: i for i, v in enumerate(mix_vals)}
    avg_keys = sorted({(int(info["meas"][k]), int(info["row"][k])) for k in n_idx})
    avg_map = {key: i for i, key in enumerate(avg_keys)}

    out = np.zeros((n_coils, n_kx, len(mix_vals), len(avg_keys)), dtype=np.complex128)
    with open(info["raw_path"], "rb") as fid:
        for idx in n_idx:
            offset = int(info["offsets"][idx])
            if offset < 0:
                continue
            d_size = int(info["data_size"][idx])
            fid.seek(offset, os.SEEK_SET)
            raw_i16 = np.fromfile(fid, dtype="<i2", count=d_size // 2).astype(
                np.float64
            )
            if raw_i16.size < 2 * n_coils * n_kx:
                continue
            cdata = (
                (raw_i16[0::2] + 1j * raw_i16[1::2])
                .reshape((n_kx, n_coils), order="F")
                .T
            )
            m_i = mix_map[int(info["mix"][idx])]
            a_i = avg_map[(int(info["meas"][idx]), int(info["row"][idx]))]
            out[:, :, m_i, a_i] = cdata

    meta = {
        "info": info,
        "n_coils": n_coils,
        "n_kx": n_kx,
        "n_mixes": len(mix_vals),
        "n_rows": len(row_vals),
        "n_measurements": len(meas_vals),
        "n_averages": len(avg_keys),
    }
    return (out, meta)


def philips_raw_read(
    mrs_struct: MRSStruct,
    raw_file: str,
    ref_cpx_file: str | None = None,
    recon_voxel: int = 3,
    editing: bool = True,
) -> MRSStruct:
    """Read Philips ``.raw/.lab/.sin`` PRIAM or single-voxel data into ``MRSStruct``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` with ``ii`` set to 0-based scan index.
    raw_file: str
        Path to metabolite ``.raw`` or ``.lab`` file.
    ref_cpx_file: str | None
        Optional SENSE reference ``.cpx`` path for PRIAM dual-voxel unfolding.
    recon_voxel: int
        ``1`` or ``2`` for single reconstructed PRIAM voxel; ``3`` (default)
        for both voxels.
    editing: bool
        Whether the acquisition uses spectral editing (keeps individual averages).

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with unfolded or combined FIDs and metadata.
    """
    ii = int(mrs_struct["ii"])
    fids, _meta = _read_lab_raw_fids(raw_file)

    prefix = re.sub(r"\.(raw|lab|sin)$", "", raw_file, flags=re.IGNORECASE)
    sin_path = f"{prefix}.sin"
    samples_info = get_sin_info(sin_path, "samples")
    noversamples = int(fids.shape[1])
    npoints = int(samples_info[-1]) if samples_info else noversamples
    if noversamples > npoints > 0:
        fids = _downsample_philips_fir(fids, noversamples, npoints)

    vox_sizes = get_sin_info(sin_path, "voxel_sizes")
    vox_off = get_sin_info(sin_path, "loc_ap_rl_fh_offcentres")
    vox_ang = get_sin_info(sin_path, "loc_ap_rl_fh_angs")
    tr_vals = get_sin_info(sin_path, "repetition_times")
    te_vals = get_sin_info(sin_path, "echo_times")

    if len(vox_sizes) >= 3:
        mrs_struct["p"]["voxdim"][ii] = [vox_sizes[1], vox_sizes[0], vox_sizes[2]]
    if len(vox_off) >= 3:
        mrs_struct["p"]["voxoff"][ii] = [vox_off[1], vox_off[0], vox_off[2]]
    if len(vox_ang) >= 3:
        mrs_struct["p"]["voxang"][ii] = [vox_ang[1], vox_ang[0], vox_ang[2]]
    if tr_vals:
        mrs_struct["p"]["TR"][ii] = float(tr_vals[0])
    if te_vals:
        mrs_struct["p"]["TE"][ii] = float(te_vals[0])

    # PRIAM SENSE unfolding when CPX reference is supplied
    if ref_cpx_file is not None and os.path.exists(ref_cpx_file):
        ref = load_ref_scan(ref_cpx_file, raw_file)
        sens = ref["sensitivities"]
        nx, ny, nz = sens.shape[1:]
        centers = [(nx // 3, ny // 2, nz // 2), (2 * nx // 3, ny // 2, nz // 2)]
        if recon_voxel == 1:
            centers = [centers[0]]
        elif recon_voxel == 2:
            centers = [centers[1]]
        unfold = calc_unfolding_matrix(sens, centers, noise=ref["noise"])
        u_mat = unfold["U"]
        n_vox = u_mat.shape[0]
        _, n_pts, n_mixes, n_avg = fids.shape
        sig_unf = np.zeros((n_vox, n_pts, n_mixes, n_avg), dtype=np.complex128)
        for m in range(n_mixes):
            for a in range(n_avg):
                sig_unf[:, :, m, a] = u_mat @ fids[:, :, m, a]

        # Reorder PRIAM interleaved ON/OFF averages matching PhilipsRawRead.m
        if editing and n_avg >= 4:
            sig_unf[:, :, :, 1::2] = -sig_unf[:, :, :, 1::2]
            half = n_avg // 2
            a_part = sig_unf[:, :, :, :half]
            b_part = sig_unf[:, :, :, half : 2 * half]
            reord = np.zeros_like(sig_unf)
            reord[:, :, :, 0 : 2 * half : 2] = a_part
            reord[:, :, :, 1 : 2 * half : 2] = b_part
            sig_unf = reord

        mrs_struct["p"]["NVoxels"] = n_vox
        mrs_struct["p"]["gfactors"] = unfold["g_factors"].tolist()
        mrs_struct["multivoxel"]["sensitivities"] = sens
        mrs_struct["multivoxel"]["allsignals"] = sig_unf
        mrs_struct["p"]["npoints"][ii] = sig_unf.shape[1]
        mrs_struct["p"]["Navg"][ii] = sig_unf.shape[3]
        mrs_struct["fids"]["data"] = np.conj(sig_unf[0, :, 0, :])
        if n_mixes > 1:
            water_slice = sig_unf[0, :, 1, :]
            nonzero = np.any(np.abs(water_slice) > 0, axis=0)
            w_valid = water_slice[:, nonzero] if np.any(nonzero) else water_slice
            mrs_struct["p"]["Nwateravg"][ii] = w_valid.shape[1]
            mrs_struct["fids"]["data_water"] = np.mean(np.conj(w_valid), axis=1)
        return mrs_struct

    # Single-voxel coil combination fallback when no CPX reference
    metab = fids[:, :, 0, :]
    weights = np.mean(metab[:, : min(10, metab.shape[1]), :], axis=(1, 2))
    wnorm = np.linalg.norm(weights)
    if wnorm > 0:
        weights = weights / wnorm
    else:
        weights = np.ones(fids.shape[0], dtype=np.complex128) / np.sqrt(fids.shape[0])
    comb = np.tensordot(weights.conj(), metab, axes=([0], [0]))
    mrs_struct["p"]["npoints"][ii] = comb.shape[0]
    mrs_struct["p"]["nrows"][ii] = comb.shape[1]
    mrs_struct["p"]["Navg"][ii] = comb.shape[1]
    mrs_struct["fids"]["data"] = np.conj(comb)

    if fids.shape[2] > 1:
        water_m = fids[:, :, 1, :]
        w_comb = np.tensordot(weights.conj(), water_m, axes=([0], [0]))
        nonzero = np.any(np.abs(w_comb) > 0, axis=0)
        w_valid = w_comb[:, nonzero] if np.any(nonzero) else w_comb
        mrs_struct["p"]["Nwateravg"][ii] = w_valid.shape[1]
        mrs_struct["fids"]["data_water"] = np.mean(np.conj(w_valid), axis=1)

    return mrs_struct
