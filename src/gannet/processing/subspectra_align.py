"""Subspectra alignment routines (AlignSubSpectra and PreAlignRef)."""

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct
from gannet.fitting.optimizers import lsqnonlin
from gannet.fitting.peak_fitters import fit_cho_cr


def _flatten_data(
    data: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split complex FIDs into real/imag tensor and compute real FFT."""
    flat = np.zeros((data.shape[0], 2, data.shape[1]), dtype=np.float64)
    flat[:, 0, :] = np.real(data)
    flat[:, 1, :] = np.imag(data)
    spec_real = np.real(np.fft.fftshift(np.fft.fft(data, axis=0), axes=0))
    result = (flat, spec_real)
    return result


def _weighted_subspec_avg(
    fids: NDArray[np.complex128],
    on_off: NDArray[np.int64],
    n_sub: int,
    t_max: int,
    water_flag: bool,
    hermes: bool,
) -> tuple[NDArray[np.complex128], list[NDArray[np.float64]], int]:
    """Compute MSE-weighted subspectra averages."""
    n_pts, n_trans = fids.shape
    if n_trans <= 4:
        return fids.copy(), [], 0

    data = np.zeros((n_pts, n_sub), dtype=np.complex128)
    weights_list: list[NDArray[np.float64]] = []
    n_per_sub = n_trans // n_sub
    data_lim = 0

    for jj in range(n_sub):
        if hermes:
            ind = np.arange(jj, n_trans, n_sub)
        else:
            ind = np.where(on_off[0] == abs(jj - 1))[0]

        dist = np.zeros((n_per_sub, n_per_sub), dtype=np.float64)
        for kk in range(n_per_sub):
            sig_k = np.real(fids[:t_max, ind[kk]])
            for ll in range(n_per_sub):
                sig_l = np.real(fids[:t_max, ind[ll]])
                dist[kk, ll] = float(np.mean((sig_k - sig_l) ** 2))

        dist[dist == 0] = np.nan
        d_vec = np.nanmean(dist, axis=0)
        d_vec = np.where(np.isnan(d_vec) | (d_vec == 0), 1.0, d_vec)
        w = 1.0 / (d_vec**2)
        w = w / np.sum(w)
        w_mat = np.tile(w, (n_pts, 1))
        weights_list.append(w_mat)

        if water_flag:
            data_lim = int(np.ceil(len(ind) / 3.0))
            data[:, jj] = np.sum(w_mat[:, :data_lim] * fids[:, ind[:data_lim]], axis=1)
        else:
            data[:, jj] = np.sum(w_mat * fids[:, ind], axis=1)

    result = (data, weights_list, data_lim)
    return result


def _get_hermes_subspec_ind(mrs_struct: MRSStruct) -> list[int]:
    """Return 0-based HERMES/HERCULES subexperiment index mapping."""
    p = mrs_struct["p"]
    targets = list(p["target"])
    vendor = str(p.get("vendor", ""))
    seqorig = str(p.get("seqorig", ""))
    hercules = bool(p.get("HERCULES", 0))

    if not hercules:
        if len(targets) == 2 and targets in (
            ["GABAGlx", "GSH"],
            ["GABA", "GSH"],
            ["Lac", "GSH"],
        ):
            if vendor == "GE":
                return [2, 1, 3, 0] if seqorig == "Lythgoe" else [2, 1, 0, 3]
            if vendor == "NIfTI":
                return [2, 1, 0, 3]
            if vendor in ("Philips", "Philips_data", "Philips_raw"):
                return [0, 1, 2, 3]
            if vendor in ("Siemens_dicom", "Siemens_rda", "Siemens_twix"):
                return [2, 0, 3, 1]
        elif len(targets) == 3 and targets == ["EtOH", "GABA", "GSH"]:
            if vendor == "GE":
                return [1, 0, 2, 3]
            if vendor in ("Philips", "Philips_data", "Philips_raw"):
                raise ValueError(
                    "HERMES of EtOH/GABA/GSH is not supported for Philips data."
                )
            if vendor in ("Siemens_dicom", "Siemens_rda", "Siemens_twix"):
                return [2, 0, 3, 1]
    else:
        if vendor in ("GE", "NIfTI"):
            return [2, 1, 0, 3]
        if vendor in ("Philips", "Philips_data", "Philips_raw"):
            return [0, 3, 2, 1]
        if vendor in ("Siemens_dicom", "Siemens_rda", "Siemens_twix"):
            return [2, 1, 0, 3]
    return [0, 1, 2, 3]


def _global_freq_phase_corr(
    data: NDArray[np.float64],
    fids: NDArray[np.complex128],
    freq: NDArray[np.float64],
    ind: int,
    t_vec: NDArray[np.float64],
    mrs_struct: MRSStruct,
) -> tuple[NDArray[np.complex128], float, float]:
    """Perform global frequency and phase shift based on Cho-Cr fit."""
    off_spec = data[:, ind]
    ii = int(mrs_struct["ii"])
    larmor = float(mrs_struct["p"]["LarmorFreq"][ii])

    freq_lim_cr = (freq <= 3.12) & (freq >= 2.92)
    idx_max = int(np.argmax(np.abs(off_spec[freq_lim_cr])))
    max_freq = float(freq[freq_lim_cr][idx_max])
    freq_lim = (freq <= max_freq + 0.58) & (freq >= max_freq - 0.42)
    off_seg = off_spec[freq_lim]
    baseline = (off_seg[0] + off_seg[-1]) / 2.0
    width = 0.05
    area = (np.max(off_seg) - np.min(off_seg)) * width * 4.0

    x0 = np.array(
        [area, width, max_freq, 0.0, baseline, 0.0, 1.0], dtype=np.float64
    ) * np.array([1.0, 2.0 * larmor, larmor, 180.0 / np.pi, 1.0, 1.0, 1.0])
    fit_res = fit_cho_cr(freq[freq_lim], off_seg, x0, larmor)
    model_param = fit_res["fit_params"]
    f_shift = float(model_param[2] - 3.02 * larmor)
    phi_shift = float(model_param[3])

    fids_out = (
        fids
        * np.exp(1j * np.pi / 180.0 * phi_shift)
        * np.exp(1j * f_shift * 2.0 * np.pi * t_vec[:, np.newaxis])
    )
    result = (fids_out, f_shift, phi_shift)
    return result


def _obj_func_subspec(
    in_pair: NDArray[np.float64],
    freq_lim: NDArray[np.bool_],
    t_vec: NDArray[np.float64],
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Frequency-domain difference objective for aligning two subspectra."""
    f_shift, phi_shift = float(x[0]), float(x[1])
    y = in_pair[:, 0, 0] + 1j * in_pair[:, 1, 0]
    y = y * np.exp(1j * np.pi * (t_vec * f_shift * 2.0 + phi_shift / 180.0))
    spec_a = np.real(np.fft.fftshift(np.fft.fft(y)))
    spec_b = np.real(
        np.fft.fftshift(np.fft.fft(in_pair[:, 0, 1] + 1j * in_pair[:, 1, 1]))
    )
    diff = spec_a - spec_b
    result: NDArray[np.float64] = diff[freq_lim]
    return result


def align_sub_spectra(
    mrs_struct: MRSStruct, water_flag: bool, lipid_ratio: float
) -> MRSStruct:
    """Align subspectra to each other after within-subspectrum registration.

    Mirrors ``AlignSubSpectra.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure with ``fids.data_align`` populated.
    water_flag: bool
        Whether unstable residual water was detected.
    lipid_ratio: float
        Estimated lipid-to-noise ratio ``r``.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with aligned ``fids.data_align``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    fids = np.asarray(mrs_struct["fids"]["data_align"], dtype=np.complex128).copy()
    on_off = np.asarray(mrs_struct["fids"]["ON_OFF"], dtype=np.int64)
    if on_off.ndim == 1:
        on_off = on_off[np.newaxis, :]

    n_pts, n_trans = fids.shape
    sw = float(p["sw"][ii])
    larmor = float(p["LarmorFreq"][ii])
    freq_range = sw / larmor
    freq = (
        (n_pts + 1 - np.arange(1, n_pts + 1)) / n_pts * freq_range
        + 4.68
        - freq_range / 2.0
    )
    t_vec = np.arange(n_pts, dtype=np.float64) / sw
    t_max = int(np.where(t_vec <= 0.1)[0][-1]) + 1
    hermes = bool(p.get("HERMES", 0))
    n_sub = 4 if hermes else 2

    data, weights_list, data_lim = _weighted_subspec_avg(
        fids, on_off, n_sub, t_max, water_flag, hermes
    )
    _, data_spec = _flatten_data(data)

    sub_spec_ind = _get_hermes_subspec_ind(mrs_struct) if hermes else [0, 1]
    ref_idx = sub_spec_ind[3] if hermes else 1
    fids, f_glob, phi_glob = _global_freq_phase_corr(
        data_spec, fids, freq, ref_idx, t_vec, mrs_struct
    )
    mrs_struct["out"]["SpecReg"]["freq"][ii] = (
        mrs_struct["out"]["SpecReg"]["freq"][ii] + f_glob
    )
    mrs_struct["out"]["SpecReg"]["phase"][ii] = (
        mrs_struct["out"]["SpecReg"]["phase"][ii] + phi_glob
    )

    if n_trans <= 4:
        data = fids.copy()
    else:
        for jj in range(n_sub):
            ind_j = (
                np.arange(jj, n_trans, 4)
                if hermes
                else np.where(on_off[0] == abs(jj - 1))[0]
            )
            w_mat = weights_list[jj]
            if water_flag:
                data[:, jj] = np.sum(
                    w_mat[:, :data_lim] * fids[:, ind_j[:data_lim]], axis=1
                )
            else:
                data[:, jj] = np.sum(w_mat * fids[:, ind_j], axis=1)

    flat_data, data_spec = _flatten_data(data)
    scale_a = float(np.max(flat_data)) or 1.0

    if hermes:
        # Residual water
        fl1 = (freq <= 4.68 + 0.22) & (freq >= 4.68 - 0.22)
        pair1 = data_spec[:, [sub_spec_ind[1], sub_spec_ind[3]]]
        idx1 = np.argmax(np.abs(pair1[fl1, :]), axis=0)
        mf1 = freq[fl1][idx1]
        fl1_union = ((freq <= mf1[0] + 0.22) & (freq >= mf1[0] - 0.22)) | (
            (freq <= mf1[1] + 0.22) & (freq >= mf1[1] - 0.22)
        )
        x0_1 = np.array([(mf1[0] - mf1[1]) * larmor, 0.0], dtype=np.float64)

        # NAA
        fl2 = (freq <= 2.01 + 0.13) & (freq >= 2.01 - 0.13)
        pair2 = data_spec[:, [sub_spec_ind[0], sub_spec_ind[3]]]
        idx2 = np.argmax(np.abs(pair2[fl2, :]), axis=0)
        mf2 = freq[fl2][idx2]
        fl2_union = ((freq <= mf2[0] + 0.18) & (freq >= mf2[0] - 0.18)) | (
            (freq <= mf2[1] + 0.18) & (freq >= mf2[1] - 0.18)
        )
        x0_2 = np.array([(mf2[0] - mf2[1]) * larmor, 0.0], dtype=np.float64)

        param1 = lsqnonlin(
            lambda x: _obj_func_subspec(
                flat_data[:, :, [sub_spec_ind[1], sub_spec_ind[3]]] / scale_a,
                fl1_union,
                t_vec,
                x,
            ),
            x0_1,
        )
        param2 = lsqnonlin(
            lambda x: _obj_func_subspec(
                flat_data[:, :, [sub_spec_ind[0], sub_spec_ind[3]]] / scale_a,
                fl2_union,
                t_vec,
                x,
            ),
            x0_2,
        )

        ind_s1 = np.arange(sub_spec_ind[0], n_trans, 4)
        fids[:, ind_s1] = (
            fids[:, ind_s1]
            * np.exp(1j * param2[0] * 2.0 * np.pi * t_vec[:, np.newaxis])
            * np.exp(1j * np.pi / 180.0 * param2[1])
        )
        mrs_struct["out"]["SpecReg"]["freq"][ii][ind_s1] += param2[0]
        mrs_struct["out"]["SpecReg"]["phase"][ii][ind_s1] += param2[1]

        for jj in range(4):
            ind_j = np.arange(jj, n_trans, 4)
            w_mat = weights_list[jj]
            if water_flag:
                data[:, jj] = np.sum(
                    w_mat[:, :data_lim] * fids[:, ind_j[:data_lim]], axis=1
                )
            else:
                data[:, jj] = np.sum(w_mat * fids[:, ind_j], axis=1)
        flat_data, data_spec = _flatten_data(data)

        # Cho
        fl3 = (freq <= 3.2 + 0.09) & (freq >= 3.2 - 0.09)
        pair3 = data_spec[:, [sub_spec_ind[2], sub_spec_ind[0]]]
        idx3 = np.argmax(np.abs(pair3[fl3, :]), axis=0)
        mf3 = freq[fl3][idx3]
        fl3_union = ((freq <= mf3[0] + 0.08) & (freq >= mf3[0] - 0.08)) | (
            (freq <= mf3[1] + 0.08) & (freq >= mf3[1] - 0.08)
        )
        x0_3 = np.array([(mf3[0] - mf3[1]) * larmor, 0.0], dtype=np.float64)
        param3 = lsqnonlin(
            lambda x: _obj_func_subspec(
                flat_data[:, :, [sub_spec_ind[2], sub_spec_ind[0]]] / scale_a,
                fl3_union,
                t_vec,
                x,
            ),
            x0_3,
        )

        ind_s2 = np.arange(sub_spec_ind[1], n_trans, 4)
        ind_s3 = np.arange(sub_spec_ind[2], n_trans, 4)
        fids[:, ind_s2] = (
            fids[:, ind_s2]
            * np.exp(1j * param1[0] * 2.0 * np.pi * t_vec[:, np.newaxis])
            * np.exp(1j * np.pi / 180.0 * param1[1])
        )
        fids[:, ind_s3] = (
            fids[:, ind_s3]
            * np.exp(1j * param3[0] * 2.0 * np.pi * t_vec[:, np.newaxis])
            * np.exp(1j * np.pi / 180.0 * param3[1])
        )
        mrs_struct["out"]["SpecReg"]["freq"][ii][ind_s2] += param1[0]
        mrs_struct["out"]["SpecReg"]["phase"][ii][ind_s2] += param1[1]
        mrs_struct["out"]["SpecReg"]["freq"][ii][ind_s3] += param3[0]
        mrs_struct["out"]["SpecReg"]["phase"][ii][ind_s3] += param3[1]
    else:
        target0 = p["target"][0]
        if target0 in ("GABAGlx", "GABA", "Glx", "Lac", "EtOH"):
            max_naa = float(
                np.max(np.abs(data_spec[(freq <= 4.25) & (freq >= 1.8), 1]))
            )
            max_water = float(
                np.max(
                    np.abs(data_spec[(freq <= 4.68 + 0.22) & (freq >= 4.68 - 0.22), 1])
                )
            )
            if lipid_ratio < 70.0 and (max_naa > 0 and max_water / max_naa < 1.5):
                win_lim = (freq <= 3.2 + 0.09) & (freq >= 3.2 - 0.09)
                half_w = 0.08
            else:
                win_lim = (freq <= 4.68 + 0.22) & (freq >= 4.68 - 0.22)
                half_w = 0.22
        else:
            win_lim = (freq <= 2.01 + 0.13) & (freq >= 2.01 - 0.13)
            half_w = 0.18

        idx_max = np.argmax(np.abs(data_spec[win_lim, :]), axis=0)
        mf = freq[win_lim][idx_max]
        freq_lim_union = ((freq <= mf[0] + half_w) & (freq >= mf[0] - half_w)) | (
            (freq <= mf[1] + half_w) & (freq >= mf[1] - half_w)
        )
        x0 = np.array([(mf[0] - mf[1]) * larmor, 0.0], dtype=np.float64)

        param = lsqnonlin(
            lambda x: _obj_func_subspec(flat_data / scale_a, freq_lim_union, t_vec, x),
            x0,
        )
        ind_on = np.where(on_off[0] == 1)[0]
        fids[:, ind_on] = (
            fids[:, ind_on]
            * np.exp(1j * param[0] * 2.0 * np.pi * t_vec[:, np.newaxis])
            * np.exp(1j * np.pi / 180.0 * param[1])
        )
        mrs_struct["out"]["SpecReg"]["freq"][ii][ind_on] += param[0]
        mrs_struct["out"]["SpecReg"]["phase"][ii][ind_on] += param[1]

    mrs_struct["fids"]["data_align"] = fids
    return mrs_struct


def align_sub_spectra_pre_align_ref(
    mrs_struct: MRSStruct, water_flag: bool
) -> MRSStruct:
    """Align subspectra using pre-aligned subspectra as references.

    Mirrors ``AlignSubSpectra_PreAlignRef.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure containing ``fids.data`` and ``fids.data_align``.
    water_flag: bool
        Whether unstable residual water was detected.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with aligned ``fids.data_align``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    fids = np.asarray(mrs_struct["fids"]["data_align"], dtype=np.complex128).copy()
    fids_ref = np.asarray(mrs_struct["fids"]["data"], dtype=np.complex128)
    on_off = np.asarray(mrs_struct["fids"]["ON_OFF"], dtype=np.int64)
    if on_off.ndim == 1:
        on_off = on_off[np.newaxis, :]

    n_pts, n_trans = fids.shape
    sw = float(p["sw"][ii])
    larmor = float(p["LarmorFreq"][ii])
    freq_range = sw / larmor
    freq = (
        (n_pts + 1 - np.arange(1, n_pts + 1)) / n_pts * freq_range
        + 4.68
        - freq_range / 2.0
    )
    t_vec = np.arange(n_pts, dtype=np.float64) / sw
    t_max = int(np.where(t_vec <= 0.1)[0][-1]) + 1
    hermes = bool(p.get("HERMES", 0))
    n_sub = 4 if hermes else 2

    data, _, _ = _weighted_subspec_avg(fids, on_off, n_sub, t_max, water_flag, hermes)
    data_ref, _, _ = _weighted_subspec_avg(
        fids_ref, on_off, n_sub, t_max, water_flag, hermes
    )

    flat_data, _ = _flatten_data(data)
    ref_flat, _ = _flatten_data(data_ref)
    scale_a = float(np.max(flat_data)) or 1.0
    freq_lim = (freq <= 3.4) & (freq >= 1.8)

    if hermes:
        sub_spec_ind = _get_hermes_subspec_ind(mrs_struct)
        for idx_pos in range(4):
            in_pair = np.zeros((n_pts, 2, 2), dtype=np.float64)
            in_pair[:, :, 0] = flat_data[:, :, sub_spec_ind[idx_pos]] / scale_a
            in_pair[:, :, 1] = ref_flat[:, :, sub_spec_ind[idx_pos]] / scale_a
            param = lsqnonlin(
                lambda x, arr=in_pair: _obj_func_subspec(arr, freq_lim, t_vec, x),
                np.array([0.0, 0.0], dtype=np.float64),
            )
            ind_s = np.arange(sub_spec_ind[idx_pos], n_trans, 4)
            fids[:, ind_s] = (
                fids[:, ind_s]
                * np.exp(1j * param[0] * 2.0 * np.pi * t_vec[:, np.newaxis])
                * np.exp(1j * np.pi / 180.0 * param[1])
            )
            mrs_struct["out"]["SpecReg"]["freq"][ii][ind_s] += param[0]
            mrs_struct["out"]["SpecReg"]["phase"][ii][ind_s] += param[1]
    else:
        for idx_pos, val in enumerate([1, 0]):
            in_pair = np.zeros((n_pts, 2, 2), dtype=np.float64)
            in_pair[:, :, 0] = flat_data[:, :, idx_pos] / scale_a
            in_pair[:, :, 1] = ref_flat[:, :, idx_pos] / scale_a
            param = lsqnonlin(
                lambda x, arr=in_pair: _obj_func_subspec(arr, freq_lim, t_vec, x),
                np.array([0.0, 0.0], dtype=np.float64),
            )
            ind_s = np.where(on_off[0] == val)[0]
            fids[:, ind_s] = (
                fids[:, ind_s]
                * np.exp(1j * param[0] * 2.0 * np.pi * t_vec[:, np.newaxis])
                * np.exp(1j * np.pi / 180.0 * param[1])
            )
            mrs_struct["out"]["SpecReg"]["freq"][ii][ind_s] += param[0]
            mrs_struct["out"]["SpecReg"]["phase"][ii][ind_s] += param[1]

    mrs_struct["fids"]["data_align"] = fids
    return mrs_struct
