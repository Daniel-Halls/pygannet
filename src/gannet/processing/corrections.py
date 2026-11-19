"""Eddy current correction and global zero-order phase correction."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gannet.config import MRSStruct
from gannet.fitting.peak_fitters import fit_cho_cr


def eddy_current_correction(
    data: ArrayLike, water_data: ArrayLike
) -> NDArray[np.complex128]:
    """Perform Klose eddy current phase correction using unsuppressed water.

    Mirrors ``EddyCurrentCorrection.m``.

    Parameters
    ------------
    data: ArrayLike
        1D or 2D time-domain FID array ``(npoints,)`` or
        ``(npoints, n_averages)``.
    water_data: ArrayLike
        1D unsuppressed water reference FID ``(npoints,)``.

    Returns
    --------
        NDArray[np.complex128]: complex numpy array
            Eddy-current-corrected complex FIDs with matching shape.
    """
    data_arr = np.asarray(data, dtype=np.complex128)
    was_1d = data_arr.ndim == 1
    if was_1d:
        data_2d = data_arr[:, np.newaxis]
    else:
        data_2d = data_arr

    water_1d = np.asarray(water_data, dtype=np.complex128).ravel()
    amp = np.abs(data_2d)
    phase_data = np.unwrap(np.angle(data_2d), axis=0)
    phase_water = np.unwrap(np.angle(water_1d), axis=0)
    phase_corr = phase_data - phase_water[:, np.newaxis]
    corrected = (amp * np.exp(1j * phase_corr)).astype(np.complex128)
    if was_1d:
        return corrected[:, 0]
    return corrected


def _compute_weighted_subspectra(
    fids: NDArray[np.complex128],
    on_off: NDArray[np.int64],
    n_sub: int,
    t_max: int,
    water_flag: bool,
    hermes: bool,
) -> tuple[NDArray[np.complex128], list[NDArray[np.float64]]]:
    """Compute MSE-weighted averaged subspectra for global phase estimation.

    Parameters
    ------------
    fids: NDArray[np.complex128]
        Time-domain FID matrix of shape ``(npoints, n_transients)``.
    on_off: NDArray[np.int64]
        ON/OFF indicator array.
    n_sub: int
        Number of subspectra (2 for MEGA-PRESS, 4 for HERMES).
    t_max: int
        Number of initial time points used for MSE similarity calculation.
    water_flag: bool
        Whether unstable residual water was detected.
    hermes: bool
        Whether data were acquired using HERMES.

    Returns
    --------
        tuple[NDArray[np.complex128], list[NDArray[np.float64]]]:
            Tuple containing ``(averaged_data, weight_matrices)``.
    """
    n_pts, n_trans = fids.shape
    if n_trans <= 4:
        return fids.copy(), []

    data = np.zeros((n_pts, n_sub), dtype=np.complex128)
    weights_list: list[NDArray[np.float64]] = []
    n_per_sub = n_trans // n_sub

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

    result = (data, weights_list)
    return result


def _apply_weights_with_stored(
    fids: NDArray[np.complex128],
    on_off: NDArray[np.int64],
    n_sub: int,
    water_flag: bool,
    hermes: bool,
    weights_list: list[NDArray[np.float64]],
) -> NDArray[np.complex128]:
    """Recompute weighted subspectra using precomputed weights.

    Parameters
    ------------
    fids: NDArray[np.complex128]
        Updated FID matrix.
    on_off: NDArray[np.int64]
        ON/OFF indicator array.
    n_sub: int
        Number of subspectra.
    water_flag: bool
        Whether unstable residual water was detected.
    hermes: bool
        Whether HERMES mode is active.
    weights_list: list[NDArray[np.float64]]
        Precomputed weight matrices per subspectrum.

    Returns
    --------
        NDArray[np.complex128]: 2D complex numpy array
            Weighted subspectra of shape ``(npoints, n_sub)``.
    """
    n_pts, n_trans = fids.shape
    if n_trans <= 4 or not weights_list:
        return fids.copy()

    data = np.zeros((n_pts, n_sub), dtype=np.complex128)
    for jj in range(n_sub):
        if hermes:
            ind = np.arange(jj, n_trans, 4)
        else:
            ind = np.where(on_off[0] == abs(jj - 1))[0]
        w_mat = weights_list[jj]
        if water_flag:
            data_lim = int(np.ceil(len(ind) / 3.0))
            data[:, jj] = np.sum(w_mat[:, :data_lim] * fids[:, ind[:data_lim]], axis=1)
        else:
            data[:, jj] = np.sum(w_mat * fids[:, ind], axis=1)
    return data


def phase_correction(fids: ArrayLike, mrs_struct: MRSStruct) -> NDArray[np.complex128]:
    """Apply global zero-order phase correction using the Cho-Cr region.

    Mirrors ``PhaseCorrection.m``.

    Parameters
    ------------
    fids: ArrayLike
        Time-domain FID array of shape ``(npoints, n_transients)``.
    mrs_struct: MRSStruct
        MRS data structure containing acquisition parameters and ``ON_OFF``.

    Returns
    --------
        NDArray[np.complex128]: 2D complex numpy array
            Globally phased FIDs.
    """
    fids_arr = np.asarray(fids, dtype=np.complex128).copy()
    p = mrs_struct["p"]
    ii = int(mrs_struct["ii"])
    npoints = int(p["npoints"][ii])
    sw = float(p["sw"][ii])
    larmor = float(p["LarmorFreq"][ii])
    hermes = bool(p.get("HERMES", 0))
    on_off = np.asarray(mrs_struct["fids"]["ON_OFF"], dtype=np.int64)
    if on_off.ndim == 1:
        on_off = on_off[np.newaxis, :]

    freq_range = sw / larmor
    freq = (
        (npoints + 1 - np.arange(1, npoints + 1)) / npoints * freq_range
        + 4.68
        - freq_range / 2.0
    )
    water_lim = (freq <= 4.68 + 0.25) & (freq >= 4.68 - 0.25)
    spec = np.fft.fftshift(np.fft.fft(fids_arr, axis=0), axes=0)

    if hermes:
        ind_w = np.all(on_off.T == 0, axis=1)
    elif p["target"][0] == "GSH":
        ind_w = on_off[0] == 0
    else:
        ind_w = np.arange(fids_arr.shape[1])

    q = np.sum(np.abs(spec[np.ix_(water_lim, ind_w)]), axis=0) * abs(freq[0] - freq[1])
    med_q = np.median(q)
    q_pct = (q - med_q) / med_q * 100.0 if med_q != 0 else np.zeros_like(q)
    water_flag = bool(np.sum(np.abs(q_pct) > 40.0) / len(q_pct) > 0.1)

    n_sub = 4 if hermes else 2
    time_vec = np.arange(npoints, dtype=np.float64) / sw
    t_max = int(np.where(time_vec <= 0.1)[0][-1]) + 1

    data, weights_list = _compute_weighted_subspectra(
        fids_arr, on_off, n_sub, t_max, water_flag, hermes
    )

    sum_spec = np.mean(
        np.real(np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)), axis=1
    )
    freq_lim_cr = (freq <= 3.12) & (freq >= 2.92)
    idx_max = int(np.argmax(np.abs(sum_spec[freq_lim_cr])))
    max_freq = float(freq[freq_lim_cr][idx_max])
    freq_lim = (freq <= max_freq + 0.58) & (freq >= max_freq - 0.42)
    sum_cho_cr = sum_spec[freq_lim]

    if np.sign(sum_cho_cr[int(np.argmax(np.abs(sum_cho_cr)))]) < 0:
        fids_arr = -fids_arr
        data = -data
        sum_spec = np.mean(
            np.real(np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)), axis=1
        )
        sum_cho_cr = sum_spec[freq_lim]

    baseline = (sum_cho_cr[0] + sum_cho_cr[-1]) / 2.0
    width = 0.05
    area = (np.max(sum_cho_cr) - np.min(sum_cho_cr)) * width * 4.0
    x0 = np.array(
        [area, width, max_freq, 0.0, baseline, 0.0, 1.0], dtype=np.float64
    ) * np.array(
        [1.0, 2.0 * larmor, larmor, 180.0 / np.pi, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    fit_res = fit_cho_cr(freq[freq_lim], sum_cho_cr, x0, larmor)
    fids_arr = fids_arr * np.exp(1j * np.pi / 180.0 * float(fit_res["fit_params"][3]))

    data = _apply_weights_with_stored(
        fids_arr, on_off, n_sub, water_flag, hermes, weights_list
    )
    sum_spec = np.mean(
        np.real(np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)), axis=1
    )
    sum_cho_cr = sum_spec[freq_lim]
    sum_cho_cr = sum_cho_cr - (sum_cho_cr[0] + sum_cho_cr[-1]) / 2.0
    if np.sign(sum_cho_cr[int(np.argmax(np.abs(sum_cho_cr)))]) < 0:
        fids_arr = -fids_arr
        data = -data
        sum_spec = np.mean(
            np.real(np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)), axis=1
        )
        sum_cho_cr = sum_spec[freq_lim]

    baseline = (sum_cho_cr[0] + sum_cho_cr[-1]) / 2.0
    area = (np.max(sum_cho_cr) - np.min(sum_cho_cr)) * width * 4.0
    x0 = np.array(
        [area, width, max_freq, 0.0, baseline, 0.0, 1.0], dtype=np.float64
    ) * np.array(
        [1.0, 2.0 * larmor, larmor, 180.0 / np.pi, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    fit_res2 = fit_cho_cr(freq[freq_lim], sum_cho_cr, x0, larmor)
    fids_arr = fids_arr * np.exp(1j * np.pi / 180.0 * float(fit_res2["fit_params"][3]))
    return fids_arr
