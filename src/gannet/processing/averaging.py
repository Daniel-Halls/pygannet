"""Signal averaging and outlier weighting routines."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gannet.config import MRSStruct


def wacfm(x: ArrayLike, cost_fun: str = "GCD") -> dict[str, NDArray[np.float64]]:
    """Compute weighted averaging based on criterion function minimization.

    Implements Pander (2015), Biocybern Biomed Eng. 35(4):317-327.

    Parameters
    ------------
    x: ArrayLike
        2D array of real spectra with shape ``(n_freq, n_transients)``.
    cost_fun: str
        Cost function type: ``'square'`` or ``'GCD'``. Defaults to ``'GCD'``.

    Returns
    --------
        dict[str, NDArray[np.float64]]: dict object
            Dictionary containing ``'v'`` (averaged spectrum) and ``'w'``
            (normalized transient weights).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    n_trans = x_arr.shape[1]
    k_stop = 100
    v = np.median(x_arr, axis=1)
    w_hist = np.zeros((k_stop, n_trans), dtype=np.float64)
    tol = 1e-6
    m = 2.0
    p = 0.2
    sigma = 1.0
    const = float(np.max(np.abs(v)))

    def compute_weights(z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute normalized weights for residual matrix ``z``."""
        if cost_fun == "square":
            norms = np.linalg.norm(z, axis=0) + const
            raw = norms ** (2.0 / (1.0 - m))
        else:
            val = np.sum(np.log(1.0 + (np.abs(z) / sigma) ** p), axis=0) + const
            raw = val ** (1.0 / (1.0 - m))
        result: NDArray[np.float64] = raw / np.sum(raw)
        return result

    final_w = np.ones(n_trans, dtype=np.float64) / max(n_trans, 1)
    for k in range(k_stop):
        z = x_arr - v[:, np.newaxis]
        w_hist[k, :] = compute_weights(z)
        if (
            k > 0 and np.linalg.norm(w_hist[k, :] - w_hist[k - 1, :]) < tol
        ) or k == k_stop - 1:
            wk = w_hist[k, :]
            if cost_fun == "square":
                final_w = (wk**m) / np.sum(wk**m)
            else:
                gcd_term = (
                    np.sum(
                        (np.abs(z) ** (2.0 - p)) / p * (sigma**p + np.abs(z) ** p),
                        axis=0,
                    )
                    ** -1
                )
                num = (wk**m) * gcd_term
                final_w = num / np.sum(num)
            break

        wk = w_hist[k, :]
        if cost_fun == "square":
            v = np.sum((wk**m)[np.newaxis, :] * x_arr, axis=1) / np.sum(wk**m)
        else:
            gcd_term = (
                np.sum(
                    (np.abs(z) ** (2.0 - p)) / p * (sigma**p + np.abs(z) ** p),
                    axis=0,
                )
                ** -1
            )
            num = np.sum(((wk**m) * gcd_term)[np.newaxis, :] * x_arr, axis=1)
            den = np.sum((wk**m) * gcd_term)
            v = num / den

    result: dict[str, NDArray[np.float64]] = {"v": v, "w": final_w}
    return result


def _ensure_spec_matrix(
    container: MRSStruct,
    field_name: str,
    num_scans: int,
    n_zf: int,
) -> NDArray[np.complex128]:
    """Ensure a 2D complex spectrum matrix exists in ``container``."""
    if field_name not in container or container[field_name].shape[0] < num_scans:
        mat = np.zeros((num_scans, n_zf), dtype=np.complex128)
        if field_name in container:
            old = container[field_name]
            mat[: old.shape[0], : old.shape[1]] = old
        container[field_name] = mat
    arr: NDArray[np.complex128] = container[field_name]
    return arr


def _compute_mse_weights(
    all_frames_ft_realign: NDArray[np.complex128],
    ind: NDArray[np.int64],
    npoints: int,
    freq_lim: NDArray[np.bool_],
    method: str,
) -> NDArray[np.float64]:
    """Compute transient averaging weights for a single subspectrum."""
    fids_tmp = np.fft.ifft(
        np.fft.ifftshift(all_frames_ft_realign[:, ind], axes=0), axis=0
    )
    spec = np.fft.fftshift(np.fft.fft(fids_tmp[:npoints, :], axis=0), axes=0)
    real_spec = np.real(spec[freq_lim, :])

    if method == "MSE":
        n_sub_t = ind.size
        dist = np.zeros((n_sub_t, n_sub_t), dtype=np.float64)
        for ll in range(n_sub_t):
            for mm in range(n_sub_t):
                dist[ll, mm] = float(
                    np.mean((real_spec[:, ll] - real_spec[:, mm]) ** 2)
                )
        dist[dist == 0] = np.nan
        d_vec = np.nanmedian(dist, axis=0)
        d_vec = np.where(np.isnan(d_vec) | (d_vec == 0), 1.0, d_vec)
        inv_sq = d_vec**-2
        w: NDArray[np.float64] = inv_sq / np.sum(inv_sq)
        return w
    if method == "MSE2":
        med = np.median(real_spec, axis=1, keepdims=True)
        d_vec = np.mean((real_spec - med) ** 2, axis=0)
        d_vec = np.where(d_vec == 0, 1.0, d_vec)
        inv_sq = d_vec**-2
        w = inv_sq / np.sum(inv_sq)
        return w
    if method == "WACFM":
        return wacfm(real_spec, "GCD")["w"]
    raise ValueError(f"Weighted averaging method '{method}' not recognized!")


def signal_averaging(
    mrs_struct: MRSStruct,
    all_frames_ft: ArrayLike,
    all_frames_ft_realign: ArrayLike,
    ii: int,
    kk: int,
    vox: list[str],
) -> MRSStruct:
    """Average aligned subspectra and compute ON, OFF, DIFF, and SUM spectra.

    Mirrors ``SignalAveraging.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure.
    all_frames_ft: ArrayLike
        Unaligned zero-filled spectra ``(n_zf, n_transients)``.
    all_frames_ft_realign: ArrayLike
        Aligned zero-filled spectra ``(n_zf, n_transients)``.
    ii: int
        Zero-based scan index.
    kk: int
        Zero-based voxel index.
    vox: list[str]
        List of voxel names.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with ``spec.<vox>`` populated.
    """
    p = mrs_struct["p"]
    ft_orig = np.asarray(all_frames_ft, dtype=np.complex128)
    ft_realign = np.asarray(all_frames_ft_realign, dtype=np.complex128)
    n_zf, n_trans = ft_realign.shape
    num_scans = int(p.get("numScans", ii + 1))
    n_sub = 4 if bool(p.get("HERMES", 0)) else 2
    experiment = ["A", "B", "C", "D"]
    method = "MSE"

    vox_name = vox[kk]
    spec_vox = mrs_struct.ensure_substruct("spec", vox_name)
    subspec_struct = spec_vox.ensure_substruct("subspec")

    on_off = np.asarray(mrs_struct["fids"]["ON_OFF"], dtype=np.int64)
    if on_off.ndim == 1:
        on_off = on_off[np.newaxis, :]

    fids_data = np.asarray(mrs_struct["fids"]["data"])
    if bool(p.get("weighted_averaging", 1)) and fids_data.shape[1] >= 4:
        p["weighted_averaging_method"] = method
        sig_avg_out = mrs_struct.ensure_substruct("out", "signal_averaging")
        if "w" not in sig_avg_out:
            sig_avg_out["w"] = [None] * num_scans
        sig_avg_out["w"][ii] = np.zeros(n_trans, dtype=np.float64)

        npoints = int(p["npoints"][ii])
        freq_range = float(p["sw"][ii]) / float(p["LarmorFreq"][ii])
        freq = (
            (npoints + 1 - np.arange(1, npoints + 1)) / npoints * freq_range
            + 4.68
            - freq_range / 2.0
        )
        freq_lim = (freq <= 3.4) & (freq >= 1.8)

        for jj in range(n_sub):
            if p.get("vendor") == "Philips" and p.get("seqorig") == "Philips":
                ind = np.where(on_off[0] == abs(jj - 1))[0]
            else:
                ind = np.arange(jj, n_trans, n_sub)

            w = _compute_mse_weights(ft_realign, ind, npoints, freq_lim, method)
            sig_avg_out["w"][ii][ind] = w
            mat = _ensure_spec_matrix(subspec_struct, experiment[jj], num_scans, n_zf)
            mat[ii, :] = np.sum(w[np.newaxis, :] * ft_realign[:, ind], axis=1)
    else:
        p["weighted_averaging"] = 0
        reject_vec = np.asarray(
            mrs_struct["out"]["reject"][ii], dtype=np.float64
        ).ravel()
        for jj in range(n_sub):
            if p.get("vendor") == "Philips" and p.get("seqorig") == "Philips":
                ind = np.where(on_off[0] == abs(jj - 1))[0]
            else:
                ind = np.arange(jj, n_trans, n_sub)
            mask = np.zeros(n_trans, dtype=bool)
            mask[ind] = True
            valid = mask & (reject_vec == 0)
            if not np.any(valid):
                valid = mask
            mat = _ensure_spec_matrix(subspec_struct, experiment[jj], num_scans, n_zf)
            mat[ii, :] = np.mean(ft_realign[:, valid], axis=1)

    targets = list(p["target"])
    hermes = bool(p.get("HERMES", 0))
    for jj, targ in enumerate(targets):
        t_struct = spec_vox.ensure_substruct(targ)
        if p.get("vendor") == "Philips" and p.get("seqorig") == "Philips":
            on_ind = np.array([0], dtype=np.int64)
            off_ind = np.array([1], dtype=np.int64)
        else:
            on_ind = np.where(on_off[jj, :n_sub] == 1)[0]
            off_ind = np.where(on_off[jj, :n_sub] == 0)[0]

        on_mat = _ensure_spec_matrix(t_struct, "on", num_scans, n_zf)
        off_mat = _ensure_spec_matrix(t_struct, "off", num_scans, n_zf)
        diff_u_mat = _ensure_spec_matrix(t_struct, "diff_unfilt", num_scans, n_zf)
        diff_na_mat = _ensure_spec_matrix(
            t_struct, "diff_unfilt_noalign", num_scans, n_zf
        )
        sum_mat = _ensure_spec_matrix(t_struct, "sum", num_scans, n_zf)

        if hermes:
            on_mat[ii, :] = (
                subspec_struct[experiment[on_ind[0]]][ii, :]
                + subspec_struct[experiment[on_ind[1]]][ii, :]
            ) / 2.0
            off_mat[ii, :] = (
                subspec_struct[experiment[off_ind[0]]][ii, :]
                + subspec_struct[experiment[off_ind[1]]][ii, :]
            ) / 2.0
            off_off_idx = int(np.where(np.all(on_off[:, :n_sub].T == 0, axis=1))[0][0])
            off_off_mat = _ensure_spec_matrix(t_struct, "off_off", num_scans, n_zf)
            off_off_mat[ii, :] = subspec_struct[experiment[off_off_idx]][ii, :]
        else:
            on_mat[ii, :] = subspec_struct[experiment[on_ind[0]]][ii, :]
            off_mat[ii, :] = subspec_struct[experiment[off_ind[0]]][ii, :]

        diff_u_mat[ii, :] = (on_mat[ii, :] - off_mat[ii, :]) / 2.0
        diff_na_mat[ii, :] = (
            np.mean(ft_orig[:, on_off[jj, :] == 1], axis=1)
            - np.mean(ft_orig[:, on_off[jj, :] == 0], axis=1)
        ) / 2.0
        sum_mat[ii, :] = (on_mat[ii, :] + off_mat[ii, :]) / 2.0

    return mrs_struct
