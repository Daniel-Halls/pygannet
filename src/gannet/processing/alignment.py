"""Time- and frequency-domain transient alignment algorithms."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from gannet.config import MRSStruct
from gannet.fitting.optimizers import lsqnonlin, nlinfit, zscore
from gannet.fitting.peak_fitters import fit_cho_cr, fit_cr, fit_peaks_by_frames
from gannet.processing.signal_filter import signal_filter
from gannet.processing.subspectra_align import (
    align_sub_spectra,
    align_sub_spectra_pre_align_ref,
)


def freq_phase_shift_nest(
    pars: ArrayLike, input_dict: dict[str, object]
) -> NDArray[np.float64]:
    """Apply frequency and phase shift to a flattened real/imag FID vector.

    Mirrors ``FreqPhaseShiftNest.m``.

    Parameters
    ------------
    pars: ArrayLike
        Two-element vector ``[freq_shift_hz, phase_shift_deg]``.
    input_dict: dict[str, object]
        Dictionary with keys ``'dwelltime'`` (float) and ``'data'`` (1D array
        of concatenated real and imaginary points).

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Shifted concatenated ``[real; imag]`` vector.
    """
    p = np.asarray(pars, dtype=np.float64)
    f_hz = float(p[0])
    phi_deg = float(p[1])
    dwell = float(input_dict["dwelltime"])  # type: ignore[arg-type]
    raw = np.asarray(input_dict["data"], dtype=np.float64).ravel()
    half = raw.size // 2
    fid = raw[:half] + 1j * raw[half:]
    t_vec = np.arange(half, dtype=np.float64) * dwell
    shifted = fid * np.exp(1j * np.pi * (t_vec * f_hz * 2.0 + phi_deg / 180.0))
    out: NDArray[np.float64] = np.concatenate([np.real(shifted), np.imag(shifted)])
    return out


def spec_reg(
    data: ArrayLike,
    target: ArrayLike,
    t_vec: ArrayLike,
    x: ArrayLike,
) -> NDArray[np.float64]:
    """Compute time-domain spectral registration residual vector.

    Mirrors ``SpecReg.m``.

    Parameters
    ------------
    data: ArrayLike
        Flattened transient vector ``[real; imag]``.
    target: ArrayLike
        Flattened reference vector ``[real; imag]``.
    t_vec: ArrayLike
        Time vector in seconds.
    x: ArrayLike
        Parameters ``[freq_shift_hz, phase_shift_deg]``.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Residual ``target - shifted_data``.
    """
    p = np.asarray(x, dtype=np.float64)
    d = np.asarray(data, dtype=np.float64).ravel()
    targ = np.asarray(target, dtype=np.float64).ravel()
    t_arr = np.asarray(t_vec, dtype=np.float64).ravel()
    half = d.size // 2
    fid = d[:half] + 1j * d[half:]
    fid = fid * np.exp(1j * np.pi * (t_arr * p[0] * 2.0 + p[1] / 180.0))
    stacked = np.concatenate([np.real(fid), np.imag(fid)])
    result: NDArray[np.float64] = targ - stacked
    return result


def align_using_peak(
    all_frames_ft: ArrayLike, mrs_struct: MRSStruct
) -> tuple[NDArray[np.complex128], MRSStruct]:
    """Align transients frame-by-frame using a Lorentzian fit to NAA, Cr, or Cho.

    Mirrors ``AlignUsingPeak.m``.

    Parameters
    ------------
    all_frames_ft: ArrayLike
        Frequency-domain spectra ``(n_zf, n_transients)``.
    mrs_struct: MRSStruct
        MRS data structure containing ``p.alignment`` and ``spec.freq``.

    Returns
    --------
        tuple[NDArray[np.complex128], MRSStruct]:
            Tuple containing ``(aligned_spectra, updated_mrs_struct)``.
    """
    realign = np.asarray(all_frames_ft, dtype=np.complex128).copy()
    p = mrs_struct["p"]
    ii = int(mrs_struct["ii"])
    freq = np.asarray(mrs_struct["spec"]["freq"], dtype=np.float64).ravel()
    align_mode = str(p["alignment"])

    if align_mode == "NAA":
        peak_ppm = 2.02
        lb_idx = int(np.argmin(np.abs(freq - 2.26)))
        ub_idx = int(np.argmin(np.abs(freq - 1.76)))
        initx = np.array([50.0, 0.1, peak_ppm, 0.0, 0.0, 0.0])
    elif align_mode == "Cr":
        peak_ppm = 3.02
        lb_idx = int(np.argmin(np.abs(freq - 3.12)))
        ub_idx = int(np.argmin(np.abs(freq - 2.72)))
        initx = np.array([30.0, 0.05, peak_ppm, 0.0, 0.0, 0.0])
    else:
        peak_ppm = 3.2
        lb_idx = int(np.argmin(np.abs(freq - 3.32)))
        ub_idx = int(np.argmin(np.abs(freq - 3.1)))
        initx = np.array([30.0, 0.05, peak_ppm, 0.0, 0.0, 0.0])

    lo, hi = min(lb_idx, ub_idx), max(lb_idx, ub_idx)
    freq_range = freq[lo : hi + 1]
    sum_spec = np.sum(realign[lo : hi + 1, :], axis=1)
    sum_params = fit_peaks_by_frames(freq_range, sum_spec, initx)["fit_params"]

    realign = realign * np.exp(1j * float(sum_params[3]))
    freq_step = abs(float(freq[0] - freq[1]))
    shift_pts = round((float(np.real(sum_params[2])) - peak_ppm) / freq_step)
    realign = np.roll(realign, shift_pts, axis=0)

    n_frames = realign.shape[1]
    init_frames = sum_params / np.array(
        [n_frames, 1.0, 1.0, 1.0, n_frames, n_frames], dtype=np.float64
    )
    init_frames[2] = peak_ppm
    init_frames[3] = 0.0

    frame_params = fit_peaks_by_frames(
        freq_range, realign[lo : hi + 1, :], init_frames
    )["fit_params"]
    if frame_params.ndim == 1:
        frame_params = frame_params[np.newaxis, :]

    frame_shifts = np.round(
        (np.real(frame_params[:, 2]) - peak_ppm) / freq_step
    ).astype(np.int64)
    for jj in range(n_frames):
        realign[:, jj] = np.roll(realign[:, jj], int(frame_shifts[jj]))
    realign = realign * np.exp(1j * frame_params[:, 3])[np.newaxis, :]
    realign = realign - frame_params[:, 4][np.newaxis, :]

    clb = int(np.argmin(np.abs(freq - 3.11)))
    cub = int(np.argmin(np.abs(freq - 2.6)))
    c_lo, c_hi = min(clb, cub), max(clb, cub)
    cr_mean = np.mean(realign[c_lo : c_hi + 1, :], axis=1)
    cr_fit = fit_cr(
        freq[c_lo : c_hi + 1], cr_mean, init_frames, float(p["LarmorFreq"][ii])
    )["fit_params"]
    out = mrs_struct.ensure_substruct("out")
    num_scans = int(p.get("numScans", ii + 1))
    if "CrFWHMHz" not in out or len(out["CrFWHMHz"]) < num_scans:
        out["CrFWHMHz"] = np.zeros(num_scans, dtype=np.float64)
    out["CrFWHMHz"][ii] = float(cr_fit[1])

    mean_fp = np.mean(frame_params, axis=0)
    std_fp = np.std(frame_params, axis=0, ddof=0)
    upper = np.tile(mean_fp + 3.0 * std_fp, (n_frames, 1))
    lower = np.tile(mean_fp - 3.0 * std_fp, (n_frames, 1))
    upper[:, 4:6] = np.inf
    lower[:, 4:6] = -np.inf
    reject = np.max((frame_params > upper) | (frame_params < lower), axis=1)
    reject_vec = reject.astype(np.float64).copy()

    on_off = np.asarray(mrs_struct["fids"]["ON_OFF"], dtype=np.int64)
    if on_off.ndim == 1:
        on_off = on_off[np.newaxis, :]
    diff_idx = np.where(on_off[0] != on_off[0, 0])[0]
    step_size = int(diff_idx[0]) if diff_idx.size > 0 else 1
    for kk in range(reject_vec.size):
        if reject[kk]:
            if on_off[0, kk] == on_off[0, 0] and kk + step_size < n_frames:
                reject_vec[kk + step_size] = 1.0
            elif kk - step_size >= 0:
                reject_vec[kk - step_size] = 1.0

    if "reject" not in out:
        out["reject"] = [None] * num_scans
    out["reject"][ii] = reject_vec
    result = (realign, mrs_struct)
    return result


def align_using_h2o(
    all_frames_ft: ArrayLike, mrs_struct: MRSStruct
) -> tuple[NDArray[np.complex128], MRSStruct]:
    """Align transients to the maximum magnitude of the residual water peak.

    Mirrors ``AlignUsingH2O.m``.

    Parameters
    ------------
    all_frames_ft: ArrayLike
        Frequency-domain spectra ``(n_zf, n_transients)``.
    mrs_struct: MRSStruct
        MRS data structure.

    Returns
    --------
        tuple[NDArray[np.complex128], MRSStruct]:
            Tuple containing ``(aligned_spectra, updated_mrs_struct)``.
    """
    inp = np.asarray(all_frames_ft, dtype=np.complex128)
    n_pts, n_trans = inp.shape
    output = np.zeros_like(inp)
    max_idx = np.argmax(np.abs(inp), axis=0)
    shifts = max_idx - (n_pts // 2)
    for col in range(n_trans):
        output[:, col] = np.roll(inp[:, col], -int(shifts[col]))

    ii = int(mrs_struct["ii"])
    num_scans = int(mrs_struct["p"].get("numScans", ii + 1))
    out = mrs_struct.ensure_substruct("out")
    if "reject" not in out:
        out["reject"] = [None] * num_scans
    out["reject"][ii] = np.zeros(n_trans, dtype=np.float64)
    result = (output, mrs_struct)
    return result


def _estimate_tmax_snr(
    fids_subset: NDArray[np.complex128],
    time_vec: NDArray[np.float64],
    mode: str = "robust",
) -> int:
    """Determine number of time-domain points ``tMax`` based on FID SNR."""
    signal = np.abs(fids_subset)
    n_pts = signal.shape[0]
    tail_start = int(np.ceil(0.75 * n_pts)) - 1
    noise = 2.0 * np.std(signal[tail_start:, :], axis=0, ddof=1)
    noise = np.where(noise == 0, 1e-12, noise)
    snr_mat = signal / noise[np.newaxis, :]
    if mode == "specreg":
        snr = np.mean(snr_mat, axis=1)
        idx = np.where(snr > 1.5)[0]
        if idx.size == 0:
            return min(100, n_pts)
        return int(max(idx[-1] + 1, min(100, n_pts)))

    snr_diff = np.abs(np.diff(np.mean(snr_mat, axis=1)))
    snr_win = snr_diff[time_vec[:-1] <= 0.2]
    idx = np.where(snr_win > 0.5)[0]
    min_pts = int(np.where(time_vec <= 0.05)[0][-1]) + 1
    if idx.size == 0 or (idx[-1] + 1) < min_pts:
        return min_pts
    return int(idx[-1] + 1)


def _fit_cauchy_mle(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit Cauchy scale and location parameters via maximum likelihood."""
    q75, q25 = np.percentile(values, [75.0, 25.0])
    s0 = max(float((q75 - q25) / 2.0), 1e-4)
    l0 = float(np.median(values))

    def nll(params: NDArray[np.float64]) -> float:
        """Compute negative log-likelihood for Cauchy distribution."""
        scale, loc = float(params[0]), float(params[1])
        if scale <= 0.0:
            return 1e12
        pdf = scale / (np.pi * (scale**2 + (values - loc) ** 2))
        return float(-np.sum(np.log(np.maximum(pdf, 1e-300))))

    res = minimize(
        nll,
        np.array([s0, l0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(1e-9, None), (None, None)],
    )
    out: NDArray[np.float64] = np.asarray(res.x, dtype=np.float64)
    return out


def spectral_registration(
    mrs_struct: MRSStruct, on_what: int = 0, dual: int = 0
) -> tuple[NDArray[np.complex128], MRSStruct]:
    """Perform time-domain spectral registration (Near et al., 2015).

    Mirrors ``SpectralRegistration.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure.
    on_what: int
        0 for metabolite data, 1 for water data. Defaults to 0.
    dual: int
        1 to register ON and OFF transients separately. Defaults to 0.

    Returns
    --------
        tuple[NDArray[np.complex128], MRSStruct]:
            Tuple containing ``(aligned_spectra, updated_mrs_struct)``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    num_scans = int(p.get("numScans", ii + 1))
    npoints = int(p["npoints"][ii])
    sw = float(p["sw"][ii])
    larmor = float(p["LarmorFreq"][ii])
    time_vec = np.arange(npoints, dtype=np.float64) / sw
    spec_reg_loop = 1 if dual == 1 else 0

    fids_data = np.asarray(mrs_struct["fids"]["data"], dtype=np.complex128)
    on_off = np.asarray(mrs_struct["fids"]["ON_OFF"], dtype=np.int64)
    if on_off.ndim == 1:
        on_off = on_off[np.newaxis, :]

    t_max = _estimate_tmax_snr(fids_data, time_vec, mode="specreg")
    out = mrs_struct.ensure_substruct("out")
    sr_out = out.ensure_substruct("SpecReg")
    for key in ("freq", "phase", "zMSE"):
        if key not in sr_out:
            sr_out[key] = [None] * num_scans
    if "FreqStdevHz" not in out or len(out["FreqStdevHz"]) < num_scans:
        out["FreqStdevHz"] = np.zeros(num_scans, dtype=np.float64)
    if "reject" not in out:
        out["reject"] = [None] * num_scans

    mrs_struct["fids"]["data_align"] = np.zeros_like(fids_data)
    all_realign = np.zeros(
        (int(p["ZeroFillTo"][ii]), fids_data.shape[1]), dtype=np.complex128
    )

    while spec_reg_loop > -1:
        if on_what:
            w_fids = np.asarray(mrs_struct["fids"]["data_water"], dtype=np.complex128)
            sub_fids = w_fids[:t_max, :]
        elif dual == 1:
            sub_fids = fids_data[:t_max, on_off[0] == spec_reg_loop]
        else:
            sub_fids = fids_data[:t_max, :]

        n_sub_t = sub_fids.shape[1]
        flatdata = np.zeros((t_max, 2, n_sub_t), dtype=np.float64)
        flatdata[:, 0, :] = np.real(sub_fids)
        flatdata[:, 1, :] = np.imag(sub_fids)
        flattarget = np.median(flatdata, axis=2)
        target = np.concatenate([flattarget[:, 0], flattarget[:, 1]])

        pars_guess = np.array([0.0, 0.0], dtype=np.float64)
        pars_fit = np.zeros((n_sub_t, 2), dtype=np.float64)
        mse_arr = np.zeros(n_sub_t, dtype=np.float64)
        dwell = 1.0 / sw

        for corrloop in range(n_sub_t):
            transient = np.concatenate(
                [flatdata[:, 0, corrloop], flatdata[:, 1, corrloop]]
            )
            inp_dict: dict[str, object] = {
                "dwelltime": dwell,
                "data": transient,
            }
            nlin_res = nlinfit(
                inp_dict,
                target,
                freq_phase_shift_nest,
                pars_guess,
                max_iter=400,
                tol_x=1e-8,
                tol_fun=1e-8,
            )
            pars_fit[corrloop, :] = nlin_res.beta
            mse_arr[corrloop] = nlin_res.mse
            pars_guess = pars_fit[corrloop, :]

        z_mse = zscore(mse_arr)
        if on_what:
            w_fids = np.asarray(
                mrs_struct["fids"]["data_water"], dtype=np.complex128
            ).copy()
            for corrloop in range(n_sub_t):
                w_fids[:, corrloop] = (
                    w_fids[:, corrloop]
                    * np.exp(1j * pars_fit[corrloop, 0] * 2.0 * np.pi * time_vec)
                    * np.exp(1j * np.pi / 180.0 * pars_fit[corrloop, 1])
                )
            mrs_struct["fids"]["data_water"] = w_fids
            full_data = (
                w_fids * np.exp(-time_vec * float(p["LB"]) * np.pi)[:, np.newaxis]
            )
            all_realign = np.fft.fftshift(
                np.fft.fft(full_data, n=int(p["ZeroFillTo"][ii]), axis=0),
                axes=0,
            )
        else:
            sr_out["freq"][ii] = pars_fit[:, 0]
            sr_out["phase"][ii] = pars_fit[:, 1]
            if dual == 1:
                ind_target = np.where(on_off[0] == spec_reg_loop)[0]
                for idx_c, corrloop_d in enumerate(ind_target):
                    mrs_struct["fids"]["data_align"][:, corrloop_d] = (
                        fids_data[:, corrloop_d]
                        * np.exp(1j * pars_fit[idx_c, 0] * 2.0 * np.pi * time_vec)
                        * np.exp(1j * np.pi / 180.0 * pars_fit[idx_c, 1])
                    )
            else:
                mrs_struct["fids"]["data_align"] = (
                    fids_data
                    * np.exp(
                        1j
                        * pars_fit[:, 0][np.newaxis, :]
                        * 2.0
                        * np.pi
                        * time_vec[:, np.newaxis]
                    )
                    * np.exp(1j * np.pi / 180.0 * pars_fit[:, 1][np.newaxis, :])
                )

            if spec_reg_loop == 0:
                full_data = (
                    mrs_struct["fids"]["data_align"]
                    * np.exp(-time_vec * float(p["LB"]) * np.pi)[:, np.newaxis]
                )
                all_realign = np.fft.fftshift(
                    np.fft.fft(full_data, n=int(p["ZeroFillTo"][ii]), axis=0),
                    axes=0,
                )
                if not bool(p.get("phantom", 0)):
                    freq_full = np.asarray(
                        mrs_struct["spec"]["freq"], dtype=np.float64
                    ).ravel()
                    freq_bounds = (freq_full <= 3.6) & (freq_full >= 2.6)
                    f_sub = freq_full[freq_bounds]
                    df_ppm = abs(float(freq_full[0] - freq_full[1]))
                    width = 0.05
                    if dual == 1:
                        for mask_val in (1, 0):
                            sel = on_off[0] == mask_val
                            cho_cr = np.mean(
                                all_realign[np.ix_(freq_bounds, sel)], axis=1
                            )
                            base_off = float(np.real(cho_cr[0] + cho_cr[-1]) / 2.0)
                            area_est = float(
                                (np.max(np.real(cho_cr)) - np.min(np.real(cho_cr)))
                                * width
                                * 4.0
                            )
                            init_cc = np.array(
                                [area_est, width, 3.02, 0.0, base_off, 0.0, 1.0]
                            ) * np.array(
                                [
                                    1.0,
                                    2.0 * larmor,
                                    larmor,
                                    180.0 / np.pi,
                                    1.0,
                                    1.0,
                                    1.0,
                                ]
                            )
                            fit_cc = fit_cho_cr(f_sub, cho_cr, init_cc, larmor)[
                                "fit_params"
                            ]
                            all_realign[:, sel] = all_realign[:, sel] * np.exp(
                                1j * np.pi / 180.0 * float(fit_cc[3])
                            )
                            shift_pts = round(
                                (float(fit_cc[2]) - 3.02 * larmor) / (larmor * df_ppm)
                            )
                            all_realign[:, sel] = np.roll(
                                all_realign[:, sel], shift_pts, axis=0
                            )
                    else:
                        cho_cr = np.mean(all_realign[freq_bounds, :], axis=1)
                        base_off = float(np.real(cho_cr[0] + cho_cr[-1]) / 2.0)
                        area_est = float(
                            (np.max(np.real(cho_cr)) - np.min(np.real(cho_cr)))
                            * width
                            * 4.0
                        )
                        init_cc = np.array(
                            [area_est, width, 3.02, 0.0, base_off, 0.0, 1.0]
                        ) * np.array(
                            [1.0, 2.0 * larmor, larmor, 180.0 / np.pi, 1.0, 1.0, 1.0]
                        )
                        fit_cc = fit_cho_cr(f_sub, cho_cr, init_cc, larmor)[
                            "fit_params"
                        ]
                        all_realign = all_realign * np.exp(
                            1j * np.pi / 180.0 * float(fit_cc[3])
                        )
                        shift_pts = round(
                            (float(fit_cc[2]) - 3.02 * larmor) / (larmor * df_ppm)
                        )
                        all_realign = np.roll(all_realign, shift_pts, axis=0)

                out["FreqStdevHz"][ii] = float(np.std(pars_fit[:, 0], ddof=0))
                sr_out["zMSE"][ii] = z_mse
                out["reject"][ii] = (z_mse > 3.0).astype(np.float64)

        spec_reg_loop -= 1

    result = (all_realign, mrs_struct)
    return result


def spectral_registration_hermes(
    mrs_struct: MRSStruct,
) -> tuple[NDArray[np.complex128], MRSStruct]:
    """Perform multistep HERMES spectral registration (Mikkelsen et al., 2018).

    Mirrors ``SpectralRegistrationHERMES.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure.

    Returns
    --------
        tuple[NDArray[np.complex128], MRSStruct]:
            Tuple containing ``(aligned_spectra, updated_mrs_struct)``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    num_scans = int(p.get("numScans", ii + 1))
    npoints = int(p["npoints"][ii])
    sw = float(p["sw"][ii])
    larmor = float(p["LarmorFreq"][ii])
    hermes = bool(p.get("HERMES", 0))

    data_to_align = np.asarray(mrs_struct["fids"]["data"], dtype=np.complex128).copy()
    n_trans = data_to_align.shape[1]
    time_vec = np.arange(npoints, dtype=np.float64) / sw
    dwell = 1.0 / sw

    on_off = np.asarray(mrs_struct["fids"]["ON_OFF"], dtype=np.int64)
    if on_off.ndim == 1:
        on_off = on_off[np.newaxis, :]

    if hermes:
        spec_reg_loop = 3
        subspec_to_align = np.tile(np.array([3, 2, 1, 0]), n_trans // 4)
    else:
        spec_reg_loop = 1
        subspec_to_align = on_off[0]

    out = mrs_struct.ensure_substruct("out")
    sr_out = out.ensure_substruct("SpecReg")
    for key in ("freq", "phase", "zMSE"):
        if key not in sr_out:
            sr_out[key] = [None] * num_scans
    if "reject" not in out:
        out["reject"] = [None] * num_scans

    sr_out["freq"][ii] = np.zeros(n_trans, dtype=np.float64)
    sr_out["phase"][ii] = np.zeros(n_trans, dtype=np.float64)
    z_mse = np.zeros(n_trans, dtype=np.float64)
    pars_guess = np.array([0.0, 0.0], dtype=np.float64)
    all_realign = np.zeros((int(p["ZeroFillTo"][ii]), n_trans), dtype=np.complex128)

    while spec_reg_loop > -1:
        ind = np.where(subspec_to_align == spec_reg_loop)[0]
        t_max = _estimate_tmax_snr(data_to_align[:, ind], time_vec, mode="robust")
        n_sub_t = ind.size
        flatdata = np.zeros((t_max, 2, n_sub_t), dtype=np.float64)
        flatdata[:, 0, :] = np.real(data_to_align[:t_max, ind])
        flatdata[:, 1, :] = np.imag(data_to_align[:t_max, ind])
        flattarget = np.median(flatdata, axis=2)
        target = np.concatenate([flattarget[:, 0], flattarget[:, 1]])

        pars_fit = np.zeros((n_sub_t, 2), dtype=np.float64)
        mse_arr = np.zeros(n_sub_t, dtype=np.float64)
        for corrloop in range(n_sub_t):
            transient = np.concatenate(
                [flatdata[:, 0, corrloop], flatdata[:, 1, corrloop]]
            )
            inp_dict: dict[str, object] = {"dwelltime": dwell, "data": transient}
            nlin_res = nlinfit(
                inp_dict,
                target,
                freq_phase_shift_nest,
                pars_guess,
                max_iter=400,
                tol_x=1e-8,
                tol_fun=1e-8,
            )
            pars_fit[corrloop, :] = nlin_res.beta
            mse_arr[corrloop] = nlin_res.mse
            pars_guess = pars_fit[corrloop, :]

        p_freq = _fit_cauchy_mle(pars_fit[:, 0])
        p_phase = _fit_cauchy_mle(pars_fit[:, 1])

        sr_out["freq"][ii][ind] = pars_fit[:, 0] - p_freq[1]
        sr_out["phase"][ii][ind] = pars_fit[:, 1] - p_phase[1]
        z_mse[ind] = zscore(mse_arr)

        for idx_c, col in enumerate(ind):
            data_to_align[:, col] = (
                data_to_align[:, col]
                * np.exp(1j * (pars_fit[idx_c, 0] - p_freq[1]) * 2.0 * np.pi * time_vec)
                * np.exp(1j * np.pi / 180.0 * (pars_fit[idx_c, 1] - p_phase[1]))
            )

        if spec_reg_loop == 0:
            full_data = (
                data_to_align
                * np.exp(-time_vec * float(p["LB"]) * np.pi)[:, np.newaxis]
            )
            all_realign = np.fft.fftshift(
                np.fft.fft(full_data, n=int(p["ZeroFillTo"][ii]), axis=0),
                axes=0,
            )
            if not bool(p.get("phantom", 0)):
                freq_full = np.asarray(
                    mrs_struct["spec"]["freq"], dtype=np.float64
                ).ravel()
                freq_lim_cr = (freq_full >= 2.925) & (freq_full <= 3.125)
                frame_max_pos = np.argmax(np.real(all_realign[freq_lim_cr, :]), axis=0)
                cr_shift = freq_full[freq_lim_cr][frame_max_pos] - 3.02
                df_ppm = abs(float(freq_full[0] - freq_full[1]))
                cr_shift_pts = np.round(cr_shift / df_ppm).astype(np.int64)
                for col in range(n_trans):
                    all_realign[:, col] = np.roll(
                        all_realign[:, col], int(cr_shift_pts[col])
                    )
                sr_out["freq"][ii] = sr_out["freq"][ii] + cr_shift * larmor

                ind_off = np.all(on_off.T == 0, axis=1)
                off_spec = np.real(np.mean(all_realign[:, ind_off], axis=1))
                fl_cr = (freq_full <= 3.12) & (freq_full >= 2.92)
                max_freq = float(
                    freq_full[fl_cr][int(np.argmax(np.abs(off_spec[fl_cr])))]
                )
                fl_cc = (freq_full <= max_freq + 0.58) & (freq_full >= max_freq - 0.42)
                off_seg = off_spec[fl_cc]
                baseline = float((off_seg[0] + off_seg[-1]) / 2.0)
                width = 0.05
                area = float((np.max(off_seg) - np.min(off_seg)) * width * 4.0)
                x0 = np.array(
                    [area, width, max_freq, 0.0, baseline, 0.0, 1.0]
                ) * np.array([1.0, 2.0 * larmor, larmor, 180.0 / np.pi, 1.0, 1.0, 1.0])
                phi = float(
                    fit_cho_cr(freq_full[fl_cc], off_seg, x0, larmor)["fit_params"][3]
                )

                tmp_fids = np.fft.ifft(np.fft.ifftshift(all_realign, axes=0), axis=0)
                tmp_fids = tmp_fids * np.exp(1j * np.pi / 180.0 * phi)
                all_realign = np.fft.fftshift(np.fft.fft(tmp_fids, axis=0), axes=0)
                sr_out["phase"][ii] = sr_out["phase"][ii] + phi

            sr_out["zMSE"][ii] = z_mse
            out["reject"][ii] = (z_mse > 3.0).astype(np.float64)

        spec_reg_loop -= 1

    result = (all_realign, mrs_struct)
    return result


def robust_spectral_registration(
    mrs_struct: MRSStruct,
) -> tuple[NDArray[np.complex128], MRSStruct]:
    """Align spectra using Robust Spectral Registration (Mikkelsen et al., 2020).

    Mirrors ``RobustSpectralRegistration.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure.

    Returns
    --------
        tuple[NDArray[np.complex128], MRSStruct]:
            Tuple containing ``(aligned_spectra, updated_mrs_struct)``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    num_scans = int(p.get("numScans", ii + 1))
    npoints = int(p["npoints"][ii])
    sw = float(p["sw"][ii])
    larmor = float(p["LarmorFreq"][ii])
    hermes = bool(p.get("HERMES", 0))

    fids_raw = np.asarray(mrs_struct["fids"]["data"], dtype=np.complex128)
    n_trans = fids_raw.shape[1]
    on_off = np.asarray(mrs_struct["fids"]["ON_OFF"], dtype=np.int64)
    if on_off.ndim == 1:
        on_off = on_off[np.newaxis, :]

    if hermes:
        spec_reg_loop = 3
        subspec_to_align = np.tile(np.array([3, 2, 1, 0]), n_trans // 4)
    else:
        spec_reg_loop = 1
        subspec_to_align = on_off[0]

    out = mrs_struct.ensure_substruct("out")
    sr_out = out.ensure_substruct("SpecReg")
    for key in ("freq", "phase", "MSE", "zMSE"):
        if key not in sr_out:
            sr_out[key] = [None] * num_scans
    if "reject" not in out:
        out["reject"] = [None] * num_scans

    sr_out["freq"][ii] = np.zeros(n_trans, dtype=np.float64)
    sr_out["phase"][ii] = np.zeros(n_trans, dtype=np.float64)
    sr_out["MSE"][ii] = np.zeros(n_trans, dtype=np.float64)
    z_mse = np.zeros(n_trans, dtype=np.float64)
    mrs_struct["fids"]["data_align"] = np.zeros_like(fids_raw)

    freq_range = sw / larmor
    freq = (
        (npoints + 1 - np.arange(1, npoints + 1)) / npoints * freq_range
        + 4.68
        - freq_range / 2.0
    )
    water_lim = (freq <= 4.68 + 0.25) & (freq >= 4.68 - 0.25)
    lipid_lim = (freq <= 1.85) & (freq >= 0.0)
    noise_lim = (freq <= 9.0) & (freq >= 8.0)

    spec = np.fft.fftshift(np.fft.fft(fids_raw, axis=0), axes=0)
    if hermes:
        ind_w = np.all(on_off.T == 0, axis=1)
        ind2 = ind_w
    elif p["target"][0] == "GSH":
        ind_w = on_off[0] == 0
        ind2 = ind_w
    else:
        ind_w = np.arange(n_trans)
        ind2 = on_off[0] == 0

    q_vals = np.sum(np.abs(spec[np.ix_(water_lim, ind_w)]), axis=0) * abs(
        freq[0] - freq[1]
    )
    med_q = np.median(q_vals)
    q_pct = (q_vals - med_q) / med_q * 100.0 if med_q != 0 else np.zeros_like(q_vals)
    q_metric = float(np.sum(np.abs(q_pct) > 40.0) / len(q_pct))

    s_abs = np.mean(np.abs(spec[:, ind2]), axis=1)
    max_naa = float(np.max(s_abs[(freq <= 4.25) & (freq >= 1.8)]))
    max_water = float(np.max(s_abs[(freq <= 4.68 + 0.22) & (freq >= 4.68 - 0.22)]))
    if max_naa > 0 and max_water / max_naa < 1.5:
        q_metric = 1.0

    s_real = np.mean(np.real(spec), axis=1)
    noise_seg = s_real[noise_lim]
    if noise_seg.size > 3:
        poly_fit = np.polyval(
            np.polyfit(freq[noise_lim], noise_seg, 2), freq[noise_lim]
        )
        noise_std = float(np.std(noise_seg - poly_fit, ddof=1))
    else:
        noise_std = 1.0
    r_metric = (
        float(np.std(s_real[lipid_lim], ddof=1)) / noise_std if noise_std > 0 else 0.0
    )

    water_flag = q_metric > 0.1
    lipid_flag = r_metric > 40.0
    if water_flag or lipid_flag:
        data_to_align = np.zeros_like(fids_raw)
        for jj in range(n_trans):
            data_to_align[:, jj] = signal_filter(
                spec[:, jj], lipid_flag, water_flag, jj, mrs_struct
            )
    else:
        data_to_align = fids_raw

    time_vec = np.arange(npoints, dtype=np.float64) / sw
    all_realign = np.zeros((int(p["ZeroFillTo"][ii]), n_trans), dtype=np.complex128)

    while spec_reg_loop > -1:
        ind = np.where(subspec_to_align == spec_reg_loop)[0]
        t_max = _estimate_tmax_snr(data_to_align[:, ind], time_vec, mode="robust")
        n_sub_t = ind.size
        flatdata = np.zeros((t_max, 2, n_sub_t), dtype=np.float64)
        flatdata[:, 0, :] = np.real(data_to_align[:t_max, ind])
        flatdata[:, 1, :] = np.imag(data_to_align[:t_max, ind])

        if n_trans <= 4:
            align_order = np.arange(n_sub_t)
        else:
            dist = np.zeros((n_sub_t, n_sub_t), dtype=np.float64)
            for jj in range(n_sub_t):
                sig_j = np.real(data_to_align[:t_max, ind[jj]])
                for kk in range(n_sub_t):
                    sig_k = np.real(data_to_align[:t_max, ind[kk]])
                    dist[jj, kk] = float(np.mean((sig_j - sig_k) ** 2))
            dist[dist == 0] = np.nan
            d_vec = np.nanmedian(dist, axis=0)
            align_order = np.argsort(d_vec, kind="stable")

        targ_0 = flatdata[:, :, align_order[0]]
        target = np.concatenate([targ_0[:, 0], targ_0[:, 1]])
        scale_a = float(np.max(np.abs(target))) or 1.0

        params = np.zeros((n_sub_t, 2), dtype=np.float64)
        mse_arr = np.zeros(n_sub_t, dtype=np.float64)
        f0_list = mrs_struct.get("spec", {}).get("F0freq2")
        if (
            f0_list is not None
            and ii < len(f0_list)
            and f0_list[ii] is not None
            and np.asarray(f0_list[ii]).ndim >= 1
            and np.asarray(f0_list[ii]).size == n_trans
        ):
            f0_vals = np.asarray(f0_list[ii], dtype=np.float64)[ind] * larmor
        else:
            f0_vals = np.zeros(n_sub_t, dtype=np.float64)
        f0_ord = f0_vals[align_order] - f0_vals[align_order[0]]
        x0_mat = np.column_stack([f0_ord, np.zeros_like(f0_ord)])
        t_sub = np.arange(t_max, dtype=np.float64) / sw

        for it_idx, jj in enumerate(align_order):
            tr_vec = np.concatenate([flatdata[:, 0, jj], flatdata[:, 1, jj]])
            params[jj, :] = lsqnonlin(
                lambda x, tr=tr_vec, tg=target, sa=scale_a, ts=t_sub: spec_reg(
                    tr / sa, tg / sa, ts, x
                ),
                x0_mat[it_idx, :],
            )
            f_val, phi_val = float(params[jj, 0]), float(params[jj, 1])
            m_c = (flatdata[:, 0, jj] + 1j * flatdata[:, 1, jj]) * np.exp(
                1j * np.pi * (t_sub * f_val * 2.0 + phi_val / 180.0)
            )
            m_col = np.concatenate([np.real(m_c), np.imag(m_c)])
            resid = target - m_col
            mse_arr[jj] = float(np.sum(resid**2) / max(resid.size - 2, 1))
            corr_val = float(np.corrcoef(target, m_col)[0, 1])
            w_jj = 0.5 * (corr_val**2) if np.isfinite(corr_val) else 0.0
            target = (1.0 - w_jj) * target + w_jj * m_col

        sr_out["freq"][ii][ind] = params[:, 0]
        sr_out["phase"][ii][ind] = params[:, 1]
        sr_out["MSE"][ii][ind] = mse_arr
        pos_mask = mse_arr > 0
        if np.any(pos_mask):
            z_mse[ind[pos_mask]] = zscore(mse_arr[pos_mask])

        mrs_struct["fids"]["data_align"][:, ind] = (
            fids_raw[:, ind]
            * np.exp(
                1j * params[:, 0][np.newaxis, :] * 2.0 * np.pi * time_vec[:, np.newaxis]
            )
            * np.exp(1j * np.pi / 180.0 * params[:, 1][np.newaxis, :])
        )

        if spec_reg_loop == 0:
            if bool(p.get("use_prealign_ref", 0)):
                mrs_struct = align_sub_spectra_pre_align_ref(mrs_struct, water_flag)
            else:
                mrs_struct = align_sub_spectra(mrs_struct, water_flag, r_metric)

            full_data = (
                mrs_struct["fids"]["data_align"]
                * np.exp(-time_vec * float(p["LB"]) * np.pi)[:, np.newaxis]
            )
            all_realign = np.fft.fftshift(
                np.fft.fft(full_data, n=int(p["ZeroFillTo"][ii]), axis=0),
                axes=0,
            )
            sr_out["zMSE"][ii] = z_mse
            out["reject"][ii] = (z_mse > 3.0).astype(np.float64)

        spec_reg_loop -= 1

    result = (all_realign, mrs_struct)
    return result
