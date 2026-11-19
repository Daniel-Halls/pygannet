"""Reference peak fitting routines (FitCr, FitChoCr, FitPeaksByFrames)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gannet.fitting.optimizers import lsqcurvefit, nlinfit
from gannet.models.lineshapes import lorentz_model, two_lorentz_model


def _unwrap_lorentz_params(
    fit_params: NDArray[np.float64],
    wrap_half_pi: bool = True,
) -> NDArray[np.float64]:
    """Normalize negative areas and unwrap Lorentzian phase angles.

    Parameters
    ------------
    fit_params: NDArray[np.float64]
        2D array of shape ``(nframes, nparams)``.
    wrap_half_pi: bool
        If True, fold phases outside ``[-pi/2, pi/2]`` by ``+/- pi``
        (as in ``FitCr`` and ``FitChoCr``). Otherwise fold to ``[-pi, pi]``.

    Returns
    --------
        NDArray[np.float64]: 2D numpy array
            Normalized parameter array.
    """
    params = fit_params.copy()
    neg_area = params[:, 0] < 0
    params[neg_area, 3] = params[neg_area, 3] + np.pi

    phase_wrapped = params[:, 3]
    phase_unwrapped = np.angle(np.cos(phase_wrapped) + 1j * np.sin(phase_wrapped))
    if wrap_half_pi:
        offset_pos = np.pi * (phase_unwrapped < -np.pi / 2.0)
        offset_neg = -np.pi * (phase_unwrapped > np.pi / 2.0)
        phase_unwrapped = phase_unwrapped + offset_pos + offset_neg
    params[:, 3] = phase_unwrapped
    params[:, 1] = np.abs(params[:, 1])
    return params


def _reject_outlier_frames(
    fit_params: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Identify frames whose non-baseline parameters exceed 3 std from mean.

    Parameters
    ------------
    fit_params: NDArray[np.float64]
        2D array of fitted parameters ``(nframes, nparams)``.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Binary rejection vector (1.0 for rejected frames, 0.0 otherwise).
    """
    mean_params = np.mean(fit_params, axis=0)
    std_params = np.std(fit_params, axis=0, ddof=0)
    upper = np.tile(mean_params + 3.0 * std_params, (fit_params.shape[0], 1))
    lower = np.tile(mean_params - 3.0 * std_params, (fit_params.shape[0], 1))
    upper[:, 4:6] = np.inf
    lower[:, 4:6] = -np.inf
    rejected = (fit_params > upper) | (fit_params < lower)
    reject_vec: NDArray[np.float64] = np.max(rejected, axis=1).astype(np.float64)
    return reject_vec


def fit_cr(
    freq: ArrayLike,
    frame_data: ArrayLike,
    initx: ArrayLike,
    larmor_freq: float,
) -> dict[str, NDArray[np.float64]]:
    """Fit a single Lorentzian peak (Creatine) across one or more frames.

    Mirrors ``FitCr.m``.

    Parameters
    ------------
    freq: ArrayLike
        Frequency vector in ppm.
    frame_data: ArrayLike
        1D or 2D spectral data of shape ``(nfreq,)`` or ``(nfreq, nframes)``.
    initx: ArrayLike
        Initial parameters in Hz/deg units ``[area, fwhm_hz, f0_hz, deg, b0, b1]``.
    larmor_freq: float
        Scanner Larmor frequency in MHz.

    Returns
    --------
        dict[str, NDArray[np.float64]]: dict object
            Dictionary with keys ``'fit_params'``, ``'reject_frame'``,
            and ``'resid'``.
    """
    freq_arr = np.asarray(freq, dtype=np.float64).ravel()
    data = np.asarray(frame_data)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    conv = np.array(
        [1.0, 2.0 * larmor_freq, larmor_freq, 180.0 / np.pi, 1.0, 1.0],
        dtype=np.float64,
    )
    curr_init = np.asarray(initx, dtype=np.float64).ravel() / conv

    nframes = data.shape[1]
    fit_params = np.zeros((nframes, 6), dtype=np.float64)
    resid = np.zeros(freq_arr.size, dtype=np.float64)

    for jj in range(nframes):
        y_real = np.real(data[:, jj])
        curr_init = lsqcurvefit(
            lorentz_model,
            curr_init,
            freq_arr,
            y_real,
            max_iter=10000,
            tol_x=1e-10,
            tol_fun=1e-10,
        )
        nlin_res = nlinfit(
            freq_arr,
            y_real,
            lorentz_model,
            curr_init,
            max_iter=10000,
        )
        fit_params[jj, :] = nlin_res.beta
        resid = nlin_res.resid

    fit_params = _unwrap_lorentz_params(fit_params, wrap_half_pi=True)
    fit_params[:, 0] = np.abs(fit_params[:, 0])
    fit_params = fit_params * np.tile(conv, (nframes, 1))
    reject_frame = _reject_outlier_frames(fit_params)

    result: dict[str, NDArray[np.float64]] = {
        "fit_params": fit_params.squeeze() if nframes == 1 else fit_params,
        "reject_frame": reject_frame,
        "resid": resid,
    }
    return result


def fit_cho_cr(
    freq: ArrayLike,
    frame_data: ArrayLike,
    initx: ArrayLike,
    larmor_freq: float,
) -> dict[str, NDArray[np.float64]]:
    """Fit a double-Lorentzian Choline + Creatine model across frames.

    Mirrors ``FitChoCr.m``.

    Parameters
    ------------
    freq: ArrayLike
        Frequency vector in ppm.
    frame_data: ArrayLike
        1D or 2D spectral data of shape ``(nfreq,)`` or ``(nfreq, nframes)``.
    initx: ArrayLike
        7-element initial parameters in Hz/deg for width, f0, phase.
    larmor_freq: float
        Scanner Larmor frequency in MHz.

    Returns
    --------
        dict[str, NDArray[np.float64]]: dict object
            Dictionary with keys ``'fit_params'``, ``'reject_frame'``,
            and ``'resid'``.
    """
    freq_arr = np.asarray(freq, dtype=np.float64).ravel()
    data = np.asarray(frame_data)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    curr_init = np.asarray(initx, dtype=np.float64).ravel().copy()
    curr_init[1:3] = curr_init[1:3] / np.array(
        [2.0 * larmor_freq, larmor_freq], dtype=np.float64
    )

    lb = np.array(
        [0.0, 0.0, curr_init[2] - 0.02, -np.pi, -np.inf, -np.inf, 0.0],
        dtype=np.float64,
    )
    ub = np.array(
        [
            2.0 * abs(curr_init[0]) + 1e-12,
            2.0 * abs(curr_init[1]) + 1e-12,
            curr_init[2] + 0.02,
            np.pi,
            np.inf,
            np.inf,
            3.0,
        ],
        dtype=np.float64,
    )

    nframes = data.shape[1]
    fit_params = np.zeros((nframes, 7), dtype=np.float64)
    resid = np.zeros(freq_arr.size, dtype=np.float64)

    for ii in range(nframes):
        y_real = np.real(data[:, ii])
        curr_init = lsqcurvefit(
            two_lorentz_model,
            curr_init,
            freq_arr,
            y_real,
            lb=lb,
            ub=ub,
            max_iter=800,
            tol_x=1e-4,
            tol_fun=1e-4,
        )
        nlin_res = nlinfit(
            freq_arr,
            y_real,
            two_lorentz_model,
            curr_init,
            max_iter=400,
            tol_x=1e-6,
            tol_fun=1e-6,
        )
        fit_params[ii, :] = nlin_res.beta
        resid = nlin_res.resid

    fit_params = _unwrap_lorentz_params(fit_params, wrap_half_pi=True)
    conv = np.array(
        [1.0, 2.0 * larmor_freq, larmor_freq, 180.0 / np.pi, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    fit_params = fit_params * np.tile(conv, (nframes, 1))
    reject_frame = _reject_outlier_frames(fit_params)

    result: dict[str, NDArray[np.float64]] = {
        "fit_params": fit_params.squeeze() if nframes == 1 else fit_params,
        "reject_frame": reject_frame,
        "resid": resid,
    }
    return result


def fit_peaks_by_frames(
    freq: ArrayLike,
    frame_data: ArrayLike,
    initx: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Fit a single Lorentzian peak frame-by-frame in ppm/radian units.

    Mirrors ``FitPeaksByFrames.m``.

    Parameters
    ------------
    freq: ArrayLike
        Frequency vector in ppm.
    frame_data: ArrayLike
        1D or 2D spectral data of shape ``(nfreq,)`` or ``(nfreq, nframes)``.
    initx: ArrayLike
        6-element initial guess ``[area, hwhm, f0, phase, b0, b1]``.

    Returns
    --------
        dict[str, NDArray[np.float64]]: dict object
            Dictionary with keys ``'fit_params'``, ``'reject_frame'``,
            and ``'resid'``.
    """
    freq_arr = np.asarray(freq, dtype=np.float64).ravel()
    data = np.asarray(frame_data)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    curr_init = np.asarray(initx, dtype=np.float64).ravel().copy()
    nframes = data.shape[1]
    fit_params = np.zeros((nframes, 6), dtype=np.float64)
    resid = np.zeros(freq_arr.size, dtype=np.float64)

    for jj in range(nframes):
        y_real = np.real(data[:, jj])
        curr_init = lsqcurvefit(
            lorentz_model,
            curr_init,
            freq_arr,
            y_real,
            max_iter=800,
            tol_x=1e-4,
            tol_fun=1e-4,
        )
        nlin_res = nlinfit(
            freq_arr,
            y_real,
            lorentz_model,
            curr_init,
            max_iter=400,
            tol_x=1e-6,
            tol_fun=1e-6,
        )
        fit_params[jj, :] = nlin_res.beta
        resid = nlin_res.resid

    fit_params = _unwrap_lorentz_params(fit_params, wrap_half_pi=False)
    fit_params[:, 0] = np.abs(fit_params[:, 0])
    reject_frame = _reject_outlier_frames(fit_params)

    result: dict[str, NDArray[np.float64]] = {
        "fit_params": fit_params.squeeze() if nframes == 1 else fit_params,
        "reject_frame": reject_frame,
        "resid": resid,
    }
    return result
