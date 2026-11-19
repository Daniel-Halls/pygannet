"""GannetFit module: in-vivo MRS peak fitting and concentration quantification."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct
from gannet.fitting.optimizers import lsqcurvefit, nlinfit
from gannet.fitting.peak_fitters import fit_cho_cr, fit_cr
from gannet.fitting.quantification_helpers import calc_iu
from gannet.models.lineshapes import (
    etoh_model,
    five_gauss_model,
    gaba_glx_model,
    gauss_model,
    lac_model,
    lorentz_gauss_model_p,
    lorentz_model,
    six_gauss_model,
    three_lorentz_model,
)
from gannet.processing.noise import calc_noise
from gannet.utils.export import export_to_csv, export_to_html, save_mrs_mat


def _ensure_list_len(
    container: dict[str, Any],
    key: str,
    idx: int,
    default: Any = 0.0,
) -> None:
    """Extend list field ``container[key]`` so index ``idx`` exists.

    Parameters
    ------------
    container: dict[str, Any]
        Target output sub-dictionary.
    key: str
        Key storing a Python list.
    idx: int
        0-based scan index required.
    default: Any
        Filler value for newly allocated slots.

    Returns
    --------
        None:
            Modifies ``container`` in place.
    """
    if key not in container or not isinstance(container[key], list):
        container[key] = []
    while len(container[key]) <= idx:
        container[key].append(default)


def _weighted_fit_target(
    freq_bounds: NDArray[np.float64],
    spec_bounds: NDArray[np.float64],
    weights: NDArray[np.float64],
    model_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    initx: NDArray[np.float64],
    lb: NDArray[np.float64],
    ub: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run two-stage weighted ``lsqcurvefit`` followed by ``nlinfit``.

    Parameters
    ------------
    freq_bounds: NDArray[np.float64]
        Chemical shift vector over the fitting window (ppm).
    spec_bounds: NDArray[np.float64]
        Real-part spectral intensities over the fitting window.
    weights: NDArray[np.float64]
        Per-point regression weights matching ``freq_bounds``.
    model_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Lineshape callable ``model_fn(params, freq)``.
    initx: NDArray[np.float64]
        Initial guess vector.
    lb: NDArray[np.float64]
        Lower parameter bounds.
    ub: NDArray[np.float64]
        Upper parameter bounds.

    Returns
    --------
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            Fitted parameter vector and unweighted residual on ``freq_bounds``.
    """
    w_sqrt = np.sqrt(weights)
    target_w = w_sqrt * spec_bounds
    lsq_params = lsqcurvefit(
        lambda x, f: w_sqrt * model_fn(x, f),
        initx,
        freq_bounds,
        target_w,
        lb=lb,
        ub=ub,
    )
    res_nl = nlinfit(
        freq_bounds,
        target_w,
        lambda x, f: w_sqrt * model_fn(x, f),
        lsq_params,
    )
    in_bounds = np.all((res_nl.beta >= lb) & (res_nl.beta <= ub))
    best_params = res_nl.beta if in_bounds else lsq_params
    resid = spec_bounds - model_fn(best_params, freq_bounds)
    return (best_params, resid)


def _fit_edited_target(
    freq: NDArray[np.float64],
    diff_spec: NDArray[np.complex128],
    targ: str,
    te_ms: float,
    larmor: float,
) -> dict[str, Any]:
    """Fit one edited difference target (GABA, GABAGlx, Glx, GSH, Lac, EtOH).

    Parameters
    ------------
    freq: NDArray[np.float64]
        Chemical shift axis (ppm).
    diff_spec: NDArray[np.complex128]
        Complex difference spectrum for one scan.
    targ: str
        Target metabolite identifier.
    te_ms: float
        Echo time in milliseconds.
    larmor: float
        Larmor frequency in MHz.

    Returns
    --------
        dict[str, Any]:
            Fit metrics (``Area``, ``FWHM``, ``SNR``, ``FitError``, ``ModelParam``,
            ``fit_freq``, ``fit_spec``, ``resid``).
    """
    real_diff = np.real(diff_spec)
    noise = calc_noise(freq, diff_spec)

    if targ == "GABA":
        mask = (freq >= 2.79) & (freq <= 3.55)
        fb = freq[mask]
        sb = real_diff[mask]
        win = (fb >= 2.85) & (fb <= 3.18)
        maxin = max(
            abs(float(np.max(sb[win])) if np.any(win) else float(np.max(sb))), 1e-6
        )
        grad_init = (
            (sb[0] - sb[-1]) / (fb[0] - fb[-1])
            if fb.size > 1 and abs(fb[0] - fb[-1]) > 1e-6
            else 0.0
        )
        # gauss_model: [amp, width(<0), f0, slope, offset]
        initx = np.array([maxin, -90.0, 3.01, grad_init, float(np.median(sb))])
        lb = np.array([0.0, -500.0, 2.85, -10.0 * maxin, -10.0 * maxin])
        ub = np.array([4.0 * maxin, -10.0, 3.18, 10.0 * maxin, 10.0 * maxin])
        weights = np.ones_like(fb)
        weights[(fb >= 3.16) & (fb <= 3.285)] = 0.001
        params, resid = _weighted_fit_target(
            fb, sb, weights, gauss_model, initx, lb, ub
        )
        area = float(params[0] / np.sqrt(-min(params[1], -1e-8)) * np.sqrt(np.pi))
        fwhm = float(abs(2.0 * larmor * np.sqrt(-np.log(2.0) / min(params[1], -1e-8))))
        height = float(params[0])
        fit_curve = gauss_model(params, fb)

    elif targ in ("GABAGlx", "Glx"):
        mask = (freq >= 2.79) & (freq <= 4.10)
        fb = freq[mask]
        sb = real_diff[mask]
        w_gaba = (fb >= 2.79) & (fb <= 3.20)
        w_glx = (fb >= 3.40) & (fb <= 4.10)
        max_gaba = max(
            float(np.max(sb[w_gaba])) if np.any(w_gaba) else float(np.max(sb)),
            1e-6,
        )
        max_glx = max(
            float(np.max(sb[w_glx])) if np.any(w_glx) else float(np.max(sb)),
            1e-6,
        )
        grad_points = (sb[-1] - sb[0]) / abs(fb.size - 1) if fb.size > 1 else 0.0
        step_ppm = abs(float(freq[0] - freq[1])) if freq.size > 1 else 1e-4
        linear_init = grad_points / max(step_ppm, 1e-8)

        # Normalize by max_glx matching GannetFit.m to avoid numerical underflow
        scale_g = max(max_glx, max_gaba)
        initx = np.array(
            [
                max_glx / scale_g,
                -700.0,
                3.71,
                max_glx / scale_g,
                -700.0,
                3.79,
                max_gaba / scale_g,
                -90.0,
                3.02,
                -linear_init / scale_g,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        r_gab = max_gaba / scale_g
        r_glx = max_glx / scale_g
        lb = np.array(
            [
                -4000.0 * r_glx,
                -1000.0,
                3.69,
                -4000.0 * r_glx,
                -1000.0,
                3.77,
                -4000.0 * r_gab,
                -200.0,
                2.97,
                -40.0 * r_gab,
                -2000.0 * r_gab,
                -2000.0 * r_gab,
            ],
            dtype=np.float64,
        )
        ub = np.array(
            [
                4000.0 * r_glx,
                -40.0,
                3.73,
                4000.0 * r_glx,
                -40.0,
                3.81,
                4000.0 * r_gab,
                -40.0,
                3.07,
                40.0 * r_gab,
                1000.0 * r_gab,
                1000.0 * r_gab,
            ],
            dtype=np.float64,
        )
        weights = np.ones_like(fb)
        weights[(fb >= 3.16) & (fb <= 3.285)] = 0.001
        params_s, resid_s = _weighted_fit_target(
            fb, sb / scale_g, weights, gaba_glx_model, initx, lb, ub
        )
        params = params_s.copy()
        params[[0, 3, 6, 9, 10, 11]] = params[[0, 3, 6, 9, 10, 11]] * scale_g
        resid = resid_s * scale_g
        if targ == "Glx":
            area = float(
                (
                    params[0] / np.sqrt(-min(params[1], -1e-8))
                    + params[3] / np.sqrt(-min(params[4], -1e-8))
                )
                * np.sqrt(np.pi)
            )
            fwhm = float(
                abs(2.0 * larmor * np.sqrt(-np.log(2.0) / min(params[1], -1e-8)))
            )
            height = float(max(params[0], params[3]))
        else:
            area = float(params[6] / np.sqrt(-min(params[7], -1e-8)) * np.sqrt(np.pi))
            fwhm = float(
                abs(2.0 * larmor * np.sqrt(-np.log(2.0) / min(params[7], -1e-8)))
            )
            height = float(params[6])
        fit_curve = gaba_glx_model(params, fb)

    elif targ == "GSH":
        mask = (freq >= 2.25) & (freq <= 3.50)
        fb = freq[mask]
        sb = real_diff[mask]
        max_gsh = max(
            abs(
                float(np.max(sb[(fb >= 2.85) & (fb <= 3.05)]))
                if np.any((fb >= 2.85) & (fb <= 3.05))
                else float(np.max(sb))
            ),
            1e-6,
        )
        min_asp = (
            float(np.min(sb[(fb >= 2.40) & (fb <= 2.80)]))
            if np.any((fb >= 2.40) & (fb <= 2.80))
            else -0.5 * max_gsh
        )
        if te_ms < 100.0:
            initx = np.array(
                [
                    max_gsh,
                    -90.0,
                    2.95,
                    min_asp,
                    -90.0,
                    2.73,
                    min_asp * 0.8,
                    -90.0,
                    2.61,
                    min_asp * 0.5,
                    -90.0,
                    2.55,
                    min_asp * 0.5,
                    -90.0,
                    2.45,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            lb = np.array(
                [
                    0.0,
                    -500.0,
                    2.88,
                    -4.0 * abs(min_asp) - max_gsh,
                    -500.0,
                    2.68,
                    -4.0 * abs(min_asp) - max_gsh,
                    -500.0,
                    2.58,
                    -4.0 * abs(min_asp) - max_gsh,
                    -500.0,
                    2.50,
                    -4.0 * abs(min_asp) - max_gsh,
                    -500.0,
                    2.38,
                    -10.0 * max_gsh,
                    -10.0 * max_gsh,
                    -10.0 * max_gsh,
                ]
            )
            ub = np.array(
                [
                    4.0 * max_gsh,
                    -10.0,
                    3.05,
                    4.0 * abs(min_asp) + max_gsh,
                    -10.0,
                    2.80,
                    4.0 * abs(min_asp) + max_gsh,
                    -10.0,
                    2.67,
                    4.0 * abs(min_asp) + max_gsh,
                    -10.0,
                    2.60,
                    4.0 * abs(min_asp) + max_gsh,
                    -10.0,
                    2.50,
                    10.0 * max_gsh,
                    10.0 * max_gsh,
                    10.0 * max_gsh,
                ]
            )
            model_fn = five_gauss_model
        else:
            initx = np.array(
                [
                    max_gsh,
                    -90.0,
                    2.95,
                    min_asp,
                    -90.0,
                    2.75,
                    min_asp * 0.8,
                    -90.0,
                    2.66,
                    min_asp * 0.8,
                    -90.0,
                    2.59,
                    min_asp * 0.5,
                    -90.0,
                    2.51,
                    min_asp * 0.5,
                    -90.0,
                    2.43,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            span = max(abs(min_asp), max_gsh) * 4.0
            lb = np.full(21, -span)
            ub = np.full(21, span)
            lb[0], ub[0] = (0.0, 4.0 * max_gsh)
            for idx_w in range(1, 18, 3):
                lb[idx_w], ub[idx_w] = (-500.0, -10.0)
            for idx_f, cen in zip(
                range(2, 18, 3), (2.95, 2.75, 2.66, 2.59, 2.51, 2.43)
            ):
                lb[idx_f], ub[idx_f] = (cen - 0.06, cen + 0.06)
            model_fn = six_gauss_model
        params, resid = _weighted_fit_target(
            fb, sb, np.ones_like(fb), model_fn, initx, lb, ub
        )
        area = float(params[0] / np.sqrt(-min(params[1], -1e-8)) * np.sqrt(np.pi))
        fwhm = float(abs(2.0 * larmor * np.sqrt(-np.log(2.0) / min(params[1], -1e-8))))
        height = float(params[0])
        fit_curve = model_fn(params, fb)

    elif targ == "Lac":
        mask = (freq >= 0.8) & (freq <= 1.8)
        fb = freq[mask]
        sb = real_diff[mask]
        max_lac = max(
            abs(
                float(np.max(sb[(fb >= 1.15) & (fb <= 1.50)]))
                if np.any((fb >= 1.15) & (fb <= 1.50))
                else float(np.max(sb))
            ),
            1e-6,
        )
        # lac_model: [a_lac, w_lac, f_lac, a_bhb, w_bhb, f_bhb, b0..b2, a_mm3]
        initx = np.array(
            [max_lac * 0.5, -300.0, 1.33, 0.0, -300.0, 1.19, 0.0, 0.0, 0.0, 0.0]
        )
        lb = np.array(
            [
                0.0,
                -1500.0,
                1.25,
                -4.0 * max_lac,
                -1500.0,
                1.12,
                -10.0 * max_lac,
                -10.0 * max_lac,
                -10.0 * max_lac,
                -4.0 * max_lac,
            ]
        )
        ub = np.array(
            [
                4.0 * max_lac,
                -10.0,
                1.40,
                4.0 * max_lac,
                -10.0,
                1.26,
                10.0 * max_lac,
                10.0 * max_lac,
                10.0 * max_lac,
                4.0 * max_lac,
            ]
        )
        params, resid = _weighted_fit_target(
            fb, sb, np.ones_like(fb), lac_model, initx, lb, ub
        )
        area = float(2.0 * params[0] / np.sqrt(-min(params[1], -1e-8)) * np.sqrt(np.pi))
        fwhm = float(abs(2.0 * larmor * np.sqrt(-np.log(2.0) / min(params[1], -1e-8))))
        height = float(params[0])
        fit_curve = lac_model(params, fb)

    elif targ == "EtOH":
        mask = (freq >= 0.6) & (freq <= 1.8)
        fb = freq[mask]
        sb = real_diff[mask]
        max_etoh = max(abs(float(np.max(sb))), 1e-6)
        # etoh_model: [a1, f1, w1, a2, f2, w2, slope, offset]
        initx = np.array([max_etoh, 1.17, 0.06, max_etoh * 0.5, 1.25, 0.06, 0.0, 0.0])
        lb = np.array(
            [0.0, 1.08, 0.005, 0.0, 1.16, 0.005, -10.0 * max_etoh, -10.0 * max_etoh]
        )
        ub = np.array(
            [
                4.0 * max_etoh,
                1.26,
                0.25,
                4.0 * max_etoh,
                1.35,
                0.25,
                10.0 * max_etoh,
                10.0 * max_etoh,
            ]
        )
        params, resid = _weighted_fit_target(
            fb, sb, np.ones_like(fb), etoh_model, initx, lb, ub
        )
        area = float(0.5 * np.pi * (params[0] * params[2] + params[3] * params[5]))
        fwhm = float(abs(params[2] * larmor))
        height = float(max(params[0], params[3]))
        fit_curve = etoh_model(params, fb)

    else:
        raise ValueError(f"Unsupported fit target: {targ}")

    snr = float(height / max(noise, 1e-12))
    fit_err = float(100.0 * np.std(resid) / max(height, 1e-12))
    return {
        "Area": area,
        "FWHM": fwhm,
        "SNR": snr,
        "FitError": fit_err,
        "ModelParam": params,
        "fit_freq": fb,
        "fit_spec": fit_curve,
        "resid": resid,
    }


def _fit_water_peak(
    freq: NDArray[np.float64],
    water_spec: NDArray[np.complex128],
    larmor: float,
    apply_phase_prealign: bool = False,
) -> dict[str, Any]:
    """Fit unsuppressed water peak with 7-param ``lorentz_gauss_model_p``.

    Parameters
    ------------
    freq: NDArray[np.float64]
        Chemical shift axis (ppm).
    water_spec: NDArray[np.complex128]
        Complex water spectrum.
    larmor: float
        Larmor frequency in MHz.
    apply_phase_prealign: bool
        Zero-order pre-phase spectrum to maximize real peak (used when ``~water_ECC``).

    Returns
    --------
        dict[str, Any]:
            Dictionary with ``Area``, ``FWHM``, ``SNR``, ``FitError``,
            ``ModelParam``, ``fit_freq``, ``fit_data``, and ``fit_spec``.
    """
    mask = (freq >= 3.8) & (freq <= 5.6)
    fb = freq[mask]
    w_seg = water_spec[mask]
    if apply_phase_prealign:
        best_ang = np.angle(w_seg[np.argmax(np.abs(w_seg))])
        w_seg = w_seg * np.exp(-1j * best_ang)
    wb = np.real(w_seg)
    max_w = max(float(np.max(wb)), 1e-6)
    f_peak = float(fb[np.argmax(wb)])
    # [amp, inv_hwhm, f0, slope, offset, g_w(<0), phase]
    initx = np.array([max_w, 25.0, f_peak, 0.0, 0.0, -10.0, 0.0])
    lb = np.array(
        [0.0, 1.0, f_peak - 0.3, -10.0 * max_w, -10.0 * max_w, -500.0, -np.pi]
    )
    ub = np.array(
        [5.0 * max_w, 500.0, f_peak + 0.3, 10.0 * max_w, 10.0 * max_w, 0.0, np.pi]
    )

    lsq_p = lsqcurvefit(lorentz_gauss_model_p, initx, fb, wb, lb=lb, ub=ub)
    res_nl = nlinfit(fb, wb, lorentz_gauss_model_p, lsq_p)
    params = res_nl.beta
    fit_curve = lorentz_gauss_model_p(params, fb)
    resid = wb - fit_curve

    peak_only = lorentz_gauss_model_p(
        np.array([params[0], params[1], params[2], 0.0, 0.0, params[5], 0.0]),
        fb,
    )
    trap_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(abs(trap_fn(peak_only, fb)))
    fwhm = float(abs(2.0 / max(params[1], 1e-6) * larmor))
    noise = calc_noise(freq, water_spec)
    snr = float(params[0] / max(noise, 1e-12))
    fit_err = float(100.0 * np.std(resid) / max(params[0], 1e-12))

    return {
        "Area": area,
        "FWHM": fwhm,
        "SNR": snr,
        "FitError": fit_err,
        "ModelParam": params,
        "fit_freq": fb,
        "fit_data": wb,
        "fit_spec": fit_curve,
    }


def _fit_naa_peak(
    freq: NDArray[np.float64],
    off_spec: NDArray[np.complex128],
    larmor: float,
) -> dict[str, Any]:
    """Fit 2.01 ppm NAA singlet on OFF spectrum with ``lorentz_model``.

    Parameters
    ------------
    freq: NDArray[np.float64]
        Chemical shift vector (ppm).
    off_spec: NDArray[np.complex128]
        Complex OFF spectrum.
    larmor: float
        Larmor frequency in MHz.

    Returns
    --------
        dict[str, Any]:
            Dictionary containing ``Area``, ``FWHM``, ``SNR``, ``FitError``, and
            ``ModelParam``.
    """
    mask = (freq >= 1.75) & (freq <= 2.25)
    fb = freq[mask]
    sb = np.real(off_spec[mask])
    max_n = max(float(np.max(sb)), 1e-6)
    init_area = max_n * 0.02 * 2.0 * np.pi
    initx = np.array([init_area, 0.02, 2.01, 0.0, 0.0, 0.0])
    lb = np.array([0.0, 0.001, 1.90, -np.pi, -10.0 * max_n, -10.0 * max_n])
    ub = np.array([10.0 * init_area, 0.2, 2.15, np.pi, 10.0 * max_n, 10.0 * max_n])
    lsq_p = lsqcurvefit(lorentz_model, initx, fb, sb, lb=lb, ub=ub)
    res_nl = nlinfit(fb, sb, lorentz_model, lsq_p)
    params = res_nl.beta
    area = float(abs(params[0]) * 0.5)
    fwhm = float(abs(2.0 * params[1] * larmor))
    noise = calc_noise(freq, off_spec)
    peak_h = abs(params[0]) / max(2.0 * np.pi * params[1], 1e-8)
    snr = float(peak_h / max(noise, 1e-12))
    fit_err = float(100.0 * np.std(res_nl.resid) / max(peak_h, 1e-12))
    return {
        "Area": area,
        "FWHM": fwhm,
        "SNR": snr,
        "FitError": fit_err,
        "ModelParam": params,
    }


def gannet_fit(
    mrs_struct: MRSStruct,
    save_csv: bool = True,
    save_html: bool = True,
    out_dir: str | None = None,
) -> MRSStruct:
    """Fit edited metabolites, water, Cr/Cho, and NAA peaks (``GannetFit.m``).

    Parameters
    ------------
    mrs_struct: MRSStruct
        Processed ``MRSStruct`` returned by ``gannet_load``.
    save_csv: bool
        Export results to ``GannetFit_output.csv`` when ``True``.
    save_html: bool
        Generate self-contained HTML report with diagnostic fits when ``True``.
    out_dir: str | None
        Destination directory for CSV/HTML/MAT outputs.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` populated under ``out.<vox>.*``.
    """
    p = mrs_struct["p"]
    n_scans = int(p["numScans"])
    vox_names = p.get("Vox", ["vox1"])[: int(p["numVox"])]
    targets = list(p.get("target", ["GABA"]))
    freq = np.asarray(mrs_struct["spec"]["freq"], dtype=np.float64)

    for vox in vox_names:
        mrs_struct.ensure_substruct("out", vox)
        spec_vox = mrs_struct["spec"].get(vox, {})
        out_vox = mrs_struct["out"][vox]

        for targ in targets:
            out_vox.setdefault(targ, {})
        for ref_key in ("Cr", "Cho", "water", "NAA", "Glu"):
            out_vox.setdefault(ref_key, {})

        for ii in range(n_scans):
            mrs_struct["ii"] = ii
            larmor = float(p["LarmorFreq"][ii]) or 127.7
            te_ms = float(p["TE"][ii]) or 68.0

            # 1. Fit edited targets on DIFF spectrum
            for targ in targets:
                diff_all = np.asarray(
                    spec_vox.get(targ, {}).get("diff", []), dtype=np.complex128
                )
                if diff_all.size == 0:
                    continue
                diff_1d = diff_all[ii] if diff_all.ndim == 2 else diff_all
                fit_res = _fit_edited_target(freq, diff_1d, targ, te_ms, larmor)
                t_out = out_vox[targ]
                for key, val in fit_res.items():
                    _ensure_list_len(t_out, key, ii, default=float("nan"))
                    t_out[key][ii] = val

            # 2. Fit Cr and Cho on OFF spectrum
            first_t_spec = spec_vox.get(targets[0], {}) if targets else {}
            off_src = first_t_spec.get(
                "off_off", first_t_spec.get("off", spec_vox.get("OFF", []))
            )
            off_all = np.asarray(off_src, dtype=np.complex128)
            if off_all.size > 0:
                off_1d = off_all[ii] if off_all.ndim == 2 else off_all
                cc_mask = (freq >= 2.6) & (freq <= 3.6)
                cc_fb = freq[cc_mask]
                cc_sb = off_1d[cc_mask]
                cc_real = np.real(cc_sb)
                bl_off = float((cc_real[0] + cc_real[-1]) / 2.0)
                w_est = 0.05
                a_est = (
                    max(float(np.max(cc_real) - np.min(cc_real)), 1e-6) * w_est * 4.0
                )
                cc_init = np.array(
                    [
                        a_est,
                        w_est * 2.0 * larmor,
                        3.02 * larmor,
                        0.0,
                        bl_off,
                        0.0,
                        0.5,
                    ]
                )
                cc_res = fit_cho_cr(cc_fb, cc_sb[:, np.newaxis], cc_init, larmor)
                cc_p = np.atleast_2d(cc_res["fit_params"])[0]
                cho_area = float(abs(cc_p[0] * cc_p[6]) * 0.5)
                cho_fwhm = float(abs(cc_p[1]))
                for k, v in (
                    ("Area", cho_area),
                    ("FWHM", cho_fwhm),
                    ("ModelParam", cc_p),
                ):
                    _ensure_list_len(out_vox["Cho"], k, ii, default=float("nan"))
                    out_vox["Cho"][k][ii] = v

                cr_mask = (freq >= 2.72) & (freq <= 3.12)
                cr_fb = freq[cr_mask]
                cr_sb = off_1d[cr_mask]
                max_cr = max(float(np.max(np.real(cr_sb))), 1e-6)
                cr_init = np.array(
                    [max_cr, 0.05 * 2.0 * larmor, 3.0 * larmor, 0.0, 0.0, 0.0]
                )
                cr_res = fit_cr(cr_fb, cr_sb[:, np.newaxis], cr_init, larmor)
                cr_p_scaled = np.atleast_2d(cr_res["fit_params"])[0]
                cr_p_ppm = np.array(
                    [
                        cr_p_scaled[0],
                        cr_p_scaled[1] / (2.0 * larmor),
                        cr_p_scaled[2] / larmor,
                        np.deg2rad(cr_p_scaled[3]),
                        cr_p_scaled[4],
                        cr_p_scaled[5],
                    ]
                )
                cr_area = float(abs(cc_p[0]) * 0.5)
                cr_fwhm = cho_fwhm
                cr_height = abs(cr_p_ppm[0]) / max(2.0 * np.pi * abs(cr_p_ppm[1]), 1e-8)
                cr_err = float(100.0 * np.std(cr_res["resid"]) / max(cr_height, 1e-12))
                for k, v in (
                    ("Area", cr_area),
                    ("FWHM", cr_fwhm),
                    ("FitError", cr_err),
                    ("ModelParam", cr_p_ppm),
                    ("fit_freq", cc_fb),
                    ("fit_data", cc_real),
                    ("fit_spec", lorentz_model(cr_p_ppm, cc_fb)),
                ):
                    _ensure_list_len(out_vox["Cr"], k, ii, default=float("nan"))
                    out_vox["Cr"][k][ii] = v

                naa_res = _fit_naa_peak(freq, off_1d, larmor)
                for k, v in naa_res.items():
                    _ensure_list_len(out_vox["NAA"], k, ii, default=float("nan"))
                    out_vox["NAA"][k][ii] = v

            # 3. Fit unsuppressed Water
            w_all = np.asarray(spec_vox.get("water", []), dtype=np.complex128)
            if w_all.size > 0:
                w_1d = w_all[ii] if w_all.ndim == 2 else w_all
                w_res = _fit_water_peak(
                    freq,
                    w_1d,
                    larmor,
                    apply_phase_prealign=(
                        p.get("vendor") == "Philips" and not p.get("water_ECC")
                    ),
                )
                for k, v in w_res.items():
                    _ensure_list_len(out_vox["water"], k, ii, default=float("nan"))
                    out_vox["water"][k][ii] = v

            # 4. Fit Glu on SUM if available
            sum_src = first_t_spec.get("sum", spec_vox.get("SUM", []))
            sum_all = np.asarray(sum_src, dtype=np.complex128)
            if sum_all.size > 0:
                sum_1d = sum_all[ii] if sum_all.ndim == 2 else sum_all
                glu_mask = (freq >= 2.15) & (freq <= 2.55)
                if np.any(glu_mask):
                    g_fb = freq[glu_mask]
                    g_sb = np.real(sum_1d[glu_mask])
                    mx = max(float(np.max(g_sb)), 1e-6)
                    # [H, a, b, T2, f0, J, phi, M1, M2, B]
                    g_init = np.array(
                        [mx * 0.05, 0.7, 0.7, 25.0, 2.35, 0.06, 0.0, 0.0, 0.0, 0.0]
                    )
                    g_lb = np.array(
                        [
                            0.0,
                            0.1,
                            0.1,
                            2.0,
                            2.28,
                            0.01,
                            -np.pi,
                            -10 * mx,
                            -10 * mx,
                            -10 * mx,
                        ]
                    )
                    g_ub = np.array(
                        [
                            mx,
                            3.0,
                            3.0,
                            200.0,
                            2.42,
                            0.15,
                            np.pi,
                            10 * mx,
                            10 * mx,
                            10 * mx,
                        ]
                    )
                    g_p = nlinfit(
                        g_fb,
                        g_sb,
                        three_lorentz_model,
                        lsqcurvefit(
                            three_lorentz_model, g_init, g_fb, g_sb, lb=g_lb, ub=g_ub
                        ),
                    ).beta
                    glu_area = float(abs(g_p[0]) * (1.0 + g_p[1] + g_p[2]) * np.pi)
                    _ensure_list_len(out_vox["Glu"], "Area", ii, default=float("nan"))
                    out_vox["Glu"]["Area"][ii] = glu_area

            # 5. Compute IU, Cr ratio, NAA ratio and combined FitErrors
            for targ in targets:
                t_out = out_vox[targ]
                m_area = float(t_out["Area"][ii]) if "Area" in t_out else 0.0
                m_err = float(t_out["FitError"][ii]) if "FitError" in t_out else 0.0

                w_area = out_vox["water"].get("Area", [float("nan")] * (ii + 1))[ii]
                w_err = out_vox["water"].get("FitError", [0.0] * (ii + 1))[ii]
                cr_area = out_vox["Cr"].get("Area", [float("nan")] * (ii + 1))[ii]
                cr_err = out_vox["Cr"].get("FitError", [0.0] * (ii + 1))[ii]
                naa_area = out_vox["NAA"].get("Area", [float("nan")] * (ii + 1))[ii]
                naa_err = out_vox["NAA"].get("FitError", [0.0] * (ii + 1))[ii]

                _ensure_list_len(t_out, "ConcCr", ii, default=float("nan"))
                _ensure_list_len(t_out, "ConcNAA", ii, default=float("nan"))
                _ensure_list_len(t_out, "FitError_W", ii, default=float("nan"))
                _ensure_list_len(t_out, "FitError_Cr", ii, default=float("nan"))
                _ensure_list_len(t_out, "FitError_NAA", ii, default=float("nan"))

                if cr_area and not np.isnan(cr_area):
                    t_out["ConcCr"][ii] = float(m_area / cr_area)
                    t_out["FitError_Cr"][ii] = float(np.sqrt(m_err**2 + cr_err**2))
                if naa_area and not np.isnan(naa_area):
                    t_out["ConcNAA"][ii] = float(m_area / naa_area)
                    t_out["FitError_NAA"][ii] = float(np.sqrt(m_err**2 + naa_err**2))
                if w_area and not np.isnan(w_area):
                    t_out["FitError_W"][ii] = float(np.sqrt(m_err**2 + w_err**2))
                    mrs_struct = calc_iu(mrs_struct, vox, targ, ii)

    dest_dir = out_dir or os.getcwd()
    if save_csv or bool(p.get("csv", False)):
        export_to_csv(mrs_struct, caller="fit", out_dir=dest_dir)
    if bool(p.get("mat", False)):
        save_mrs_mat(mrs_struct, os.path.join(dest_dir, "MRS_struct.mat"))
    if save_html:
        export_to_html(
            mrs_struct,
            out_path=os.path.join(dest_dir, "GannetFit_report.html"),
            title="GannetFit Spectral Modeling Report",
        )
    return mrs_struct
