"""GannetFitPhantom module: multi-Lorentzian phantom fitting and quantification."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct
from gannet.fitting.optimizers import lsqcurvefit, nlinfit
from gannet.fitting.quantification_helpers import calc_conc_phantom
from gannet.models.phantom_lineshapes import (
    phantom_six_lorentz_model,
    phantom_three_lorentz_model,
    phantom_two_lorentz_model,
)
from gannet.pipeline.fit import _ensure_list_len, _fit_water_peak
from gannet.processing.noise import calc_noise
from gannet.utils.export import export_to_csv, export_to_html, save_mrs_mat


def _fit_phantom_target(
    freq: NDArray[np.float64],
    diff_spec: NDArray[np.complex128],
    targ: str,
    larmor: float,
) -> dict[str, Any]:
    """Fit one edited phantom multiplet target using multi-Lorentzian models.

    Parameters
    ------------
    freq: NDArray[np.float64]
        Chemical shift vector in ppm.
    diff_spec: NDArray[np.complex128]
        Complex edited difference spectrum for one scan.
    targ: str
        Target metabolite name (``GABA``, ``Glx``, ``GSH``, ``Lac``, ``EtOH``).
    larmor: float
        Larmor frequency in MHz.

    Returns
    --------
        dict[str, Any]:
            Fit metrics ``Area``, ``FWHM``, ``SNR``, ``FitError``, ``ModelParam``,
            ``fit_freq``, ``fit_spec``, and ``resid``.
    """
    real_diff = np.real(diff_spec)
    noise = calc_noise(freq, diff_spec)

    if targ in ("GABA", "EtOH"):
        f_min, f_max = (2.75, 3.30) if targ == "GABA" else (0.85, 1.55)
        cen = 3.01 if targ == "GABA" else 1.18
        j_init = 7.3 / larmor
        mask = (freq >= f_min) & (freq <= f_max)
        fb = freq[mask]
        sb = real_diff[mask]
        mx = max(abs(float(np.max(sb))), 1e-6)
        t2_0 = 50.0
        amp_0 = mx / t2_0
        # [Ha, Hb, H0, T2, f0, J, phi_a, phi_b, phi_0, M, C]
        initx = np.array(
            [
                amp_0 * 0.5,
                amp_0 * 0.5,
                amp_0,
                t2_0,
                cen,
                j_init,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        lb = np.array(
            [
                0.0,
                0.0,
                0.0,
                5.0,
                cen - 0.15,
                j_init * 0.5,
                -np.pi,
                -np.pi,
                -np.pi,
                -10.0 * mx,
                -10.0 * mx,
            ]
        )
        ub = np.array(
            [
                4.0 * amp_0 * 5.0,
                4.0 * amp_0 * 5.0,
                4.0 * amp_0 * 5.0,
                300.0,
                cen + 0.15,
                j_init * 1.5,
                np.pi,
                np.pi,
                np.pi,
                10.0 * mx,
                10.0 * mx,
            ]
        )
        lsq_p = lsqcurvefit(phantom_three_lorentz_model, initx, fb, sb, lb=lb, ub=ub)
        res_nl = nlinfit(fb, sb, phantom_three_lorentz_model, lsq_p)
        params = res_nl.beta
        fit_curve = phantom_three_lorentz_model(params, fb)
        area = float(np.sum(np.abs(params[:3])) * np.pi)
        fwhm = float(abs(larmor / (np.pi * max(params[3], 1e-6) / 1.0)))
        height = float(np.max(np.abs(params[:3])) * params[3])

    elif targ in ("Glx", "Lac"):
        f_min, f_max = (3.55, 3.95) if targ == "Glx" else (1.05, 1.60)
        cen = 3.75 if targ == "Glx" else 1.33
        j_init = 3.5 / larmor if targ == "Glx" else 6.9 / larmor
        mask = (freq >= f_min) & (freq <= f_max)
        fb = freq[mask]
        sb = real_diff[mask]
        mx = max(abs(float(np.max(sb))), 1e-6)
        t2_0 = 50.0
        amp_0 = mx / t2_0
        # [Ha, Hb, T2, f0, J, phi_a, phi_b, M, C]
        initx = np.array([amp_0, amp_0, t2_0, cen, j_init, 0.0, 0.0, 0.0, 0.0])
        lb = np.array(
            [
                0.0,
                0.0,
                5.0,
                cen - 0.15,
                j_init * 0.4,
                -np.pi,
                -np.pi,
                -10.0 * mx,
                -10.0 * mx,
            ]
        )
        ub = np.array(
            [
                10.0 * amp_0,
                10.0 * amp_0,
                300.0,
                cen + 0.15,
                j_init * 1.8,
                np.pi,
                np.pi,
                10.0 * mx,
                10.0 * mx,
            ]
        )
        lsq_p = lsqcurvefit(phantom_two_lorentz_model, initx, fb, sb, lb=lb, ub=ub)
        res_nl = nlinfit(fb, sb, phantom_two_lorentz_model, lsq_p)
        params = res_nl.beta
        fit_curve = phantom_two_lorentz_model(params, fb)
        area = float(np.sum(np.abs(params[:2])) * np.pi)
        fwhm = float(abs(larmor / max(np.pi * params[2], 1e-6)))
        height = float(np.max(np.abs(params[:2])) * params[2])

    elif targ == "GSH":
        mask = (freq >= 2.75) & (freq <= 3.15)
        fb = freq[mask]
        sb = real_diff[mask]
        mx = max(abs(float(np.max(sb))), 1e-6)
        t2_0 = 50.0
        amp_0 = mx / t2_0
        # [Ha..Hf (6), T2, f0, J, phi_a..phi_f (6), M, C]
        initx = np.array(
            [amp_0 * 0.5] * 6 + [t2_0, 2.95, 2.5 / larmor] + [0.0] * 6 + [0.0, 0.0],
            dtype=np.float64,
        )
        lb = np.array(
            [0.0] * 6 + [5.0, 2.85, 0.005] + [-np.pi] * 6 + [-10.0 * mx, -10.0 * mx],
            dtype=np.float64,
        )
        ub = np.array(
            [10.0 * amp_0] * 6
            + [300.0, 3.05, 0.08]
            + [np.pi] * 6
            + [10.0 * mx, 10.0 * mx],
            dtype=np.float64,
        )
        lsq_p = lsqcurvefit(phantom_six_lorentz_model, initx, fb, sb, lb=lb, ub=ub)
        res_nl = nlinfit(fb, sb, phantom_six_lorentz_model, lsq_p)
        params = res_nl.beta
        fit_curve = phantom_six_lorentz_model(params, fb)
        area = float(np.sum(np.abs(params[:6])) * np.pi)
        fwhm = float(abs(larmor / max(np.pi * params[6], 1e-6)))
        height = float(np.max(np.abs(params[:6])) * params[6])
    else:
        raise ValueError(f"Unsupported phantom fit target: {targ}")

    resid = sb - fit_curve
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


def gannet_fit_phantom(
    mrs_struct: MRSStruct,
    save_csv: bool = True,
    save_html: bool = True,
    out_dir: str | None = None,
) -> MRSStruct:
    """Fit phantom edited MRS peaks and compute absolute mM concentration.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Processed ``MRSStruct`` returned by ``gannet_load`` (with ``p.phantom=1``).
    save_csv: bool
        Export results to ``GannetFitPhantom_output.csv`` when ``True``.
    save_html: bool
        Generate standalone HTML report when ``True``.
    out_dir: str | None
        Destination directory for CSV/HTML/MAT outputs.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` containing phantom fit parameters and ``Conc``.
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
        out_vox.setdefault("water", {})
        for targ in targets:
            out_vox.setdefault(targ, {})

        for ii in range(n_scans):
            mrs_struct["ii"] = ii
            larmor = float(p["LarmorFreq"][ii]) or 127.7

            w_all = np.asarray(spec_vox.get("water", []), dtype=np.complex128)
            if w_all.size > 0:
                w_1d = w_all[ii] if w_all.ndim == 2 else w_all
                w_res = _fit_water_peak(freq, w_1d, larmor)
                for k, v in w_res.items():
                    _ensure_list_len(out_vox["water"], k, ii, default=float("nan"))
                    out_vox["water"][k][ii] = v

            for targ in targets:
                diff_all = np.asarray(
                    spec_vox.get(targ, {}).get("diff", []), dtype=np.complex128
                )
                if diff_all.size == 0:
                    continue
                diff_1d = diff_all[ii] if diff_all.ndim == 2 else diff_all
                fit_res = _fit_phantom_target(freq, diff_1d, targ, larmor)
                for k, v in fit_res.items():
                    _ensure_list_len(out_vox[targ], k, ii, default=float("nan"))
                    out_vox[targ][k][ii] = v
                if out_vox["water"].get("Area"):
                    mrs_struct = calc_conc_phantom(mrs_struct, vox, targ, ii)

    dest_dir = out_dir or os.getcwd()
    if save_csv or bool(p.get("csv", False)):
        export_to_csv(mrs_struct, caller="phantom", out_dir=dest_dir)
    if bool(p.get("mat", False)):
        save_mrs_mat(mrs_struct, os.path.join(dest_dir, "MRS_struct.mat"))
    if save_html:
        export_to_html(
            mrs_struct,
            out_path=os.path.join(dest_dir, "GannetFitPhantom_report.html"),
            title="GannetFitPhantom Report",
        )
    return mrs_struct
