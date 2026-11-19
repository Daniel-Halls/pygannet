"""GannetQuantify module: Gasparovic and Harris tissue-corrected quantification."""

from __future__ import annotations

import os

import numpy as np

from gannet.config import MRSStruct
from gannet.fitting.quantification_helpers import get_in_vivo_metabolite_params
from gannet.pipeline.fit import _ensure_list_len
from gannet.utils.export import export_to_csv, export_to_html, save_mrs_mat


def _water_tissue_params(larmor_mhz: float) -> dict[str, dict[str, float]]:
    """Return literature tissue water T1/T2 (s) and Ernst densities by field strength.

    Matches Wansapura et al. (1999) for 3T and 1.5T/7T parameters from
    ``GannetQuantify.m``.

    Parameters
    ------------
    larmor_mhz: float
        Larmor proton frequency in MHz.

    Returns
    --------
        dict[str, dict[str, float]]:
            Nested dictionary for ``GM``, ``WM``, and ``CSF`` with keys
            ``T1``, ``T2``, and ``conc`` (molar concentration * 1e3).
    """
    if abs(larmor_mhz - 63.8) < 15.0:
        # 1.5T
        return {
            "GM": {"T1": 1.120, "T2": 0.095, "conc": 43.30e3},
            "WM": {"T1": 0.884, "T2": 0.072, "conc": 36.08e3},
            "CSF": {"T1": 3.817, "T2": 0.503, "conc": 53.84e3},
        }
    if abs(larmor_mhz - 298.0) < 40.0:
        # 7T
        return {
            "GM": {"T1": 2.130, "T2": 0.050, "conc": 43.30e3},
            "WM": {"T1": 1.220, "T2": 0.055, "conc": 36.08e3},
            "CSF": {"T1": 4.425, "T2": 0.141, "conc": 53.84e3},
        }
    # Default 3T (Wansapura et al., 1999; Lu et al., 2005; Piechnik et al., 2009)
    return {
        "GM": {"T1": 1.331, "T2": 0.110, "conc": 43.30e3},
        "WM": {"T1": 0.832, "T2": 0.0792, "conc": 36.08e3},
        "CSF": {"T1": 3.817, "T2": 0.503, "conc": 53.84e3},
    }


def gannet_quantify(
    mrs_struct: MRSStruct,
    alpha: float = 0.5,
    save_csv: bool = True,
    save_html: bool = True,
    out_dir: str | None = None,
) -> MRSStruct:
    """Compute tissue- and alpha-corrected concentrations (``GannetQuantify.m``).

    Implements both Gasparovic et al. (2006) relaxation/water-density tissue
    correction (``ConcIU_TissCorr``) and Harris et al. (2015) alpha GM/WM
    tissue correction + group normalization (``ConcIU_AlphaTissCorr`` and
    ``ConcIU_AlphaTissCorr_GrpNorm``).

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` populated by ``gannet_segment``.
    alpha: float
        Assumed WM:GM metabolite concentration ratio (default ``0.5``).
    save_csv: bool
        Export results to ``GannetQuantify_output.csv`` when ``True``.
    save_html: bool
        Generate self-contained HTML report when ``True``.
    out_dir: str | None
        Optional directory for saved CSV/HTML/MAT artifacts.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with ``ConcIU_TissCorr``,
            ``ConcIU_AlphaTissCorr``, and ``ConcIU_AlphaTissCorr_GrpNorm``.
    """
    p = mrs_struct["p"]
    n_scans = int(p["numScans"])
    vox_names = p.get("Vox", ["vox1"])[: int(p["numVox"])]
    targets = list(p.get("target", ["GABA"]))

    for vox in vox_names:
        out_vox = mrs_struct.get("out", {}).get(vox, {})
        tissue = out_vox.get("tissue", {})
        water_out = out_vox.get("water", {})
        f_gm_list = tissue.get("fGM", [])
        f_wm_list = tissue.get("fWM", [])
        f_csf_list = tissue.get("fCSF", [])
        w_area_list = water_out.get("Area", [])

        if not f_gm_list:
            continue

        mean_fgm = float(np.nanmean(np.asarray(f_gm_list[:n_scans], dtype=np.float64)))
        mean_fwm = float(np.nanmean(np.asarray(f_wm_list[:n_scans], dtype=np.float64)))

        for targ in targets:
            t_out = out_vox.get(targ, {})
            m_area_list = t_out.get("Area", [])
            if not m_area_list or not w_area_list:
                continue

            m_params = get_in_vivo_metabolite_params(targ)
            tr_w_arr = p.get("TR_water")
            te_w_arr = p.get("TE_water")

            for ii in range(n_scans):
                mrs_struct["ii"] = ii
                if (
                    ii >= len(m_area_list)
                    or ii >= len(w_area_list)
                    or ii >= len(f_gm_list)
                ):
                    continue

                f_gm = float(f_gm_list[ii])
                f_wm = float(f_wm_list[ii])
                f_csf = float(f_csf_list[ii])
                larmor = float(p["LarmorFreq"][ii]) or 127.7
                tr_s = (float(p["TR"][ii]) or 2000.0) / 1000.0
                te_s = (float(p["TE"][ii]) or 68.0) / 1000.0
                tr_w_val = (
                    float(tr_w_arr[ii])
                    if tr_w_arr is not None and ii < len(tr_w_arr) and tr_w_arr[ii] > 0
                    else (float(p["TR"][ii]) or 2000.0)
                )
                te_w_val = (
                    float(te_w_arr[ii])
                    if te_w_arr is not None and ii < len(te_w_arr) and te_w_arr[ii] > 0
                    else (float(p["TE"][ii]) or 68.0)
                )
                tr_w_s = tr_w_val / 1000.0
                te_w_s = te_w_val / 1000.0

                t_pars = _water_tissue_params(larmor)

                # Relaxation attenuation R for water across GM, WM, CSF
                r_gm = (1.0 - np.exp(-tr_w_s / t_pars["GM"]["T1"])) * np.exp(
                    -te_w_s / t_pars["GM"]["T2"]
                )
                r_wm = (1.0 - np.exp(-tr_w_s / t_pars["WM"]["T1"])) * np.exp(
                    -te_w_s / t_pars["WM"]["T2"]
                )
                r_csf = (1.0 - np.exp(-tr_w_s / t_pars["CSF"]["T1"])) * np.exp(
                    -te_w_s / t_pars["CSF"]["T2"]
                )
                r_met = (1.0 - np.exp(-tr_s / m_params.t1_metab)) * np.exp(
                    -te_s / m_params.t2_metab
                )

                # Mole fractions of water in GM, WM, CSF (Gasparovic et al. 2006)
                c_gm = t_pars["GM"]["conc"]
                c_wm = t_pars["WM"]["conc"]
                c_csf = t_pars["CSF"]["conc"]
                molal_conc_w = 55.51e3
                denom_moles = f_gm * c_gm + f_wm * c_wm + f_csf * c_csf
                if denom_moles <= 0 or f_gm + f_wm <= 0:
                    continue
                f_gm_mol = (f_gm * c_gm) / denom_moles
                f_wm_mol = (f_wm * c_wm) / denom_moles
                f_csf_mol = (f_csf * c_csf) / denom_moles

                s_m = float(m_area_list[ii])
                s_w = float(w_area_list[ii])
                ed_eff = max(m_params.editing_efficiency, 1e-8)
                mm_fac = m_params.mm_factor
                n_scale = 2.0 / max(float(m_params.n_h_metab), 1.0)
                if s_w == 0:
                    continue

                # 1. Gasparovic tissue-corrected concentration (ConcIU_TissCorr)
                numer_g = f_gm_mol * r_gm + f_wm_mol * r_wm + f_csf_mol * r_csf
                conc_tiss = (
                    (s_m / s_w)
                    * n_scale
                    * mm_fac
                    / ed_eff
                    * molal_conc_w
                    * (numer_g / max((1.0 - f_csf_mol) * r_met, 1e-8))
                )

                # 2. Harris alpha-corrected tissue concentration (Harris et al. 2015)
                alpha_m = (
                    float(m_params.c_wm / m_params.c_gm) if m_params.c_gm > 0 else alpha
                )
                numer_h = (
                    f_gm * c_gm * r_gm + f_wm * c_wm * r_wm + f_csf * c_csf * r_csf
                )
                conc_harris = (
                    (s_m / s_w)
                    * n_scale
                    * mm_fac
                    / ed_eff
                    * (numer_h / max(r_met, 1e-8))
                )
                conc_alpha = conc_harris / max(f_gm + alpha_m * f_wm, 1e-6)
                grp_factor = (mean_fgm + alpha_m * mean_fwm) / max(
                    (f_gm + alpha_m * f_wm) * (mean_fgm + mean_fwm), 1e-6
                )
                conc_alpha_grp = conc_harris * grp_factor
                t_out["alpha"] = alpha_m

                for k, v in (
                    ("ConcIU_TissCorr", float(conc_tiss)),
                    ("ConcIU_AlphaTissCorr", float(conc_alpha)),
                    ("ConcIU_AlphaTissCorr_GrpNorm", float(conc_alpha_grp)),
                ):
                    _ensure_list_len(t_out, k, ii, default=float("nan"))
                    t_out[k][ii] = v

    dest_dir = out_dir or os.getcwd()
    if save_csv or bool(p.get("csv", False)):
        export_to_csv(mrs_struct, caller="quantify", out_dir=dest_dir)
    if bool(p.get("mat", False)):
        save_mrs_mat(mrs_struct, os.path.join(dest_dir, "MRS_struct.mat"))
    if save_html:
        export_to_html(
            mrs_struct,
            out_path=os.path.join(dest_dir, "GannetQuantify_report.html"),
            title="GannetQuantify Final Report",
        )
    return mrs_struct
