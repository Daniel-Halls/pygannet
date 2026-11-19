"""GannetCoRegister module: align MRS voxel coordinates onto T1 structural MRI."""

from __future__ import annotations

import os

from gannet.config import MRSStruct
from gannet.coreg.masks import (
    gannet_mask_ge,
    gannet_mask_ge_nii,
    gannet_mask_nifti,
    gannet_mask_philips,
    gannet_mask_siemens_rda,
    gannet_mask_siemens_twix,
)
from gannet.utils.export import export_to_html, save_mrs_mat


def _resolve_structural_list(
    struc: str | list[str],
    n_scans: int,
) -> list[str]:
    """Normalize structural image path(s) to match the scan count ``n_scans``.

    Parameters
    ------------
    struc: str | list[str]
        Single structural T1 NIfTI/DICOM path or list of paths per scan.
    n_scans: int
        Number of MRS scans in ``MRSStruct``.

    Returns
    --------
        list[str]:
            Structural file/folder paths of length ``n_scans``.
    """
    if isinstance(struc, str):
        return [struc] * n_scans
    return list(struc)


def gannet_co_register(
    mrs_struct: MRSStruct,
    struc: str | list[str],
    save_html: bool = True,
    out_dir: str | None = None,
) -> MRSStruct:
    """Co-register MRS voxel geometry onto structural T1 MRI (``GannetCoRegister.m``).

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` populated by ``gannet_load`` or ``gannet_fit``.
    struc: str | list[str]
        Path(s) to structural T1 NIfTI files (or DICOM folders for GE).
    save_html: bool
        Whether to generate an updated HTML report including 3-plane voxel overlays.
    out_dir: str | None
        Optional directory for saved HTML/MAT artifacts.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with ``mask.<vox>`` masks and overlay slices.
    """
    p = mrs_struct["p"]
    n_scans = int(p["numScans"])
    n_vox = int(p["numVox"])
    vendor = str(p.get("vendor", ""))
    metab_files = mrs_struct.get("metabfile", [])
    struc_list = _resolve_structural_list(struc, n_scans)
    mrs_struct["mask"]["T1image"] = struc_list

    for ii in range(n_scans):
        mrs_struct["ii"] = ii
        m_file = metab_files[ii] if ii < len(metab_files) else ""
        t1_path = struc_list[ii]
        for kk in range(n_vox):
            if vendor in ("Philips", "Philips_data", "Philips_raw"):
                mrs_struct = gannet_mask_philips(
                    m_file, t1_path, mrs_struct, ii, vox_idx=kk
                )
            elif vendor == "Siemens_twix":
                mrs_struct = gannet_mask_siemens_twix(
                    m_file, t1_path, mrs_struct, ii, vox_idx=kk
                )
            elif vendor == "Siemens_rda":
                mrs_struct = gannet_mask_siemens_rda(
                    m_file, t1_path, mrs_struct, ii, vox_idx=kk
                )
            elif vendor == "NIfTI":
                mrs_struct = gannet_mask_nifti(
                    m_file, t1_path, mrs_struct, ii, vox_idx=kk
                )
            elif vendor == "GE":
                if os.path.isdir(t1_path):
                    mrs_struct = gannet_mask_ge(
                        m_file, t1_path, mrs_struct, ii, vox_idx=kk
                    )
                else:
                    mrs_struct = gannet_mask_ge_nii(
                        m_file, t1_path, mrs_struct, ii, vox_idx=kk
                    )
            else:
                # Generic fallback: NIfTI if MRS is .nii, else center box
                if m_file.lower().endswith((".nii", ".nii.gz")):
                    mrs_struct = gannet_mask_nifti(
                        m_file, t1_path, mrs_struct, ii, vox_idx=kk
                    )
                else:
                    mrs_struct = gannet_mask_ge_nii(
                        m_file, t1_path, mrs_struct, ii, vox_idx=kk
                    )

    dest_dir = out_dir or os.getcwd()
    if bool(p.get("mat", False)):
        save_mrs_mat(mrs_struct, os.path.join(dest_dir, "MRS_struct.mat"))
    if save_html:
        export_to_html(
            mrs_struct,
            out_path=os.path.join(dest_dir, "GannetCoRegister_report.html"),
            title="GannetCoRegister Report",
        )
    return mrs_struct
