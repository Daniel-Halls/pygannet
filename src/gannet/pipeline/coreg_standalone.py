"""Standalone MRS voxel co-registration and segmentation (CoRegStandAlone)."""

from __future__ import annotations

import os

import numpy as np

from gannet.config import GannetConfig, MRSStruct, gannet_pre_initialise
from gannet.io.ge import ge_read
from gannet.io.philips import parse_spar_header
from gannet.io.siemens_rda import read_rda_data
from gannet.io.siemens_twix import map_vbvd
from gannet.io.utils import discern_data_type
from gannet.pipeline.coregister import gannet_co_register
from gannet.pipeline.segment import gannet_segment
from gannet.utils.export import export_to_csv, export_to_html


def _populate_header_only_metadata(
    mrs_struct: MRSStruct,
    metab_file: str,
    ii: int,
) -> MRSStruct:
    """Populate geometry metadata from header without full spectral preprocessing.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct``.
    metab_file: str
        Path to metabolite MRS file.
    ii: int
        0-based scan index.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with header geometry parameters populated.
    """
    vendor = str(mrs_struct["p"].get("vendor", ""))
    p = mrs_struct["p"]
    if vendor == "Philips":
        stem = os.path.splitext(metab_file)[0]
        spar_p = f"{stem}.spar" if os.path.exists(f"{stem}.spar") else f"{stem}.SPAR"
        if not os.path.exists(spar_p):
            spar_p = metab_file
        spar = parse_spar_header(spar_p)
        p["voxdim"][ii] = [
            float(spar.get("lr_size", 30.0)),
            float(spar.get("ap_size", 30.0)),
            float(spar.get("cc_size", 30.0)),
        ]
        p["voxoff"][ii] = [
            float(spar.get("lr_off_center", 0.0)),
            float(spar.get("ap_off_center", 0.0)),
            float(spar.get("cc_off_center", 0.0)),
        ]
        p["voxang"][ii] = [
            float(spar.get("lr_angulation", 0.0)),
            float(spar.get("ap_angulation", 0.0)),
            float(spar.get("cc_angulation", 0.0)),
        ]
    elif vendor == "Siemens_rda":
        res = read_rda_data(metab_file)
        hdr = res["hdr"]
        if not isinstance(hdr, dict):
            hdr = {}
        p["voxdim"][ii] = [
            float(hdr.get("VOIPhaseFOV", 30.0)),
            float(hdr.get("VOIReadoutFOV", 30.0)),
            float(hdr.get("VOIThickness", 30.0)),
        ]
        p["voxoff"][ii] = [
            float(hdr.get("VOIPositionSag", 0.0)),
            float(hdr.get("VOIPositionCor", 0.0)),
            float(hdr.get("VOIPositionTra", 0.0)),
        ]
        p["NormSag"][ii] = float(hdr.get("VOINormalSag", 0.0))
        p["NormCor"][ii] = float(hdr.get("VOINormalCor", 0.0))
        p["NormTra"][ii] = float(hdr.get("VOINormalTra", 1.0))
    elif vendor == "Siemens_twix":
        meas = map_vbvd(metab_file, ignore_seg=True)[-1]
        yaps = meas.hdr.get("MeasYaps", {})
        voi = yaps.get("sSpecPara", {}).get("sVoI", {})
        pos = voi.get("sPosition", {})
        norm = voi.get("sNormal", {})
        p["voxdim"][ii] = [
            float(voi.get("dPhaseFOV", 30.0)),
            float(voi.get("dReadoutFOV", 30.0)),
            float(voi.get("dThickness", 30.0)),
        ]
        p["voxoff"][ii] = [
            float(pos.get("dSag", 0.0)),
            float(pos.get("dCor", 0.0)),
            float(pos.get("dTra", 0.0)),
        ]
        p["NormSag"][ii] = float(norm.get("dSag", 0.0))
        p["NormCor"][ii] = float(norm.get("dCor", 0.0))
        p["NormTra"][ii] = float(norm.get("dTra", 1.0))
        p["VoI_InPlaneRot"][ii] = float(voi.get("dInPlaneRot", 0.0))
    elif vendor == "GE":
        mrs_struct = ge_read(mrs_struct, metab_file)
    return mrs_struct


def co_reg(
    mrs_struct: MRSStruct,
    struc: list[str],
    out_dir: str | None = None,
) -> MRSStruct:
    """Standalone wrapper calling ``gannet_co_register`` (``CoReg.m``).

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct``.
    struc: list[str]
        List of structural T1 NIfTI or DICOM folder paths.
    out_dir: str | None
        Destination folder for HTML report.

    Returns
    --------
        MRSStruct:
            Co-registered ``MRSStruct``.
    """
    return gannet_co_register(mrs_struct, struc, save_html=True, out_dir=out_dir)


def seg(
    mrs_struct: MRSStruct,
    out_dir: str | None = None,
) -> MRSStruct:
    """Standalone wrapper calling ``gannet_segment`` (``Seg.m``).

    Parameters
    ------------
    mrs_struct: MRSStruct
        Co-registered ``MRSStruct``.
    out_dir: str | None
        Destination folder for CSV and HTML outputs.

    Returns
    --------
        MRSStruct:
            Segmented ``MRSStruct`` with ``fGM/fWM/fCSF`` and MRIQC QA metrics.
    """
    return gannet_segment(mrs_struct, save_csv=True, save_html=True, out_dir=out_dir)


def co_reg_stand_alone(
    metab: list[str] | str,
    struc: list[str] | str,
    config: GannetConfig | None = None,
    out_dir: str | None = None,
) -> MRSStruct:
    """Perform MRS voxel co-registration and T1 segmentation without spectral fitting.

    Ports ``CoRegStandAlone/CoRegStandAlone.m``.

    Parameters
    ------------
    metab: list[str] | str
        MRS file path(s) (only geometry headers are parsed).
    struc: list[str] | str
        Matching structural T1 NIfTI path(s) or GE DICOM directory path(s).
    config: GannetConfig | None
        Optional Gannet configuration override.
    out_dir: str | None
        Optional directory for HTML and CSV output files.

    Returns
    --------
        MRSStruct:
            ``MRSStruct`` containing voxel masks, tissue fractions, and QA metrics.
    """
    metab_list = [metab] if isinstance(metab, str) else list(metab)
    struc_list = [struc] * len(metab_list) if isinstance(struc, str) else list(struc)

    mrs_struct = gannet_pre_initialise(config=config)
    mrs_struct["metabfile"] = metab_list
    n_scans = len(metab_list)
    p = mrs_struct["p"]
    p["numScans"] = n_scans
    mrs_struct = discern_data_type(metab_list[0], mrs_struct)

    p["voxdim"] = np.zeros((n_scans, 3), dtype=float)
    p["voxoff"] = np.zeros((n_scans, 3), dtype=float)
    p["voxang"] = np.zeros((n_scans, 3), dtype=float)
    p["tlhc"] = np.zeros((n_scans, 3), dtype=float)
    p["trhc"] = np.zeros((n_scans, 3), dtype=float)
    p["brhc"] = np.zeros((n_scans, 3), dtype=float)
    p["TablePosition"] = np.zeros((n_scans, 3), dtype=float)
    p["NormSag"] = np.zeros(n_scans, dtype=float)
    p["NormCor"] = np.zeros(n_scans, dtype=float)
    p["NormTra"] = np.ones(n_scans, dtype=float)
    p["VoI_InPlaneRot"] = np.zeros(n_scans, dtype=float)

    for ii, m_file in enumerate(metab_list):
        mrs_struct["ii"] = ii
        mrs_struct = _populate_header_only_metadata(mrs_struct, m_file, ii)

    mrs_struct = co_reg(mrs_struct, struc_list, out_dir=out_dir)
    mrs_struct = seg(mrs_struct, out_dir=out_dir)
    dest_dir = out_dir or os.getcwd()
    export_to_csv(mrs_struct, caller="segment", out_dir=dest_dir)
    export_to_html(
        mrs_struct,
        out_path=os.path.join(dest_dir, "CoRegStandAlone_report.html"),
        title="Gannet CoRegStandAlone Report",
    )
    return mrs_struct
