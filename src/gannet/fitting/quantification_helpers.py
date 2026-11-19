"""Metabolite quantification helpers (CalcIU and phantom CalcConc)."""

from dataclasses import dataclass

import numpy as np

from gannet.config import MRSStruct


@dataclass(frozen=True)
class MetaboliteRelaxationParams:
    """Relaxation and editing constants for a metabolite.

    Attributes
    ----------
    editing_efficiency: float
        Editing efficiency factor.
    t1_metab: float
        Metabolite T1 relaxation time in seconds.
    t2_metab: float
        Metabolite T2 relaxation time in seconds.
    n_h_metab: int
        Number of protons contributing to the target resonance.
    mm_factor: float
        Macromolecule correction factor.
    c_wm: float
        Relative intrinsic concentration in pure WM.
    c_gm: float
        Relative intrinsic concentration in pure GM.
    """

    editing_efficiency: float
    t1_metab: float
    t2_metab: float
    n_h_metab: int
    mm_factor: float
    c_wm: float
    c_gm: float


def get_in_vivo_metabolite_params(metab: str) -> MetaboliteRelaxationParams:
    """Return literature relaxation and editing constants for in vivo MRS.

    Parameters
    ------------
    metab: str
        Metabolite name (``'GABA'``, ``'Glx'``, ``'GSH'``, ``'Lac'``,
        ``'EtOH'``, ``'Cr'``, ``'Cho'``, ``'NAA'``, or ``'Glu'``).

    Returns
    --------
        MetaboliteRelaxationParams: dataclass object
            Constants matching ``CalcIU.m`` and ``GannetQuantify.m``.
    """
    table: dict[str, MetaboliteRelaxationParams] = {
        "GABA": MetaboliteRelaxationParams(0.5, 1.31, 0.088, 2, 0.45, 1.0, 2.0),
        "GABAGlx": MetaboliteRelaxationParams(0.5, 1.31, 0.088, 2, 0.45, 1.0, 2.0),
        "Glx": MetaboliteRelaxationParams(0.4, 1.23, 0.18, 1, 1.0, 1.0, 2.0),
        "GSH": MetaboliteRelaxationParams(0.74, 0.40, 0.12, 2, 1.0, 1.0, 1.0),
        "Lac": MetaboliteRelaxationParams(0.94, 1.50, 0.24, 3, 1.0, 1.0, 1.0),
        "EtOH": MetaboliteRelaxationParams(0.5, 1.31, 0.088, 3, 1.0, 1.0, 1.0),
        "Cr": MetaboliteRelaxationParams(
            1.0,
            (1.46 + 1.24) / 2.0,
            (166.0 + 144.0 + 148.0) / 3.0 / 1e3,
            3,
            1.0,
            1.0,
            1.5,
        ),
        "Cho": MetaboliteRelaxationParams(
            1.0,
            (1.30 + 1.08) / 2.0,
            (218.0 + 222.0 + 274.0) / 3.0 / 1e3,
            9,
            1.0,
            1.0,
            1.0,
        ),
        "NAA": MetaboliteRelaxationParams(
            1.0,
            (1.47 + 1.35) / 2.0,
            (343.0 + 263.0 + 253.0) / 3.0 / 1e3,
            3,
            1.0,
            1.0,
            1.5,
        ),
        "Glu": MetaboliteRelaxationParams(0.4, 1.23, 0.18, 2, 1.0, 1.0, 2.0),
    }
    if metab not in table:
        raise ValueError(f"Unknown metabolite '{metab}' for quantification.")
    return table[metab]


def get_phantom_metabolite_params(metab: str) -> MetaboliteRelaxationParams:
    """Return room-temperature phantom relaxation and editing constants.

    Parameters
    ------------
    metab: str
        Metabolite name (``'GABA'``, ``'Glx'``, ``'GSH'``, ``'Lac'``, ``'EtOH'``).

    Returns
    --------
        MetaboliteRelaxationParams: dataclass object
            Constants matching ``CalcConc`` in ``GannetFitPhantom.m``.
    """
    table: dict[str, MetaboliteRelaxationParams] = {
        "GABA": MetaboliteRelaxationParams(0.5, 1.84, 248.0 / 1e3, 2, 1.0, 1.0, 1.0),
        "Glx": MetaboliteRelaxationParams(0.4, 1.18, 640.0 / 1e3, 1, 1.0, 1.0, 1.0),
        "GSH": MetaboliteRelaxationParams(0.74, 0.40, 0.12, 2, 1.0, 1.0, 1.0),
        "Lac": MetaboliteRelaxationParams(0.94, 1.50, 0.24, 3, 1.0, 1.0, 1.0),
        "EtOH": MetaboliteRelaxationParams(0.5, 1.84, 248.0 / 1e3, 3, 1.0, 1.0, 1.0),
    }
    if metab not in table:
        raise ValueError(f"Unknown phantom metabolite '{metab}' for quantification.")
    return table[metab]


def _ensure_scan_vector(container: MRSStruct, field_name: str, num_scans: int) -> None:
    """Ensure a 1D float array of length ``num_scans`` exists in ``container``.

    Parameters
    ------------
    container: MRSStruct
        Target output substruct.
    field_name: str
        Key name for the vector.
    num_scans: int
        Total number of scans in the batch.

    Returns
    --------
        None: None
            Modifies ``container`` in place if needed.
    """
    if field_name not in container or len(container[field_name]) < num_scans:
        existing = list(container[field_name]) if field_name in container else []
        arr = np.full(num_scans, np.nan, dtype=np.float64)
        if existing:
            arr[: len(existing)] = existing
        container[field_name] = arr


def calc_iu(mrs_struct: MRSStruct, vox: str, metab: str, ii: int) -> MRSStruct:
    """Convert metabolite and water areas to institutional units (mmol/L).

    Mirrors ``CalcIU.m``. Note ``ii`` is 0-based scan index in Python.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure containing fit areas and acquisition timings.
    vox: str
        Voxel name (e.g., ``'vox1'``).
    metab: str
        Target metabolite name.
    ii: int
        Zero-based scan index.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with ``ConcIU[ii]`` populated.
    """
    p = mrs_struct["p"]
    tr = float(p["TR"][ii]) / 1e3
    te = float(p["TE"][ii]) / 1e3
    tr_water = (
        float(p["TR_water"][ii]) / 1e3
        if "TR_water" in p and len(p["TR_water"]) > ii
        else tr
    )
    te_water = (
        float(p["TE_water"][ii]) / 1e3
        if "TE_water" in p and len(p["TE_water"]) > ii
        else te
    )

    pure_water_conc = 55.51 * 1e3
    water_visibility = 0.65
    t1_water = 1.100
    t2_water = 0.095
    n_h_water = 2.0

    mp = get_in_vivo_metabolite_params(metab)
    t1_factor = (1.0 - np.exp(-tr_water / t1_water)) / (1.0 - np.exp(-tr / mp.t1_metab))
    t2_factor = np.exp(-te_water / t2_water) / np.exp(-te / mp.t2_metab)

    out_vox = mrs_struct["out"][vox]
    metab_area = float(out_vox[metab]["Area"][ii])
    water_area = float(out_vox["water"]["Area"][ii])

    conc = (
        (metab_area / water_area)
        * pure_water_conc
        * water_visibility
        * t1_factor
        * t2_factor
        * (n_h_water / mp.n_h_metab)
        * mp.mm_factor
        / mp.editing_efficiency
    )
    if p.get("vendor") == "Siemens_rda":
        conc = conc / 2.0

    num_scans = int(p.get("numScans", ii + 1))
    _ensure_scan_vector(out_vox[metab], "ConcIU", num_scans)
    out_vox[metab]["ConcIU"][ii] = float(conc)
    return mrs_struct


def calc_conc_phantom(
    mrs_struct: MRSStruct, vox: str, metab: str, ii: int
) -> MRSStruct:
    """Quantify phantom metabolite concentration in absolute units (mM).

    Mirrors ``CalcConc`` in ``GannetFitPhantom.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure containing fit areas and acquisition timings.
    vox: str
        Voxel name (e.g., ``'vox1'``).
    metab: str
        Target metabolite name.
    ii: int
        Zero-based scan index.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with ``ConcIU[ii]`` populated.
    """
    p = mrs_struct["p"]
    tr = float(p["TR"][ii]) / 1e3
    te = float(p["TE"][ii]) / 1e3
    tr_water = (
        float(p["TR_water"][ii]) / 1e3
        if "TR_water" in p and len(p["TR_water"]) > ii
        else tr
    )
    te_water = (
        float(p["TE_water"][ii]) / 1e3
        if "TE_water" in p and len(p["TE_water"]) > ii
        else te
    )

    pure_water_conc = 55.51 * 1e3
    t1_water = 2928.0 / 1e3
    t2_water = 0.503
    n_h_water = 2.0

    mp = get_phantom_metabolite_params(metab)
    t1_factor = (1.0 - np.exp(-tr_water / t1_water)) / (1.0 - np.exp(-tr / mp.t1_metab))
    t2_factor = np.exp(-te_water / t2_water) / np.exp(-te / mp.t2_metab)

    out_vox = mrs_struct["out"][vox]
    metab_area = float(out_vox[metab]["Area"][ii])
    water_area = float(out_vox["water"]["Area"][ii])

    conc = (
        (metab_area / water_area)
        * pure_water_conc
        * t1_factor
        * t2_factor
        * (n_h_water / mp.n_h_metab)
        / mp.editing_efficiency
    )
    if str(p.get("vendor", "")).lower() == "siemens_rda":
        conc = conc / 2.0

    num_scans = int(p.get("numScans", ii + 1))
    _ensure_scan_vector(out_vox[metab], "ConcIU", num_scans)
    out_vox[metab]["ConcIU"][ii] = float(conc)
    return mrs_struct
