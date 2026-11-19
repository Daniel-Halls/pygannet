"""Siemens RDA file reader."""

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct


def read_rda_data(fname: str) -> dict[str, object]:
    """Read header metadata and complex FID data from a Siemens ``.rda`` file.

    Mirrors ``read_rda_data`` in ``SiemensRead.m``.

    Parameters
    ------------
    fname: str
        Path to the Siemens ``.rda`` file.

    Returns
    --------
        dict[str, object]: dict object
            Dictionary with keys ``'data'`` (complex array) and ``'hdr'``
            (parsed header dictionary).
    """
    hdr: dict[str, object] = {}
    str_vars = {
        "PatientID",
        "PatientName",
        "StudyDescription",
        "PatientBirthDate",
        "StudyDate",
        "StudyTime",
        "PatientAge",
        "SeriesDate",
        "SeriesTime",
        "SeriesDescription",
        "ProtocolName",
        "PatientPosition",
        "ModelName",
        "StationName",
        "InstitutionName",
        "DeviceSerialNumber",
        "InstanceDate",
        "InstanceTime",
        "InstanceComments",
        "SequenceName",
        "SequenceDescription",
        "Nucleus",
        "TransmitCoil",
    }
    num_vars = {
        "SeriesNumber",
        "InstanceNumber",
        "AcquisitionNumber",
        "NumOfPhaseEncodingSteps",
        "NumberOfRows",
        "NumberOfColumns",
        "VectorSize",
        "PatientWeight",
        "TR",
        "TE",
        "TM",
        "DwellTime",
        "NumberOfAverages",
        "MRFrequency",
        "MagneticFieldStrength",
        "FlipAngle",
        "SliceThickness",
        "FoVHeight",
        "FoVWidth",
        "PercentOfRectFoV",
        "PixelSpacingRow",
        "PixelSpacingCol",
        "VOIRotationInPlane",
    }

    csi_size = [1, 1, 1]
    pos_vec = [0.0, 0.0, 0.0]
    row_vec = [0.0, 0.0, 0.0]
    col_vec = [0.0, 0.0, 0.0]
    tab_vec = [0.0, 0.0, 0.0]

    with open(fname, "rb") as fid:
        while True:
            raw_line = fid.readline()
            if not raw_line:
                break
            line = raw_line.decode("latin-1").strip()
            if ">>> End of header <<<" in line:
                break
            if ">>> Begin of header <<<" in line or ":" not in line:
                continue
            var, val = line.split(":", 1)
            var = var.strip()
            val = val.strip()

            if var in str_vars:
                hdr[var] = val
            elif var == "PatientSex":
                sex_map = {"0": "Unknown", "1": "Male", "2": "Female"}
                hdr["sex"] = sex_map.get(val, "Unknown")
            elif var in num_vars:
                hdr[var] = float(val)
            elif var == "SoftwareVersion[0]":
                hdr["software_version"] = val
            elif var.startswith("CSIMatrixSize["):
                idx = int(var.split("[")[1].split("]")[0])
                csi_size[idx] = int(float(val))
            elif var.startswith("PositionVector["):
                idx = int(var.split("[")[1].split("]")[0])
                pos_vec[idx] = float(val)
            elif var.startswith("RowVector["):
                idx = int(var.split("[")[1].split("]")[0])
                row_vec[idx] = float(val)
            elif var.startswith("ColumnVector["):
                idx = int(var.split("[")[1].split("]")[0])
                col_vec[idx] = float(val)
            elif var == "TablePosSag":
                tab_vec[0] = float(val)
            elif var == "TablePosCor":
                tab_vec[1] = float(val)
            elif var == "TablePosTra":
                tab_vec[2] = float(val)

        hdr["CSIMatrix_Size"] = csi_size
        hdr["PositionVector"] = pos_vec
        hdr["RowVector"] = row_vec
        hdr["ColumnVector"] = col_vec
        hdr["TablePosition"] = tab_vec

        vec_size = int(hdr["VectorSize"])  # type: ignore[arg-type]
        n_elem = csi_size[0] * csi_size[1] * csi_size[2] * vec_size * 2
        raw_vals = np.fromfile(fid, dtype="<f8", count=n_elem)

    shaped = raw_vals.reshape(
        (2, vec_size, csi_size[0], csi_size[1], csi_size[2]), order="F"
    )
    data: NDArray[np.complex128] = shaped[0, :, :, :, :] + 1j * shaped[1, :, :, :, :]
    result: dict[str, object] = {"data": np.squeeze(data), "hdr": hdr}
    return result


def _set_p_val(
    p: MRSStruct, key: str, idx: int, val: float | list[float], cols: int = 0
) -> None:
    """Assign value into scan-indexed array inside ``MRS_struct.p``."""
    if cols == 0:
        if key not in p or len(p[key]) <= idx:
            arr = np.zeros(idx + 1, dtype=np.float64)
            if key in p:
                arr[: len(p[key])] = p[key]
            p[key] = arr
        p[key][idx] = float(val)  # type: ignore[arg-type]
    else:
        if key not in p or p[key].shape[0] <= idx:
            mat = np.zeros((idx + 1, cols), dtype=np.float64)
            if key in p:
                mat[: p[key].shape[0], :] = p[key]
            p[key] = mat
        p[key][idx, :] = np.asarray(val, dtype=np.float64)


def siemens_read(
    mrs_struct: MRSStruct,
    off_filename: str,
    on_filename: str | None = None,
    water_filename: str | None = None,
) -> MRSStruct:
    """Load Siemens ``.rda`` spectroscopy data (single file or paired ON/OFF).

    Mirrors ``SiemensRead.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure.
    off_filename: str
        Path to the OFF (or combined) ``.rda`` file.
    on_filename: str | None
        Optional path to separate ON subspectrum ``.rda`` file.
    water_filename: str | None
        Optional path to the water reference ``.rda`` file.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with header parameters and ``fids``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    off_res = read_rda_data(off_filename)
    off_raw = np.asarray(off_res["data"], dtype=np.complex128)
    hdr: dict[str, object] = off_res["hdr"]  # type: ignore[assignment]

    fids = mrs_struct.ensure_substruct("fids")
    n_pts_hdr = int(float(hdr.get("VectorSize", off_raw.shape[0])))  # type: ignore[arg-type]
    if on_filename is not None:
        on_res = read_rda_data(on_filename)
        fids["data"] = np.column_stack(
            [
                np.asarray(on_res["data"], dtype=np.complex128).ravel(),
                off_raw.ravel(),
            ]
        )
    elif off_raw.ndim == 1:
        fids["data"] = off_raw.reshape((n_pts_hdr, -1), order="F")
    else:
        fids["data"] = off_raw.reshape((n_pts_hdr, -1))

    in_plane_rot = float(hdr.get("VOIRotationInPlane", 0.0))  # type: ignore[arg-type]
    _set_p_val(p, "VoI_InPlaneRot", ii, in_plane_rot)
    _set_p_val(p, "LarmorFreq", ii, float(hdr["MRFrequency"]))  # type: ignore[arg-type]
    _set_p_val(p, "npoints", ii, float(hdr["VectorSize"]))  # type: ignore[arg-type]
    _set_p_val(p, "sw", ii, 1.0 / float(hdr["DwellTime"]) * 1e6)  # type: ignore[arg-type]
    _set_p_val(p, "TR", ii, float(hdr["TR"]))  # type: ignore[arg-type]
    _set_p_val(p, "TE", ii, float(hdr["TE"]))  # type: ignore[arg-type]
    _set_p_val(p, "Navg", ii, float(hdr["NumberOfAverages"]))  # type: ignore[arg-type]

    if "FoVHeight" in hdr and "FoVWidth" in hdr:
        d1 = float(hdr["FoVHeight"])  # type: ignore[arg-type]
        d2 = float(hdr["FoVWidth"])  # type: ignore[arg-type]
    elif "PixelSpacingRow" in hdr and "PixelSpacingCol" in hdr:
        d1 = float(hdr["PixelSpacingRow"])  # type: ignore[arg-type]
        d2 = float(hdr["PixelSpacingCol"])  # type: ignore[arg-type]
    else:
        d1 = float(hdr.get("VOIPhaseFOV", 30.0))  # type: ignore[arg-type]
        d2 = float(hdr.get("VOIReadoutFOV", 30.0))  # type: ignore[arg-type]
    d3 = float(hdr.get("SliceThickness", hdr.get("VOIThickness", 30.0)))  # type: ignore[arg-type]
    _set_p_val(p, "voxdim", ii, [d1, d2, d3], cols=3)
    _set_p_val(p, "voxoff", ii, hdr["PositionVector"], cols=3)  # type: ignore[arg-type]

    if water_filename is not None:
        w_res = read_rda_data(water_filename)
        w_hdr: dict[str, object] = w_res["hdr"]  # type: ignore[assignment]
        fids["data_water"] = np.asarray(w_res["data"], dtype=np.complex128).ravel()
        if "TR" in w_hdr:
            _set_p_val(p, "TR_water", ii, float(w_hdr["TR"]))  # type: ignore[arg-type]
        if "TE" in w_hdr:
            _set_p_val(p, "TE_water", ii, float(w_hdr["TE"]))  # type: ignore[arg-type]

    return mrs_struct
