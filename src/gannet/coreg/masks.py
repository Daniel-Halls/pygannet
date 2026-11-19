"""Vendor-specific MRS voxel mask generators for T1-weighted structural images."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import nibabel as nib
import numpy as np
import pydicom
from numpy.typing import NDArray
from pydicom.errors import InvalidDicomError

from gannet.config import MRSStruct
from gannet.coreg.geometry import (
    build_mask_output_path,
    get_bbox_fv,
    make_three_plane_overlay,
    rasterize_corners_to_mask,
    save_voxel_mask_nifti,
)
from gannet.io.philips import parse_spar_header
from gannet.io.siemens_rda import read_rda_data


def _rot_x(deg: float) -> NDArray[np.float64]:
    """Return 3D rotation matrix around X axis for angle in degrees.

    Parameters
    ------------
    deg: float
        Rotation angle in degrees.

    Returns
    --------
        NDArray[np.float64]:
            3x3 rotation matrix.
    """
    rad = np.deg2rad(deg)
    c, s = (np.cos(rad), np.sin(rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> NDArray[np.float64]:
    """Return 3D rotation matrix around Y axis for angle in degrees.

    Parameters
    ------------
    deg: float
        Rotation angle in degrees.

    Returns
    --------
        NDArray[np.float64]:
            3x3 rotation matrix.
    """
    rad = np.deg2rad(deg)
    c, s = (np.cos(rad), np.sin(rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(deg: float) -> NDArray[np.float64]:
    """Return 3D rotation matrix around Z axis for angle in degrees.

    Parameters
    ------------
    deg: float
        Rotation angle in degrees.

    Returns
    --------
        NDArray[np.float64]:
            3x3 rotation matrix.
    """
    rad = np.deg2rad(deg)
    c, s = (np.cos(rad), np.sin(rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _siemens_rotation_matrix(
    norm_sag: float,
    norm_cor: float,
    norm_tra: float,
    in_plane_rot: float,
) -> NDArray[np.float64]:
    """Construct Siemens VoI orientation rotation matrix in RAS world coordinates.

    Parameters
    ------------
    norm_sag: float
        Sagittal (L->R) component of slice normal in LPS convention.
    norm_cor: float
        Coronal (A->P) component of slice normal in LPS convention.
    norm_tra: float
        Transverse (F->H) component of slice normal in LPS convention.
    in_plane_rot: float
        In-plane VoI rotation angle in radians.

    Returns
    --------
        NDArray[np.float64]:
            3x3 orthonormal rotation matrix ``[row_vec, col_vec, norm_vec]`` in RAS.
    """
    norm_lps = np.array([norm_sag, norm_cor, norm_tra], dtype=np.float64)
    n_len = np.linalg.norm(norm_lps)
    norm_lps = norm_lps / n_len if n_len > 0 else np.array([0.0, 0.0, 1.0])

    # Convert LPS normal to RAS (negate X=Sag and Y=Cor)
    z_ras = np.array([-norm_lps[0], -norm_lps[1], norm_lps[2]], dtype=np.float64)

    # Siemens DICOM/TWIX standard in-plane phase direction initialization
    if abs(norm_lps[0]) >= abs(norm_lps[1]) and abs(norm_lps[0]) >= abs(norm_lps[2]):
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif abs(norm_lps[1]) >= abs(norm_lps[2]):
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_ras = np.cross(ref, z_ras)
    if np.linalg.norm(x_ras) < 1e-8:
        x_ras = np.cross(np.array([1.0, 0.0, 0.0]), z_ras)
    x_ras /= np.linalg.norm(x_ras)
    y_ras = np.cross(z_ras, x_ras)
    y_ras /= np.linalg.norm(y_ras)

    c = np.cos(in_plane_rot)
    s = np.sin(in_plane_rot)
    x_rot = c * x_ras + s * y_ras
    y_rot = -s * x_ras + c * y_ras
    return np.column_stack([x_rot, y_rot, z_ras])


def _finalize_mask_outputs(
    mrs_struct: MRSStruct,
    ii: int,
    vox_idx: int,
    corners_world: NDArray[np.float64],
    vox_center_mm: NDArray[np.float64],
    t1_img: nib.Nifti1Image,
    mrs_file: str,
    t1_file: str,
) -> MRSStruct:
    """Rasterize voxel mask, save NIfTI file, and populate ``MRSStruct.mask``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` object.
    ii: int
        0-based scan index.
    vox_idx: int
        0-based voxel index within scan.
    corners_world: NDArray[np.float64]
        8 world RAS corners ``(8, 3)``.
    vox_center_mm: NDArray[np.float64]
        3-element voxel center in RAS world mm.
    t1_img: nib.Nifti1Image
        Loaded T1 structural NIfTI image.
    mrs_file: str
        Source MRS file path.
    t1_file: str
        Source T1 NIfTI path.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with ``mask`` fields populated.
    """
    mask_arr, _corners_vox = rasterize_corners_to_mask(corners_world, t1_img)
    vox_names = mrs_struct["p"].get("Vox", ["vox1"])
    vox_name = vox_names[vox_idx] if vox_idx < len(vox_names) else f"vox{vox_idx + 1}"
    out_path = build_mask_output_path(
        mrs_file, t1_file, vox_name=vox_name, num_voxels=int(mrs_struct["p"]["numVox"])
    )
    save_voxel_mask_nifti(mask_arr, t1_img, out_path)
    overlay = make_three_plane_overlay(t1_file, out_path, vox_center_mm=vox_center_mm)

    mrs_struct.ensure_substruct("mask", vox_name)
    vox_mask = mrs_struct["mask"][vox_name]
    vox_mask.setdefault("outfile", [])
    vox_mask.setdefault("img", [])
    vox_mask.setdefault("T1image", [])
    vox_mask.setdefault("center_mm", [])
    while len(vox_mask["outfile"]) <= ii:
        vox_mask["outfile"].append("")
    while len(vox_mask["img"]) <= ii:
        vox_mask["img"].append(None)
    while len(vox_mask["T1image"]) <= ii:
        vox_mask["T1image"].append("")
    while len(vox_mask["center_mm"]) <= ii:
        vox_mask["center_mm"].append([0.0, 0.0, 0.0])

    vox_mask["outfile"][ii] = out_path
    vox_mask["img"][ii] = overlay
    vox_mask["T1image"][ii] = t1_file
    vox_mask["center_mm"][ii] = vox_center_mm.tolist()
    return mrs_struct


def gannet_mask_philips(
    spar_file: str,
    nii_file: str,
    mrs_struct: MRSStruct,
    ii: int,
    vox_idx: int = 0,
) -> MRSStruct:
    """Create T1-aligned binary MRS voxel mask from a Philips ``.spar`` header.

    Parameters
    ------------
    spar_file: str
        Path to Philips ``.spar`` (or ``.sdat``) file.
    nii_file: str
        Path to T1-weighted structural NIfTI file.
    mrs_struct: MRSStruct
        Active ``MRSStruct``.
    ii: int
        0-based scan index.
    vox_idx: int
        0-based voxel index.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with co-registered voxel mask.
    """
    prefix = re.sub(r"\.(sdat|spar)$", "", spar_file, flags=re.IGNORECASE)
    cand_spar = (
        f"{prefix}.spar" if os.path.exists(f"{prefix}.spar") else f"{prefix}.SPAR"
    )
    if not os.path.exists(cand_spar):
        cand_spar = spar_file
    spar = parse_spar_header(cand_spar)

    # Philips LPS-based off_center & angulation to RAS mm (GannetMask_Philips.m)
    lr_off = float(spar.get("lr_off_center", 0.0))
    ap_off = float(spar.get("ap_off_center", 0.0))
    cc_off = float(spar.get("cc_off_center", 0.0))
    lr_sz = float(spar.get("lr_size", 30.0))
    ap_sz = float(spar.get("ap_size", 30.0))
    cc_sz = float(spar.get("cc_size", 30.0))
    lr_ang = float(spar.get("lr_angulation", 0.0))
    ap_ang = float(spar.get("ap_angulation", 0.0))
    cc_ang = float(spar.get("cc_angulation", 0.0))

    vox_ctr = np.array([-lr_off, -ap_off, cc_off], dtype=np.float64)
    vox_dim = np.array([lr_sz, ap_sz, cc_sz], dtype=np.float64)
    rot_mat = _rot_y(ap_ang) @ _rot_x(-lr_ang) @ _rot_z(cc_ang)
    corners_world, _faces = get_bbox_fv(vox_ctr, rot_mat, vox_dim)
    t1_img = nib.load(nii_file)
    return _finalize_mask_outputs(
        mrs_struct, ii, vox_idx, corners_world, vox_ctr, t1_img, spar_file, nii_file
    )


def gannet_mask_siemens_twix(
    twix_file: str,
    nii_file: str,
    mrs_struct: MRSStruct,
    ii: int,
    vox_idx: int = 0,
) -> MRSStruct:
    """Create T1-aligned binary MRS voxel mask from Siemens TWIX metadata.

    Parameters
    ------------
    twix_file: str
        Path to Siemens TWIX ``.dat`` file.
    nii_file: str
        Path to T1-weighted structural NIfTI file.
    mrs_struct: MRSStruct
        Active ``MRSStruct`` containing TWIX geometry parameters in ``p``.
    ii: int
        0-based scan index.
    vox_idx: int
        0-based voxel index.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with co-registered voxel mask.
    """
    p = mrs_struct["p"]
    vox_dim = np.asarray(p["voxdim"][ii], dtype=np.float64)
    vox_off_lps = np.asarray(p["voxoff"][ii], dtype=np.float64)
    table_pos = np.asarray(p["TablePosition"][ii], dtype=np.float64)
    pos_lps = vox_off_lps + table_pos
    vox_ctr = np.array([-pos_lps[0], -pos_lps[1], pos_lps[2]], dtype=np.float64)

    rot_mat = _siemens_rotation_matrix(
        float(p["NormSag"][ii]),
        float(p["NormCor"][ii]),
        float(p["NormTra"][ii]),
        float(p["VoI_InPlaneRot"][ii]),
    )
    corners_world, _faces = get_bbox_fv(vox_ctr, rot_mat, vox_dim)
    t1_img = nib.load(nii_file)
    return _finalize_mask_outputs(
        mrs_struct, ii, vox_idx, corners_world, vox_ctr, t1_img, twix_file, nii_file
    )


def gannet_mask_siemens_rda(
    rda_file: str,
    nii_file: str,
    mrs_struct: MRSStruct,
    ii: int,
    vox_idx: int = 0,
) -> MRSStruct:
    """Create T1-aligned binary MRS voxel mask from a Siemens ``.rda`` header.

    Parameters
    ------------
    rda_file: str
        Path to Siemens ``.rda`` file.
    nii_file: str
        Path to T1-weighted structural NIfTI file.
    mrs_struct: MRSStruct
        Active ``MRSStruct``.
    ii: int
        0-based scan index.
    vox_idx: int
        0-based voxel index.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with co-registered voxel mask.
    """
    rda_res = read_rda_data(rda_file)
    hdr = rda_res["hdr"]
    if not isinstance(hdr, dict):
        hdr = {}
    pos_lps = np.array(
        [
            float(hdr.get("VOIPositionSag", 0.0)),
            float(hdr.get("VOIPositionCor", 0.0)),
            float(hdr.get("VOIPositionTra", 0.0)),
        ],
        dtype=np.float64,
    )
    vox_ctr = np.array([-pos_lps[0], -pos_lps[1], pos_lps[2]], dtype=np.float64)
    vox_dim = np.array(
        [
            float(hdr.get("VOIPhaseFOV", 30.0)),
            float(hdr.get("VOIReadoutFOV", 30.0)),
            float(hdr.get("VOIThickness", 30.0)),
        ],
        dtype=np.float64,
    )
    norm_sag = float(hdr.get("VOINormalSag", 0.0))
    norm_cor = float(hdr.get("VOINormalCor", 0.0))
    norm_tra = float(hdr.get("VOINormalTra", 1.0))

    if "RowVector" in hdr and "ColumnVector" in hdr:
        row_lps = np.asarray(hdr["RowVector"], dtype=np.float64)[:3]
        col_lps = np.asarray(hdr["ColumnVector"], dtype=np.float64)[:3]
        nrm_lps = np.array([norm_sag, norm_cor, norm_tra], dtype=np.float64)
        lps_to_ras = np.diag([-1.0, -1.0, 1.0])
        rot_mat = lps_to_ras @ np.column_stack([row_lps, col_lps, nrm_lps])
    else:
        rot_mat = _siemens_rotation_matrix(norm_sag, norm_cor, norm_tra, 0.0)

    corners_world, _faces = get_bbox_fv(vox_ctr, rot_mat, vox_dim)
    t1_img = nib.load(nii_file)
    return _finalize_mask_outputs(
        mrs_struct, ii, vox_idx, corners_world, vox_ctr, t1_img, rda_file, nii_file
    )


def gannet_mask_nifti(
    mrs_nii_file: str,
    t1_nii_file: str,
    mrs_struct: MRSStruct,
    ii: int,
    vox_idx: int = 0,
) -> MRSStruct:
    """Create T1-aligned binary MRS voxel mask directly from a NIfTI-MRS affine.

    Parameters
    ------------
    mrs_nii_file: str
        Path to the NIfTI-MRS file whose ``qform``/``sform`` encodes SVS geometry.
    t1_nii_file: str
        Path to T1-weighted structural NIfTI file.
    mrs_struct: MRSStruct
        Active ``MRSStruct``.
    ii: int
        0-based scan index.
    vox_idx: int
        0-based voxel index.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with co-registered voxel mask.
    """
    mrs_img = nib.load(mrs_nii_file)
    mrs_aff = np.asarray(mrs_img.affine, dtype=np.float64)
    # Unit cube centered at voxel (0, 0, 0) in NIfTI-MRS spatial grid
    unit_corners = np.array(
        [
            [-0.5, -0.5, -0.5, 1.0],
            [+0.5, -0.5, -0.5, 1.0],
            [+0.5, +0.5, -0.5, 1.0],
            [-0.5, +0.5, -0.5, 1.0],
            [-0.5, -0.5, +0.5, 1.0],
            [+0.5, -0.5, +0.5, 1.0],
            [+0.5, +0.5, +0.5, 1.0],
            [-0.5, +0.5, +0.5, 1.0],
        ],
        dtype=np.float64,
    )
    corners_world = (mrs_aff @ unit_corners.T).T[:, :3]
    vox_ctr = (mrs_aff @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    t1_img = nib.load(t1_nii_file)
    return _finalize_mask_outputs(
        mrs_struct,
        ii,
        vox_idx,
        corners_world,
        vox_ctr,
        t1_img,
        mrs_nii_file,
        t1_nii_file,
    )


def _ge_corners_from_header(
    mrs_struct: MRSStruct, ii: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute 8 world RAS corners and center from GE P-file plane coordinates.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` populated by ``ge_read``.
    ii: int
        0-based scan index.

    Returns
    --------
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            ``(corners_world, vox_center)`` in RAS world mm.
    """
    p = mrs_struct["p"]
    tlhc = np.asarray(p["tlhc"][ii], dtype=np.float64)
    trhc = np.asarray(p["trhc"][ii], dtype=np.float64)
    brhc = np.asarray(p["brhc"][ii], dtype=np.float64)
    vox_dim = np.asarray(p["voxdim"][ii], dtype=np.float64)
    vox_ctr = np.asarray(p["voxoff"][ii], dtype=np.float64)

    v1 = trhc - tlhc
    v2 = brhc - trhc
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 > 0.0 and n2 > 0.0:
        e1 = v1 / n1
        e2 = v2 / n2
        e3 = np.cross(e1, e2)
        n3 = float(np.linalg.norm(e3))
        if n3 > 0.0:
            e3 = e3 / n3
            rot_mat = np.column_stack([e1, e2, e3])
            corners_world, _ = get_bbox_fv(vox_ctr, rot_mat, vox_dim)
            return (corners_world, vox_ctr)

    corners_world, _ = get_bbox_fv(vox_ctr, np.eye(3, dtype=np.float64), vox_dim)
    return (corners_world, vox_ctr)


def gannet_mask_ge_nii(
    pfile: str,
    nii_file: str,
    mrs_struct: MRSStruct,
    ii: int,
    vox_idx: int = 0,
) -> MRSStruct:
    """Create T1-aligned binary MRS voxel mask for GE P-files with a NIfTI T1.

    Parameters
    ------------
    pfile: str
        Path to GE ``.7`` P-file.
    nii_file: str
        Path to structural T1 NIfTI file.
    mrs_struct: MRSStruct
        Active ``MRSStruct``.
    ii: int
        0-based scan index.
    vox_idx: int
        0-based voxel index.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with co-registered voxel mask.
    """
    corners_world, vox_ctr = _ge_corners_from_header(mrs_struct, ii)
    t1_img = nib.load(nii_file)
    return _finalize_mask_outputs(
        mrs_struct, ii, vox_idx, corners_world, vox_ctr, t1_img, pfile, nii_file
    )


def _dicom_slice_sort_key(ds: pydicom.Dataset) -> float:
    """Compute physical slice location along imaging normal for sorting DICOM stack.

    Parameters
    ------------
    ds: pydicom.Dataset
        Loaded DICOM dataset slice.

    Returns
    --------
        float:
            Scalar spatial coordinate or instance number for slice ordering.
    """
    if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
        iop = [float(x) for x in ds.ImageOrientationPatient]
        normal = np.cross(iop[:3], iop[3:6])
        ipp = np.asarray([float(x) for x in ds.ImagePositionPatient])
        return float(np.dot(ipp, normal))
    return float(getattr(ds, "InstanceNumber", 0))


def convert_dicom_dir_to_nifti(dcm_dir: str, out_nii_path: str | None = None) -> str:
    """Convert a 2D T1 DICOM folder stack into a 3D RAS-oriented NIfTI file.

    Uses ``dcm2niix`` when installed on PATH and falls back to pure-Python
    ``pydicom`` + ``nibabel`` slice stacking if ``dcm2niix`` is unavailable.

    Parameters
    ------------
    dcm_dir: str
        Path to directory containing T1 anatomical DICOM slices (or one DICOM file).
    out_nii_path: str | None
        Optional output ``.nii`` path; placed beside ``dcm_dir`` if omitted.

    Returns
    --------
        str:
            Path to the written NIfTI file.
    """
    target_dir = dcm_dir if os.path.isdir(dcm_dir) else os.path.dirname(dcm_dir)
    dcm2niix_bin = shutil.which("dcm2niix")
    if dcm2niix_bin is not None:
        dest_path = out_nii_path or os.path.join(
            target_dir, "T1_dicom_converted.nii.gz"
        )
        out_folder = os.path.dirname(os.path.abspath(dest_path))
        stem = re.sub(r"\.nii(\.gz)?$", "", os.path.basename(dest_path))
        gz_flag = "y" if dest_path.lower().endswith(".gz") else "n"
        proc = subprocess.run(
            [dcm2niix_bin, "-z", gz_flag, "-f", stem, "-o", out_folder, target_dir],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and os.path.exists(dest_path):
            return dest_path

    slices: list[pydicom.Dataset] = []
    for entry in sorted(os.listdir(target_dir)):
        fpath = os.path.join(target_dir, entry)
        if not os.path.isfile(fpath) or entry.startswith("."):
            continue
        try:
            ds = pydicom.dcmread(fpath, force=True)
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except (InvalidDicomError, OSError):
            continue
    if not slices:
        raise ValueError(f"No readable DICOM image slices found in {target_dir}")

    slices.sort(key=_dicom_slice_sort_key)
    first = slices[0]
    rows = int(first.Rows)
    cols = int(first.Columns)
    vol = np.zeros((cols, rows, len(slices)), dtype=np.float32)
    for k, ds in enumerate(slices):
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        vol[:, :, k] = (arr * slope + intercept).T

    ps = [float(x) for x in getattr(first, "PixelSpacing", [1.0, 1.0])]
    iop = [
        float(x) for x in getattr(first, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    ]
    row_cos = np.asarray(iop[:3], dtype=np.float64)
    col_cos = np.asarray(iop[3:6], dtype=np.float64)
    ipp0 = np.asarray(
        [float(x) for x in getattr(first, "ImagePositionPatient", [0, 0, 0])]
    )
    if len(slices) > 1 and hasattr(slices[-1], "ImagePositionPatient"):
        ipp_end = np.asarray([float(x) for x in slices[-1].ImagePositionPatient])
        slice_vec = (ipp_end - ipp0) / (len(slices) - 1)
    else:
        thk = float(getattr(first, "SliceThickness", 1.0))
        slice_vec = np.cross(row_cos, col_cos) * thk

    lps_aff = np.eye(4, dtype=np.float64)
    lps_aff[:3, 0] = row_cos * ps[1]
    lps_aff[:3, 1] = col_cos * ps[0]
    lps_aff[:3, 2] = slice_vec
    lps_aff[:3, 3] = ipp0
    # Convert DICOM LPS -> NIfTI RAS
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    ras_aff = lps_to_ras @ lps_aff

    if out_nii_path is None:
        out_nii_path = os.path.join(target_dir, "T1_dicom_converted.nii")
    img = nib.Nifti1Image(vol, ras_aff)
    nib.save(img, out_nii_path)
    return out_nii_path


def gannet_mask_ge(
    pfile: str,
    t1_dcm_dir: str,
    mrs_struct: MRSStruct,
    ii: int,
    vox_idx: int = 0,
) -> MRSStruct:
    """Create T1-aligned binary MRS voxel mask for GE P-files with T1 DICOM dir.

    Parameters
    ------------
    pfile: str
        Path to GE ``.7`` P-file.
    t1_dcm_dir: str
        Path to T1 DICOM directory (or single slice path inside it).
    mrs_struct: MRSStruct
        Active ``MRSStruct``.
    ii: int
        0-based scan index.
    vox_idx: int
        0-based voxel index.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with co-registered voxel mask.
    """
    nii_file = convert_dicom_dir_to_nifti(t1_dcm_dir)
    return gannet_mask_ge_nii(pfile, nii_file, mrs_struct, ii, vox_idx=vox_idx)
