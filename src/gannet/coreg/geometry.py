"""3D voxel geometry, NIfTI mask generation, and slice resampling utilities."""

from __future__ import annotations

import os
from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay, QhullError


def image_center(nii_path: str, write_back: bool = False) -> NDArray[np.float64]:
    """Compute (and optionally reset) a centered origin affine for a NIfTI volume.

    Reproduces ``image_center.m`` by placing the coordinate origin at the
    geometric center of the 3D volume while preserving voxel zooms.

    Parameters
    ------------
    nii_path: str
        Path to the NIfTI (``.nii`` / ``.nii.gz``) file.
    write_back: bool
        If ``True``, updates the image sform/qform on disk.

    Returns
    --------
        NDArray[np.float64]:
            Updated 4x4 affine matrix.
    """
    img = nib.load(nii_path)
    shape = np.asarray(img.shape[:3], dtype=np.float64)
    affine = np.asarray(img.affine, dtype=np.float64).copy()
    center_vox = (shape - 1.0) / 2.0
    affine[:3, 3] = -(affine[:3, :3] @ center_vox)
    if write_back:
        out_img = nib.Nifti1Image(img.get_fdata(), affine, img.header)
        out_img.set_qform(affine)
        out_img.set_sform(affine)
        nib.save(out_img, nii_path)
    return affine


def get_bbox_fv(
    voxel_ctr: NDArray[np.float64],
    rot_mat: NDArray[np.float64],
    vox_dim: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Compute the 8 world-space vertices and 6 quad faces of an MRS voxel.

    Parameters
    ------------
    voxel_ctr: NDArray[np.float64]
        3-element voxel center coordinates in world space (mm).
    rot_mat: NDArray[np.float64]
        3x3 rotation matrix mapping local voxel axes into world RAS mm.
    vox_dim: NDArray[np.float64]
        3-element voxel edge lengths in mm.

    Returns
    --------
        tuple[NDArray[np.float64], NDArray[np.int32]]:
            Tuple ``(vertices, faces)`` with ``vertices`` shaped ``(8, 3)``
            and ``faces`` shaped ``(6, 4)``.
    """
    half = np.asarray(vox_dim, dtype=np.float64).reshape(3) / 2.0
    local_corners = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [+half[0], -half[1], -half[2]],
            [+half[0], +half[1], -half[2]],
            [-half[0], +half[1], -half[2]],
            [-half[0], -half[1], +half[2]],
            [+half[0], -half[1], +half[2]],
            [+half[0], +half[1], +half[2]],
            [-half[0], +half[1], +half[2]],
        ],
        dtype=np.float64,
    )
    vertices = (rot_mat @ local_corners.T).T + np.asarray(voxel_ctr, dtype=np.float64)
    faces = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 3, 7, 4],
            [1, 2, 6, 5],
        ],
        dtype=np.int32,
    )
    return (vertices, faces)


def rasterize_corners_to_mask(
    corners_world: NDArray[np.float64],
    t1_img: nib.Nifti1Image,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Rasterize 8 world-space MRS voxel corners onto a T1 NIfTI grid.

    Reproduces Gannet's ``delaunayn`` / ``tsearchn`` volumetric inclusion check
    within the axis-aligned bounding box of the transformed corners.

    Parameters
    ------------
    corners_world: NDArray[np.float64]
        Array shaped ``(8, 3)`` containing corner coordinates in world RAS mm.
    t1_img: nib.Nifti1Image
        Loaded target T1 structural NIfTI image.

    Returns
    --------
        tuple[NDArray[np.uint8], NDArray[np.float64]]:
            Binary 3D uint8 mask array matching ``t1_img.shape[:3]`` and the
            8 corner coordinates in T1 voxel index space ``(8, 3)``.
    """
    dims = t1_img.shape[:3]
    inv_aff = np.linalg.inv(np.asarray(t1_img.affine, dtype=np.float64))
    hom = np.column_stack([corners_world, np.ones(corners_world.shape[0])])
    corners_vox = (inv_aff @ hom.T).T[:, :3]

    mask = np.zeros(dims, dtype=np.uint8)
    ijk_min = np.maximum(0, np.floor(np.min(corners_vox, axis=0)).astype(int) - 1)
    ijk_max = np.minimum(
        np.array(dims) - 1, np.ceil(np.max(corners_vox, axis=0)).astype(int) + 1
    )
    if np.any(ijk_min > ijk_max):
        return (mask, corners_vox)

    xs = np.arange(ijk_min[0], ijk_max[0] + 1)
    ys = np.arange(ijk_min[1], ijk_max[1] + 1)
    zs = np.arange(ijk_min[2], ijk_max[2] + 1)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    try:
        hull = Delaunay(corners_vox)
        inside = hull.find_simplex(grid) >= 0
    except QhullError:
        jitter = np.random.default_rng(0).normal(0.0, 1e-6, corners_vox.shape)
        hull = Delaunay(corners_vox + jitter)
        inside = hull.find_simplex(grid) >= 0

    pts = grid[inside]
    if pts.size > 0:
        mask[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    return (mask, corners_vox)


def save_voxel_mask_nifti(
    mask: NDArray[np.uint8],
    t1_img: nib.Nifti1Image,
    out_path: str,
) -> str:
    """Save a binary MRS voxel mask using the geometry header of the T1 NIfTI.

    Parameters
    ------------
    mask: NDArray[np.uint8]
        Binary 3D mask array matching T1 spatial dimensions.
    t1_img: nib.Nifti1Image
        Reference structural NIfTI image supplying affine and spatial header.
    out_path: str
        Output file path (``.nii`` or ``.nii.gz``).

    Returns
    --------
        str:
            Written output file path.
    """
    out_img = nib.Nifti1Image(mask.astype(np.uint8), t1_img.affine)
    out_img.header.set_data_dtype(np.uint8)
    out_img.header.set_zooms(tuple(float(z) for z in t1_img.header.get_zooms()[:3]))
    out_img.set_qform(t1_img.affine, code=1)
    out_img.set_sform(t1_img.affine, code=1)
    nib.save(out_img, out_path)
    return out_path


def _sample_world_plane(
    vol: NDArray[np.float64],
    inv_aff: NDArray[np.float64],
    world_pts: NDArray[np.float64],
    n_pix: int,
) -> NDArray[np.float64]:
    """Interpolate a 3D NIfTI volume at world RAS coordinates for a square plane.

    Parameters
    ------------
    vol: NDArray[np.float64]
        3D volume data array.
    inv_aff: NDArray[np.float64]
        Inverse 4x4 voxel-to-world affine matrix.
    world_pts: NDArray[np.float64]
        Query coordinates shaped ``(n_pix * n_pix, 3)``.
    n_pix: int
        Side resolution of the output 2D plane.

    Returns
    --------
        NDArray[np.float64]:
            2D interpolated slice shaped ``(n_pix, n_pix)``.
    """
    hom = np.column_stack([world_pts, np.ones(world_pts.shape[0])])
    ijk = (inv_aff @ hom.T)[:3, :]
    sampled = map_coordinates(vol, ijk, order=1, mode="constant", cval=0.0)
    return sampled.reshape((n_pix, n_pix))


def voxel2world_space(
    img: nib.Nifti1Image,
    vox_center_mm: NDArray[np.float64],
    plane_fov_mm: float = 240.0,
    n_pix: int = 256,
) -> dict[str, Any]:
    """Extract three orthogonal world-aligned 2D slices centered on the MRS voxel.

    Ports ``voxel2world_space.m`` by sampling ``axial``, ``coronal``, and
    ``sagittal`` planes in world RAS mm using trilinear interpolation.

    Parameters
    ------------
    img: nib.Nifti1Image
        3D NIfTI volume (T1 structural image or binary voxel mask).
    vox_center_mm: NDArray[np.float64]
        3-element voxel center in world RAS mm ``[x, y, z]``.
    plane_fov_mm: float
        Square field-of-view width in mm for extracted display slices.
    n_pix: int
        Output slice matrix resolution (``n_pix x n_pix``).

    Returns
    --------
        dict[str, Any]:
            Dictionary with keys ``axial``, ``coronal``, ``sagittal``,
            ``fov_mm``, and ``voxel_center_mm``.
    """
    vol = np.asarray(img.get_fdata(), dtype=np.float64)
    if vol.ndim > 3:
        vol = vol[..., 0]
    inv_aff = np.linalg.inv(np.asarray(img.affine, dtype=np.float64))
    cx = float(vox_center_mm[0])
    cy = float(vox_center_mm[1])
    cz = float(vox_center_mm[2])
    half = plane_fov_mm / 2.0
    grid_1d = np.linspace(-half, half, n_pix)

    gx, gy = np.meshgrid(cx + grid_1d, cy + grid_1d, indexing="xy")
    ax_pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, cz)])
    axial = _sample_world_plane(vol, inv_aff, ax_pts, n_pix)

    gx_c, gz_c = np.meshgrid(cx + grid_1d, cz + grid_1d, indexing="xy")
    cor_pts = np.column_stack([gx_c.ravel(), np.full(gx_c.size, cy), gz_c.ravel()])
    coronal = _sample_world_plane(vol, inv_aff, cor_pts, n_pix)

    gy_s, gz_s = np.meshgrid(cy + grid_1d, cz + grid_1d, indexing="xy")
    sag_pts = np.column_stack([np.full(gy_s.size, cx), gy_s.ravel(), gz_s.ravel()])
    sagittal = _sample_world_plane(vol, inv_aff, sag_pts, n_pix)

    return {
        "axial": np.flipud(axial),
        "coronal": np.flipud(coronal),
        "sagittal": np.flipud(sagittal),
        "fov_mm": plane_fov_mm,
        "voxel_center_mm": np.asarray([cx, cy, cz], dtype=np.float64),
    }


def make_three_plane_overlay(
    t1_path: str,
    mask_path: str,
    vox_center_mm: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Build a combined normalized 3-plane structural + tinted mask montage image.

    Parameters
    ------------
    t1_path: str
        Path to T1 structural NIfTI file.
    mask_path: str
        Path to binary MRS voxel mask NIfTI file.
    vox_center_mm: NDArray[np.float64] | None
        Optional world center; computed as mask centroid if ``None``.

    Returns
    --------
        NDArray[np.float64]:
            3D RGB array shaped ``(H, W, 3)`` normalized to ``[0, 1]`` for display.
    """
    t1_img = nib.load(t1_path)
    mask_img = nib.load(mask_path)
    if vox_center_mm is None:
        m_arr = np.asarray(mask_img.get_fdata())
        nz = np.argwhere(m_arr > 0)
        if nz.size > 0:
            mean_ijk = np.mean(nz, axis=0)
            vox_center_mm = (mask_img.affine @ np.append(mean_ijk, 1.0))[:3]
        else:
            vox_center_mm = np.zeros(3, dtype=np.float64)

    t1_slices = voxel2world_space(t1_img, vox_center_mm)
    mask_slices = voxel2world_space(mask_img, vox_center_mm)

    t1_row = np.concatenate(
        [t1_slices["sagittal"], t1_slices["coronal"], t1_slices["axial"]], axis=1
    )
    m_row = np.concatenate(
        [mask_slices["sagittal"], mask_slices["coronal"], mask_slices["axial"]], axis=1
    )
    max_val = (
        float(np.percentile(t1_row[t1_row > 0], 99)) if np.any(t1_row > 0) else 1.0
    )
    bg = np.clip(t1_row / max(max_val, 1e-12), 0.0, 1.0)
    rgb = np.stack([bg, bg, bg], axis=-1)
    overlay_mask = m_row > 0.3
    rgb[overlay_mask, 0] = np.clip(0.5 * bg[overlay_mask] + 0.5, 0.0, 1.0)
    rgb[overlay_mask, 1] = 0.5 * bg[overlay_mask]
    rgb[overlay_mask, 2] = 0.5 * bg[overlay_mask]
    return rgb


def build_mask_output_path(
    mrs_file: str,
    t1_file: str,
    vox_name: str = "vox1",
    num_voxels: int = 1,
) -> str:
    """Construct the standard Gannet ``*_mask.nii`` filename in the T1/MRS folder.

    Parameters
    ------------
    mrs_file: str
        Path to the source MRS file.
    t1_file: str
        Path to the T1 structural file.
    vox_name: str
        Voxel tag (e.g., ``vox1``) used when multiple voxels exist.
    num_voxels: int
        Number of MRS voxels in the scan.

    Returns
    --------
        str:
            Output path ending in ``_mask.nii``.
    """
    mrs_dir = os.path.dirname(os.path.abspath(mrs_file))
    mrs_base = os.path.basename(mrs_file)
    for ext in (
        ".nii.gz",
        ".sdat",
        ".SPAR",
        ".spar",
        ".rda",
        ".dat",
        ".dcm",
        ".ima",
        ".7",
        ".nii",
        ".data",
    ):
        if mrs_base.lower().endswith(ext.lower()):
            mrs_base = mrs_base[: -len(ext)]
            break
    suffix = f"_{vox_name}" if num_voxels > 1 else ""
    out_dir = (
        mrs_dir if os.path.isdir(mrs_dir) else os.path.dirname(os.path.abspath(t1_file))
    )
    return os.path.join(out_dir, f"{mrs_base}{suffix}_mask.nii")
