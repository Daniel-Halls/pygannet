"""FSL wrappers (BET, FAST, FLIRT, FNIRT, applywarp) for T1 segmentation and MNI."""

from __future__ import annotations

import os
import shutil
import subprocess

import nibabel as nib
import numpy as np

from gannet.segment.segmentation import (
    find_existing_tissue_maps,
    segment_t1_python,
)


def _strip_nii_ext(path: str) -> str:
    """Strip ``.nii`` or ``.nii.gz`` extension from a NIfTI path.

    Parameters
    ------------
    path: str
        Input NIfTI file path.

    Returns
    --------
        str:
            Path stem without NIfTI suffix.
    """
    if path.lower().endswith(".nii.gz"):
        return path[:-7]
    if path.lower().endswith(".nii"):
        return path[:-4]
    return os.path.splitext(path)[0]


def _resolve_nii_path(prefix: str) -> str | None:
    """Resolve an existing ``.nii.gz`` or ``.nii`` path given its output prefix.

    Parameters
    ------------
    prefix: str
        Output file prefix without NIfTI suffix.

    Returns
    --------
        str | None:
            Existing file path or ``None`` if neither extension exists.
    """
    for ext in (".nii.gz", ".nii"):
        cand = f"{prefix}{ext}"
        if os.path.exists(cand):
            return cand
    return None


def find_fsl_bin(tool: str) -> str | None:
    """Locate an FSL executable on ``PATH`` or inside ``$FSLDIR``.

    Parameters
    ------------
    tool: str
        Name of FSL CLI binary (e.g., ``bet``, ``fast``, ``flirt``).

    Returns
    --------
        str | None:
            Absolute path to executable, or ``None`` if unavailable.
    """
    found = shutil.which(tool)
    if found is not None:
        return found
    fsldir = os.environ.get("FSLDIR", "")
    if fsldir:
        for rel in ("share/fsl/bin", "bin"):
            cand = os.path.join(fsldir, rel, tool)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


def get_mni152_template(brain: bool = True, res_mm: int = 2) -> str | None:
    """Locate standard MNI152 T1 template shipped with FSL under ``$FSLDIR``.

    Parameters
    ------------
    brain: bool
        Return skull-stripped template when ``True`` or full-head when ``False``.
    res_mm: int
        Spatial resolution in mm (``1`` or ``2``).

    Returns
    --------
        str | None:
            Path to the MNI152 template NIfTI file, or ``None`` if not found.
    """
    fsldir = os.environ.get("FSLDIR", "")
    if not fsldir:
        flirt_bin = find_fsl_bin("flirt")
        if flirt_bin:
            # .../fsl/share/fsl/bin/flirt -> .../fsl
            fsldir = os.path.abspath(
                os.path.join(os.path.dirname(flirt_bin), "..", "..", "..")
            )
    if not fsldir:
        return None
    suffix = "_brain" if brain else ""
    cand = os.path.join(
        fsldir, "data", "standard", f"MNI152_T1_{res_mm}mm{suffix}.nii.gz"
    )
    if os.path.exists(cand):
        return cand
    return None


def run_fsl_bet(
    t1_path: str,
    out_brain_path: str | None = None,
    frac: float = 0.5,
    robust: bool = True,
) -> dict[str, str]:
    """Run FSL ``bet`` skull-stripping on a T1-weighted structural MRI.

    Parameters
    ------------
    t1_path: str
        Input structural T1 NIfTI path.
    out_brain_path: str | None
        Output skull-stripped brain path (defaults to ``<stem>_brain.nii.gz``).
    frac: float
        Fractional intensity threshold ``-f`` in ``(0, 1)``.
    robust: bool
        Enable ``-R`` robust brain center estimation.

    Returns
    --------
        dict[str, str]:
            Mapping containing ``brain`` and ``brain_mask`` NIfTI paths.
    """
    bet_bin = find_fsl_bin("bet")
    if bet_bin is None:
        raise FileNotFoundError("FSL 'bet' executable not found.")

    stem = _strip_nii_ext(os.path.abspath(t1_path))
    target_prefix = (
        _strip_nii_ext(os.path.abspath(out_brain_path))
        if out_brain_path is not None
        else f"{stem}_brain"
    )
    cmd = [bet_bin, os.path.abspath(t1_path), target_prefix, "-f", f"{frac:.3f}", "-m"]
    if robust:
        cmd.append("-R")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FSL bet failed ({proc.returncode}): {proc.stderr}")

    brain_file = _resolve_nii_path(target_prefix) or f"{target_prefix}.nii.gz"
    mask_file = (
        _resolve_nii_path(f"{target_prefix}_mask") or f"{target_prefix}_mask.nii.gz"
    )
    return {"brain": brain_file, "brain_mask": mask_file}


def run_fsl_fast(
    brain_path: str,
    out_prefix: str | None = None,
    n_classes: int = 3,
) -> dict[str, str]:
    """Run FSL ``fast`` 3-class T1 tissue segmentation (CSF, GM, WM).

    Parameters
    ------------
    brain_path: str
        Path to skull-stripped T1 brain NIfTI produced by ``bet``.
    out_prefix: str | None
        Output filename prefix for FAST (defaults to ``<stem>`` of ``brain_path``).
    n_classes: int
        Number of tissue classes (default ``3``: CSF=pve_0, GM=pve_1, WM=pve_2).

    Returns
    --------
        dict[str, str]:
            Dictionary mapping ``c1`` (GM), ``c2`` (WM), ``c3`` (CSF), and
            ``pveseg`` to the FAST output NIfTI file paths.
    """
    fast_bin = find_fsl_bin("fast")
    if fast_bin is None:
        raise FileNotFoundError("FSL 'fast' executable not found.")

    prefix = out_prefix or _strip_nii_ext(os.path.abspath(brain_path))
    cmd = [
        fast_bin,
        "-t",
        "1",
        "-n",
        str(n_classes),
        "-o",
        prefix,
        os.path.abspath(brain_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FSL fast failed ({proc.returncode}): {proc.stderr}")

    pve0 = _resolve_nii_path(f"{prefix}_pve_0")
    pve1 = _resolve_nii_path(f"{prefix}_pve_1")
    pve2 = _resolve_nii_path(f"{prefix}_pve_2")
    pveseg = _resolve_nii_path(f"{prefix}_pveseg")
    if not (pve0 and pve1 and pve2):
        raise FileNotFoundError(
            f"FSL fast did not produce expected pve outputs at {prefix}"
        )

    return {
        "c1": pve1,
        "c2": pve2,
        "c3": pve0,
        "pve_0": pve0,
        "pve_1": pve1,
        "pve_2": pve2,
        "pveseg": pveseg or "",
    }


def _create_background_map(brain_mask_path: str, out_bg_path: str) -> str:
    """Create complement background probability map ``c6 = 1 - brain_mask``.

    Parameters
    ------------
    brain_mask_path: str
        Binary FSL BET brain mask NIfTI path.
    out_bg_path: str
        Destination path for background mask NIfTI.

    Returns
    --------
        str:
            Written background NIfTI path.
    """
    img = nib.load(brain_mask_path)
    mask_data = np.asarray(img.get_fdata(), dtype=np.float32)
    bg = np.clip(1.0 - mask_data, 0.0, 1.0).astype(np.float32)
    out_img = nib.Nifti1Image(bg, img.affine)
    out_img.set_qform(img.affine, code=1)
    out_img.set_sform(img.affine, code=1)
    nib.save(out_img, out_bg_path)
    return out_bg_path


def call_fsl_segmentation(
    t1_path: str,
    bet_frac: float = 0.5,
    backend: str = "fsl",
) -> dict[str, str]:
    """Run FSL ``bet`` + ``fast`` segmentation (or reuse existing / fallback).

    Parameters
    ------------
    t1_path: str
        Path to the structural T1-weighted NIfTI volume.
    bet_frac: float
        Fractional intensity threshold ``-f`` for FSL ``bet``.
    backend: str
        Backend selection: ``fsl``, ``auto``, or ``python``.

    Returns
    --------
        dict[str, str]:
            Dictionary mapping ``c1`` (GM), ``c2`` (WM), ``c3`` (CSF),
            ``c6`` (background), and optional ``brain`` / ``brain_mask``.
    """
    existing = find_existing_tissue_maps(t1_path)
    if existing:
        if "c6" not in existing:
            ref_img = nib.load(t1_path)
            c1 = np.asarray(nib.load(existing["c1"]).get_fdata(), dtype=np.float32)
            c2 = np.asarray(nib.load(existing["c2"]).get_fdata(), dtype=np.float32)
            c3 = np.asarray(nib.load(existing["c3"]).get_fdata(), dtype=np.float32)
            bg = np.clip(1.0 - (c1 + c2 + c3), 0.0, 1.0).astype(np.float32)
            bg_path = f"{_strip_nii_ext(os.path.abspath(t1_path))}_bg.nii.gz"
            out_img = nib.Nifti1Image(bg, ref_img.affine)
            nib.save(out_img, bg_path)
            existing["c6"] = bg_path
        return existing

    use_fsl = backend.lower() in ("fsl", "auto") and (
        find_fsl_bin("bet") is not None and find_fsl_bin("fast") is not None
    )
    if not use_fsl:
        return segment_t1_python(t1_path)

    bet_res = run_fsl_bet(t1_path, frac=bet_frac, robust=True)
    fast_res = run_fsl_fast(bet_res["brain"])
    bg_path = f"{_strip_nii_ext(os.path.abspath(t1_path))}_bg.nii.gz"
    _create_background_map(bet_res["brain_mask"], bg_path)

    return {
        "c1": fast_res["c1"],
        "c2": fast_res["c2"],
        "c3": fast_res["c3"],
        "c6": bg_path,
        "brain": bet_res["brain"],
        "brain_mask": bet_res["brain_mask"],
        "backend": "fsl",
    }


def run_fsl_flirt(
    in_vol: str,
    ref_vol: str,
    out_vol: str | None = None,
    out_mat: str | None = None,
    dof: int = 12,
) -> dict[str, str]:
    """Run FSL ``flirt`` linear image registration.

    Parameters
    ------------
    in_vol: str
        Source subject brain/volume NIfTI path.
    ref_vol: str
        Target reference template NIfTI path (e.g., MNI152 2mm brain).
    out_vol: str | None
        Optional resampled output volume NIfTI path.
    out_mat: str | None
        Destination 4x4 affine ``.mat`` path.
    dof: int
        Degrees of freedom (6 rigid, 12 affine). Defaults to ``12``.

    Returns
    --------
        dict[str, str]:
            Dictionary containing ``mat`` and optionally ``out`` paths.
    """
    flirt_bin = find_fsl_bin("flirt")
    if flirt_bin is None:
        raise FileNotFoundError("FSL 'flirt' executable not found.")

    mat_path = out_mat or f"{_strip_nii_ext(os.path.abspath(in_vol))}_to_MNI.mat"
    cmd = [
        flirt_bin,
        "-in",
        os.path.abspath(in_vol),
        "-ref",
        os.path.abspath(ref_vol),
        "-omat",
        mat_path,
        "-dof",
        str(dof),
    ]
    if out_vol is not None:
        cmd.extend(["-out", out_vol])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FSL flirt failed ({proc.returncode}): {proc.stderr}")

    res: dict[str, str] = {"mat": mat_path}
    if out_vol is not None:
        res["out"] = out_vol
    return res


def run_fsl_fnirt(
    in_head: str,
    ref_head: str,
    aff_mat: str,
    out_warp: str | None = None,
) -> str:
    """Run FSL ``fnirt`` nonlinear registration initialized from a FLIRT matrix.

    Parameters
    ------------
    in_head: str
        Unstripped full-head subject T1 NIfTI path.
    ref_head: str
        Full-head reference template NIfTI path (e.g., ``MNI152_T1_2mm``).
    aff_mat: str
        Affine initialization matrix generated by ``flirt``.
    out_warp: str | None
        Destination warp-field coefficient NIfTI path.

    Returns
    --------
        str:
            Path to the computed FNIRT spline warp-field NIfTI.
    """
    fnirt_bin = find_fsl_bin("fnirt")
    if fnirt_bin is None:
        raise FileNotFoundError("FSL 'fnirt' executable not found.")

    warp_path = (
        out_warp or f"{_strip_nii_ext(os.path.abspath(in_head))}_to_MNI_warp.nii.gz"
    )
    cmd = [
        fnirt_bin,
        f"--in={os.path.abspath(in_head)}",
        f"--ref={os.path.abspath(ref_head)}",
        f"--aff={os.path.abspath(aff_mat)}",
        f"--cout={warp_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FSL fnirt failed ({proc.returncode}): {proc.stderr}")
    return warp_path


def apply_fsl_transform_to_mask(
    mask_path: str,
    ref_vol: str,
    out_path: str,
    aff_mat: str | None = None,
    warp_field: str | None = None,
) -> str:
    """Warp a binary MRS voxel mask into MNI space using ``applywarp`` or ``flirt``.

    Parameters
    ------------
    mask_path: str
        Subject-space binary MRS voxel mask NIfTI path.
    ref_vol: str
        MNI152 reference template NIfTI path.
    out_path: str
        Output MNI-space binary mask NIfTI path.
    aff_mat: str | None
        Linear transformation matrix from ``flirt``.
    warp_field: str | None
        Optional nonlinear warp coefficient field from ``fnirt``.

    Returns
    --------
        str:
            Path to the warped MNI binary voxel mask.
    """
    if warp_field is not None:
        applywarp_bin = find_fsl_bin("applywarp")
        if applywarp_bin is None:
            raise FileNotFoundError("FSL 'applywarp' executable not found.")
        cmd = [
            applywarp_bin,
            f"--in={os.path.abspath(mask_path)}",
            f"--ref={os.path.abspath(ref_vol)}",
            f"--warp={os.path.abspath(warp_field)}",
            f"--out={out_path}",
            "--interp=nn",
        ]
    elif aff_mat is not None:
        flirt_bin = find_fsl_bin("flirt")
        if flirt_bin is None:
            raise FileNotFoundError("FSL 'flirt' executable not found.")
        cmd = [
            flirt_bin,
            "-in",
            os.path.abspath(mask_path),
            "-ref",
            os.path.abspath(ref_vol),
            "-applyxfm",
            "-init",
            os.path.abspath(aff_mat),
            "-interp",
            "nearestneighbour",
            "-out",
            out_path,
        ]
    else:
        raise ValueError("Either aff_mat or warp_field must be provided.")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FSL mask warp failed ({proc.returncode}): {proc.stderr}")
    return _resolve_nii_path(_strip_nii_ext(out_path)) or out_path
