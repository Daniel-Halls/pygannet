"""T1-weighted tissue segmentation, MRIQC QA metrics, and MNI mask normalization."""

from __future__ import annotations

import os
import re

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import affine_transform, binary_fill_holes, gaussian_filter

from gannet.config import MRSStruct


def _prefix_t1_filename(t1_path: str, prefix: str) -> str:
    """Build an SPM-convention prefixed NIfTI filename alongside ``t1_path``.

    Parameters
    ------------
    t1_path: str
        Path to the structural T1 NIfTI image (``.nii`` or ``.nii.gz``).
    prefix: str
        Prefix string such as ``c1``, ``c2``, ``c3``, ``c6``, or ``y_``.

    Returns
    --------
        str:
            Path with ``prefix`` prepended to the base filename (saved as ``.nii``).
    """
    folder = os.path.dirname(os.path.abspath(t1_path))
    base = os.path.basename(t1_path)
    if base.lower().endswith(".nii.gz"):
        base = base[:-3]
    elif not base.lower().endswith(".nii"):
        base = f"{base}.nii"
    return os.path.join(folder, f"{prefix}{base}")


def find_existing_tissue_maps(t1_path: str) -> dict[str, str]:
    """Locate existing FSL FAST, ``fsl_anat``, BIDS ``probseg``, or ``c1..c3`` maps.

    Parameters
    ------------
    t1_path: str
        Path to the T1-weighted structural NIfTI image.

    Returns
    --------
        dict[str, str]:
            Mapping of keys ``c1`` (GM), ``c2`` (WM), ``c3`` (CSF), and
            optionally ``c6`` (background), ``brain``, ``brain_mask``, and
            ``y`` to file paths when found on disk; empty dict if incomplete.
    """
    t1_abs = os.path.abspath(t1_path)
    t1_dir = os.path.dirname(t1_abs)
    t1_stem = re.sub(r"\.nii(\.gz)?$", "", t1_abs, flags=re.IGNORECASE)

    # 1. Check FSL FAST (*_brain_pve_{0,1,2} or *_pve_{0,1,2}) and fsl_anat outputs
    fsl_prefixes = [
        f"{t1_stem}_brain",
        t1_stem,
        os.path.join(f"{t1_stem}.anat", "T1_fast"),
    ]
    for pref in fsl_prefixes:
        fsl_hits: dict[str, str] = {}
        for key, pve_idx in (("c3", 0), ("c1", 1), ("c2", 2)):
            for ext in (".nii.gz", ".nii"):
                cand = f"{pref}_pve_{pve_idx}{ext}"
                if os.path.exists(cand):
                    fsl_hits[key] = cand
                    break
        if all(k in fsl_hits for k in ("c1", "c2", "c3")):
            for ext in (".nii.gz", ".nii"):
                if os.path.exists(f"{t1_stem}_brain{ext}"):
                    fsl_hits["brain"] = f"{t1_stem}_brain{ext}"
                if os.path.exists(f"{t1_stem}_bg{ext}"):
                    fsl_hits["c6"] = f"{t1_stem}_bg{ext}"
                if os.path.exists(f"{t1_stem}_brain_mask{ext}"):
                    fsl_hits["brain_mask"] = f"{t1_stem}_brain_mask{ext}"
            return fsl_hits

    # 2. Check SPM-style prefixed c1/c2/c3/c6 maps
    maps: dict[str, str] = {}
    for code in ("c1", "c2", "c3", "c6", "y_"):
        cand = _prefix_t1_filename(t1_path, code)
        for test_path in (cand, f"{cand}.gz"):
            if os.path.exists(test_path):
                key = "y" if code == "y_" else code
                maps[key] = test_path
                break

    if all(k in maps for k in ("c1", "c2", "c3")):
        return maps

    # 3. Check BIDS *_label-{GM,WM,CSF}_probseg.nii[.gz] naming convention
    t1_base = re.sub(r"_T1w\.nii(\.gz)?$", "", os.path.basename(t1_path))
    bids_hits: dict[str, str] = {}
    for key, label in (("c1", "GM"), ("c2", "WM"), ("c3", "CSF")):
        for suffix in (
            f"_label-{label}_probseg.nii.gz",
            f"_label-{label}_probseg.nii",
        ):
            cand = os.path.join(t1_dir, f"{t1_base}{suffix}")
            if os.path.exists(cand):
                bids_hits[key] = cand
                break
    if all(k in bids_hits for k in ("c1", "c2", "c3")):
        return bids_hits
    return {}


def _save_prob_nifti(
    data: NDArray[np.float32],
    ref_img: nib.Nifti1Image,
    out_path: str,
) -> str:
    """Write a float32 probability map NIfTI sharing geometry with ``ref_img``.

    Parameters
    ------------
    data: NDArray[np.float32]
        3D probability map clipped to ``[0, 1]``.
    ref_img: nib.Nifti1Image
        Reference structural NIfTI image.
    out_path: str
        Destination NIfTI file path.

    Returns
    --------
        str:
            Written file path.
    """
    out_img = nib.Nifti1Image(
        np.clip(data, 0.0, 1.0).astype(np.float32), ref_img.affine
    )
    out_img.header.set_data_dtype(np.float32)
    out_img.header.set_zooms(tuple(float(z) for z in ref_img.header.get_zooms()[:3]))
    out_img.set_qform(ref_img.affine, code=1)
    out_img.set_sform(ref_img.affine, code=1)
    nib.save(out_img, out_path)
    return out_path


def segment_t1_python(t1_path: str, n_iter: int = 12) -> dict[str, str]:
    """Perform pure-Python 3-tissue + background EM-GMM T1 MRI segmentation.

    Produces SPM12-compatible probability maps ``c1`` (GM), ``c2`` (WM),
    ``c3`` (CSF), and ``c6`` (background/non-brain) alongside ``t1_path``.

    Parameters
    ------------
    t1_path: str
        Path to the T1-weighted structural NIfTI file.
    n_iter: int
        Number of EM refinement iterations.

    Returns
    --------
        dict[str, str]:
            Dictionary mapping ``c1``, ``c2``, ``c3``, ``c6`` to written NIfTI paths.
    """
    img = nib.load(t1_path)
    vol = np.asarray(img.get_fdata(), dtype=np.float64)
    if vol.ndim > 3:
        vol = vol[..., 0]
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.maximum(vol, 0.0)

    smooth = gaussian_filter(vol, sigma=1.0)
    nonzero_vals = smooth[smooth > 0]
    if nonzero_vals.size == 0:
        zeros = np.zeros_like(vol, dtype=np.float32)
        paths = {
            "c1": _save_prob_nifti(zeros, img, _prefix_t1_filename(t1_path, "c1")),
            "c2": _save_prob_nifti(zeros, img, _prefix_t1_filename(t1_path, "c2")),
            "c3": _save_prob_nifti(zeros, img, _prefix_t1_filename(t1_path, "c3")),
            "c6": _save_prob_nifti(
                np.ones_like(zeros), img, _prefix_t1_filename(t1_path, "c6")
            ),
        }
        return paths

    # Coarse foreground / brain mask via intensity thresholding + hole fill
    thresh = float(np.percentile(nonzero_vals, 15)) * 0.5
    fg_mask = binary_fill_holes(smooth > thresh)
    air_mask = ~fg_mask

    # Mild multiplicative bias correction inside foreground
    fg_smooth = gaussian_filter(np.where(fg_mask, vol, 0.0), sigma=12.0)
    fg_weight = gaussian_filter(fg_mask.astype(np.float64), sigma=12.0)
    bias_field = np.where(
        fg_weight > 0.05, fg_smooth / np.maximum(fg_weight, 1e-6), 1.0
    )
    med_fg = float(np.median(vol[fg_mask])) if np.any(fg_mask) else 1.0
    norm_vol = np.where(
        fg_mask & (bias_field > 1e-6),
        vol * (med_fg / np.maximum(bias_field, 1e-6)),
        vol,
    )

    fg_vals = norm_vol[fg_mask]
    # 3-class initialization: CSF (low), GM (mid), WM (high)
    means = np.percentile(fg_vals, [20.0, 55.0, 85.0]).astype(np.float64)
    iqr = max(float(np.percentile(fg_vals, 75.0) - np.percentile(fg_vals, 25.0)), 1e-3)
    stds = np.array([iqr * 0.35, iqr * 0.25, iqr * 0.22], dtype=np.float64)
    pis = np.array([0.22, 0.43, 0.35], dtype=np.float64)

    resp = np.zeros((3, fg_vals.size), dtype=np.float64)
    for _ in range(n_iter):
        for k in range(3):
            diff = (fg_vals - means[k]) / max(stds[k], 1e-6)
            resp[k] = (
                pis[k]
                * np.exp(-0.5 * diff * diff)
                / (max(stds[k], 1e-6) * np.sqrt(2.0 * np.pi))
            )
        denom = np.sum(resp, axis=0) + 1e-12
        resp /= denom
        nk = np.sum(resp, axis=1)
        for k in range(3):
            if nk[k] > 1.0:
                means[k] = float(np.sum(resp[k] * fg_vals) / nk[k])
                var_k = float(np.sum(resp[k] * (fg_vals - means[k]) ** 2) / nk[k])
                stds[k] = max(float(np.sqrt(var_k)), 1e-3 * med_fg)
                pis[k] = float(nk[k] / fg_vals.size)
        order = np.argsort(means)
        means, stds, pis, resp = (means[order], stds[order], pis[order], resp[order])

    # Resp order after sorting: 0=CSF (c3), 1=GM (c1), 2=WM (c2)
    csf_prob = np.zeros_like(vol, dtype=np.float64)
    gm_prob = np.zeros_like(vol, dtype=np.float64)
    wm_prob = np.zeros_like(vol, dtype=np.float64)
    csf_prob[fg_mask] = resp[0]
    gm_prob[fg_mask] = resp[1]
    wm_prob[fg_mask] = resp[2]

    # Spatial regularization for smooth tissue probabilities
    gm_prob = gaussian_filter(gm_prob, sigma=0.5) * fg_mask
    wm_prob = gaussian_filter(wm_prob, sigma=0.5) * fg_mask
    csf_prob = gaussian_filter(csf_prob, sigma=0.5) * fg_mask
    tot = gm_prob + wm_prob + csf_prob
    safe_tot = np.where(tot > 1e-8, tot, 1.0)
    gm_prob = np.where(fg_mask, gm_prob / safe_tot, 0.0)
    wm_prob = np.where(fg_mask, wm_prob / safe_tot, 0.0)
    csf_prob = np.where(fg_mask, csf_prob / safe_tot, 0.0)
    bg_prob = air_mask.astype(np.float64)

    paths = {
        "c1": _save_prob_nifti(
            gm_prob.astype(np.float32), img, _prefix_t1_filename(t1_path, "c1")
        ),
        "c2": _save_prob_nifti(
            wm_prob.astype(np.float32), img, _prefix_t1_filename(t1_path, "c2")
        ),
        "c3": _save_prob_nifti(
            csf_prob.astype(np.float32), img, _prefix_t1_filename(t1_path, "c3")
        ),
        "c6": _save_prob_nifti(
            bg_prob.astype(np.float32), img, _prefix_t1_filename(t1_path, "c6")
        ),
    }
    return paths


def call_spm12_segmentation(t1_path: str) -> dict[str, str]:
    """Deprecated alias delegating to FSL-backed ``call_fsl_segmentation``.

    Parameters
    ------------
    t1_path: str
        Path to the T1-weighted structural NIfTI image.

    Returns
    --------
        dict[str, str]:
            Dictionary mapping ``c1``, ``c2``, ``c3``, and ``c6`` to paths.
    """
    from gannet.segment.fsl import call_fsl_segmentation

    return call_fsl_segmentation(t1_path, backend="auto")


def compute_mriqc_metrics(
    t1_vol: NDArray[np.float64],
    gm_prob: NDArray[np.float64],
    wm_prob: NDArray[np.float64],
    csf_prob: NDArray[np.float64],
    air_prob: NDArray[np.float64] | None = None,
    prob_threshold: float = 0.9,
) -> dict[str, dict[str, float]]:
    """Compute structural MRI quality metrics reproducing ``GannetSegment.m``.

    Computes SNR (GM/WM/CSF/total), Dietrich SNR (``SNR_D``), CNR, CV,
    CJV, EFC, FBER, and WM2MAX across masks thresholded at ``prob_threshold``.

    Parameters
    ------------
    t1_vol: NDArray[np.float64]
        3D T1 structural MRI intensity volume.
    gm_prob: NDArray[np.float64]
        Gray matter probability map matching ``t1_vol``.
    wm_prob: NDArray[np.float64]
        White matter probability map matching ``t1_vol``.
    csf_prob: NDArray[np.float64]
        CSF probability map matching ``t1_vol``.
    air_prob: NDArray[np.float64] | None
        Background air probability map ``c6`` (inferred as ``1 - brain`` if omitted).
    prob_threshold: float
        Tissue probability threshold (default ``0.9``).

    Returns
    --------
        dict[str, dict[str, float]]:
            Nested dictionary containing ``SNR``, ``SNR_D``, ``CNR``,
            ``CV``, ``CJV``, ``EFC``, ``FBER``, and ``WM2MAX``.
    """
    vol = np.nan_to_num(np.asarray(t1_vol, dtype=np.float64), nan=0.0)
    gm_mask = gm_prob >= prob_threshold
    wm_mask = wm_prob >= prob_threshold
    csf_mask = csf_prob >= prob_threshold
    if air_prob is not None:
        air_mask = air_prob >= prob_threshold
    else:
        air_mask = (gm_prob + wm_prob + csf_prob) < 0.05

    brain_mask = gm_mask | wm_mask | csf_mask

    def _safe_mean(mask: NDArray[np.bool_]) -> float:
        """Compute masked mean safely."""
        return float(np.mean(vol[mask])) if np.any(mask) else 0.0

    def _safe_std(mask: NDArray[np.bool_], ddof: int = 0) -> float:
        """Compute masked standard deviation safely."""
        vals = vol[mask]
        if vals.size <= ddof:
            return 1e-12
        s = float(np.std(vals, ddof=ddof))
        return s if s > 0 else 1e-12

    mu_gm = _safe_mean(gm_mask)
    mu_wm = _safe_mean(wm_mask)
    mu_csf = _safe_mean(csf_mask)
    sd_gm = _safe_std(gm_mask)
    sd_wm = _safe_std(wm_mask)
    sd_csf = _safe_std(csf_mask)

    n_gm = max(1, int(np.sum(gm_mask)))
    n_wm = max(1, int(np.sum(wm_mask)))
    n_csf = max(1, int(np.sum(csf_mask)))

    # Dietrich SNR noise estimate from air voxels (Rayleigh bias correction)
    if np.any(air_mask):
        sigma_air = max(float(np.std(vol[air_mask])), 1e-12)
        bg_vals = vol[air_mask]
        rms_bg = float(np.sqrt(np.mean(bg_vals * bg_vals)))
    else:
        sigma_air = max(float(np.std(vol)) * 0.05, 1e-12)
        rms_bg = sigma_air

    snr_gm = mu_gm / (sd_gm * np.sqrt(n_gm / max(1, n_gm - 1)))
    snr_wm = mu_wm / (sd_wm * np.sqrt(n_wm / max(1, n_wm - 1)))
    snr_csf = mu_csf / (sd_csf * np.sqrt(n_csf / max(1, n_csf - 1)))
    snr_tot = float(np.mean([snr_gm, snr_wm, snr_csf]))

    rayleigh_factor = np.sqrt(2.0 / (4.0 - np.pi))
    snr_d_gm = mu_gm / (rayleigh_factor * sigma_air)
    snr_d_wm = mu_wm / (rayleigh_factor * sigma_air)
    snr_d_csf = mu_csf / (rayleigh_factor * sigma_air)
    snr_d_tot = float(np.mean([snr_d_gm, snr_d_wm, snr_d_csf]))

    cnr = abs(mu_wm - mu_gm) / np.sqrt(sigma_air**2 + sd_wm**2 + sd_gm**2)
    cv = (
        float(np.std(vol[brain_mask]) / max(_safe_mean(brain_mask), 1e-12))
        if np.any(brain_mask)
        else 0.0
    )
    cjv = (sd_wm + sd_gm) / max(abs(mu_wm - mu_gm), 1e-12)

    bmax = float(np.sqrt(np.sum(vol * vol)))
    if bmax > 0:
        b = np.abs(vol) / bmax
        efc = float(
            -np.sum((b / 1.0) * np.log(np.maximum(b, 1e-16)))
            / max(vol.size * np.sqrt(1.0 / vol.size), 1.0)
        )
    else:
        efc = 0.0

    head_med = float(np.median(vol[brain_mask] ** 2)) if np.any(brain_mask) else 0.0
    fber = head_med / max(rms_bg**2, 1e-12)
    p9995 = float(np.percentile(vol, 99.95)) if vol.size > 0 else 1.0
    wm2max = mu_wm / max(p9995, 1e-12)

    return {
        "SNR": {"GM": snr_gm, "WM": snr_wm, "CSF": snr_csf, "total": snr_tot},
        "SNR_D": {"GM": snr_d_gm, "WM": snr_d_wm, "CSF": snr_d_csf, "total": snr_d_tot},
        "CNR": {"val": float(cnr)},
        "CV": {"val": float(cv)},
        "CJV": {"val": float(cjv)},
        "EFC": {"val": float(efc)},
        "FBER": {"val": float(fber)},
        "WM2MAX": {"val": float(wm2max)},
    }


def normalize_voxel_mask(
    mask_path: str,
    out_path: str | None = None,
    t1_path: str | None = None,
    brain_path: str | None = None,
    use_fnirt: bool = False,
    backend: str = "fsl",
) -> str:
    """Register structural MRI to MNI-152 via FSL FLIRT/FNIRT and warp voxel mask.

    Falls back to affine voxel-to-world resampling onto the 2 mm MNI-152 grid
    when FSL or a structural reference is unavailable.

    Parameters
    ------------
    mask_path: str
        Path to subject-space binary MRS voxel mask NIfTI file.
    out_path: str | None
        Destination path for MNI-normalized mask; defaults to ``w<name>`` in
        the same directory.
    t1_path: str | None
        Optional subject full-head T1 structural NIfTI path for FLIRT/FNIRT.
    brain_path: str | None
        Optional skull-stripped T1 brain path produced by FSL ``bet``.
    use_fnirt: bool
        Also run nonlinear FSL ``fnirt`` + ``applywarp`` after ``flirt``.
    backend: str
        Registration backend: ``fsl``, ``auto``, or ``python``.

    Returns
    --------
        str:
            Path to the written normalized NIfTI mask file.
    """
    if out_path is None:
        folder = os.path.dirname(os.path.abspath(mask_path))
        base = os.path.basename(mask_path)
        if base.lower().endswith(".nii.gz"):
            base = base[:-3]
        out_path = os.path.join(folder, f"w{base}")

    if backend.lower() in ("fsl", "auto") and (brain_path or t1_path):
        from gannet.segment.fsl import (
            apply_fsl_transform_to_mask,
            find_fsl_bin,
            get_mni152_template,
            run_fsl_flirt,
            run_fsl_fnirt,
        )

        mni_brain = get_mni152_template(brain=True, res_mm=2)
        if find_fsl_bin("flirt") is not None and mni_brain is not None:
            reg_src = brain_path or t1_path or ""
            reg_stem = re.sub(
                r"\.nii(\.gz)?$", "", os.path.abspath(reg_src), flags=re.IGNORECASE
            )
            aff_mat = f"{reg_stem}_to_MNI.mat"
            if not os.path.exists(aff_mat):
                flirt_res = run_fsl_flirt(reg_src, mni_brain, out_mat=aff_mat, dof=12)
                aff_mat = flirt_res["mat"]

            warp_field: str | None = None
            mni_head = get_mni152_template(brain=False, res_mm=2)
            if use_fnirt and t1_path and mni_head and find_fsl_bin("fnirt"):
                t1_stem = re.sub(
                    r"\.nii(\.gz)?$",
                    "",
                    os.path.abspath(t1_path),
                    flags=re.IGNORECASE,
                )
                cand_warp = f"{t1_stem}_to_MNI_warp.nii.gz"
                if not os.path.exists(cand_warp):
                    cand_warp = run_fsl_fnirt(
                        t1_path, mni_head, aff_mat, out_warp=cand_warp
                    )
                warp_field = cand_warp
                return apply_fsl_transform_to_mask(
                    mask_path, mni_head, out_path, warp_field=warp_field
                )

            return apply_fsl_transform_to_mask(
                mask_path, mni_brain, out_path, aff_mat=aff_mat
            )

    mask_img = nib.load(mask_path)
    mask_data = np.asarray(mask_img.get_fdata(), dtype=np.float64)
    if mask_data.ndim > 3:
        mask_data = mask_data[..., 0]

    # Standard FSL/SPM MNI152 2mm grid (91 x 109 x 91) RAS affine
    mni_shape = (91, 109, 91)
    mni_affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    inv_mask_aff = np.linalg.inv(np.asarray(mask_img.affine, dtype=np.float64))
    mapping = inv_mask_aff @ mni_affine
    resampled = affine_transform(
        mask_data,
        mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=mni_shape,
        order=1,
        mode="constant",
        cval=0.0,
    )
    bin_mni = (resampled >= 0.5).astype(np.uint8)

    out_img = nib.Nifti1Image(bin_mni, mni_affine)
    out_img.set_qform(mni_affine, code=4)
    out_img.set_sform(mni_affine, code=4)
    nib.save(out_img, out_path)
    return out_path


def voxel_mask_overlap(mrs_struct: MRSStruct) -> MRSStruct:
    """Compute across-subject MNI voxel overlap map from normalized masks.

    Ports ``VoxelMaskOverlap.m`` by summing normalized ``w*_mask.nii`` files
    across all scans in ``MRSStruct``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` populated by ``gannet_segment``.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with group overlap slice/array stored in
            ``mask.<vox>.overlap``.
    """
    vox_names = mrs_struct["p"].get("Vox", ["vox1"])[: int(mrs_struct["p"]["numVox"])]
    n_scans = int(mrs_struct["p"]["numScans"])

    for vox in vox_names:
        vox_mask = mrs_struct.get("mask", {}).get(vox, {})
        norm_files = vox_mask.get("outfile_norm", [])
        valid = [f for f in norm_files[:n_scans] if f and os.path.exists(f)]
        if not valid:
            continue
        acc: NDArray[np.float64] | None = None
        ref_aff = None
        for fpath in valid:
            img = nib.load(fpath)
            arr = np.asarray(img.get_fdata(), dtype=np.float64)
            if acc is None:
                acc = arr
                ref_aff = img.affine
            else:
                acc = acc + arr
        if acc is not None:
            pct = 100.0 * acc / len(valid)
            mrs_struct["mask"][vox]["overlap"] = pct
            out_dir = os.path.dirname(os.path.abspath(valid[0]))
            overlap_path = os.path.join(out_dir, f"MRS_voxel_overlap_{vox}.nii")
            out_img = nib.Nifti1Image(pct.astype(np.float32), ref_aff)
            nib.save(out_img, overlap_path)
            mrs_struct["mask"][vox]["overlap_file"] = overlap_path
    return mrs_struct
