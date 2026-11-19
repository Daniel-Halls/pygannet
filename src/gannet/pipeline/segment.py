"""GannetSegment module: T1 segmentation, tissue fractions, QA, and CSF correction."""

from __future__ import annotations

import os
from typing import Any

import nibabel as nib
import numpy as np

from gannet.config import MRSStruct
from gannet.pipeline.fit import _ensure_list_len
from gannet.segment.fsl import call_fsl_segmentation
from gannet.segment.segmentation import (
    compute_mriqc_metrics,
    normalize_voxel_mask,
    voxel_mask_overlap,
)
from gannet.utils.export import export_to_csv, export_to_html, save_mrs_mat


def gannet_segment(
    mrs_struct: MRSStruct,
    save_csv: bool = False,
    save_html: bool = True,
    out_dir: str | None = None,
) -> MRSStruct:
    """Segment T1 volumes with FSL (BET/FAST), compute tissue fractions and MRIQC QA.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` populated by ``gannet_co_register``.
    save_csv: bool
        Export segmentation metrics to CSV when ``True``.
    save_html: bool
        Generate standalone HTML report with segmentation panels when ``True``.
    out_dir: str | None
        Optional directory for saved CSV/HTML/MAT artifacts.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with ``out.<vox>.tissue`` (``fGM``, ``fWM``,
            ``fCSF``), ``ConcIU_CSFcorr``, and MRIQC ``QA``.
    """
    p = mrs_struct["p"]
    n_scans = int(p["numScans"])
    vox_names = p.get("Vox", ["vox1"])[: int(p["numVox"])]
    targets = list(p.get("target", ["GABA"]))
    backend = str(p.get("registration_backend", "fsl"))
    bet_frac = float(p.get("bet_frac", 0.5))
    use_fnirt = bool(p.get("use_fnirt", False))

    mrs_struct["out"].setdefault("QA", [None] * n_scans)

    for ii in range(n_scans):
        mrs_struct["ii"] = ii
        first_vox = vox_names[0]
        t1_list = mrs_struct.get("mask", {}).get(first_vox, {}).get("T1image", [])
        if ii >= len(t1_list) or not t1_list[ii]:
            continue
        t1_path = t1_list[ii]
        t1_img = nib.load(t1_path)
        t1_vol = np.asarray(t1_img.get_fdata(), dtype=np.float64)
        if t1_vol.ndim > 3:
            t1_vol = t1_vol[..., 0]

        seg_maps = call_fsl_segmentation(t1_path, bet_frac=bet_frac, backend=backend)
        c1 = np.asarray(nib.load(seg_maps["c1"]).get_fdata(), dtype=np.float64)
        c2 = np.asarray(nib.load(seg_maps["c2"]).get_fdata(), dtype=np.float64)
        c3 = np.asarray(nib.load(seg_maps["c3"]).get_fdata(), dtype=np.float64)
        c6 = (
            np.asarray(nib.load(seg_maps["c6"]).get_fdata(), dtype=np.float64)
            if "c6" in seg_maps and os.path.exists(seg_maps["c6"])
            else None
        )

        qa = compute_mriqc_metrics(t1_vol, c1, c2, c3, c6, prob_threshold=0.9)
        mrs_struct["out"]["QA"][ii] = qa

        for vox in vox_names:
            mask_files = mrs_struct.get("mask", {}).get(vox, {}).get("outfile", [])
            if ii >= len(mask_files) or not mask_files[ii]:
                continue
            mask_path = mask_files[ii]
            vox_mask = (
                np.asarray(nib.load(mask_path).get_fdata(), dtype=np.float64) > 0.5
            )
            n_vox_pts = int(np.sum(vox_mask))

            gm_vals = c1[vox_mask] if n_vox_pts > 0 else np.array([0.0])
            wm_vals = c2[vox_mask] if n_vox_pts > 0 else np.array([0.0])
            csf_vals = c3[vox_mask] if n_vox_pts > 0 else np.array([0.0])
            gm_sum = float(np.sum(gm_vals))
            wm_sum = float(np.sum(wm_vals))
            csf_sum = float(np.sum(csf_vals))
            tot = gm_sum + wm_sum + csf_sum
            if tot > 0:
                f_gm = gm_sum / tot
                f_wm = wm_sum / tot
                f_csf = csf_sum / tot
            else:
                f_gm, f_wm, f_csf = (0.0, 0.0, 0.0)

            mrs_struct.ensure_substruct("out", vox, "tissue")
            tissue = mrs_struct["out"][vox]["tissue"]
            for key, val in (("fGM", f_gm), ("fWM", f_wm), ("fCSF", f_csf)):
                _ensure_list_len(tissue, key, ii, default=float("nan"))
                tissue[key][ii] = val

            # CSF-corrected water-scaled concentration
            for targ in targets:
                t_out: dict[str, Any] = (
                    mrs_struct.get("out", {}).get(vox, {}).get(targ, {})
                )
                conc_list = t_out.get("ConcIU", [])
                if ii < len(conc_list) and not np.isnan(conc_list[ii]):
                    _ensure_list_len(t_out, "ConcIU_CSFcorr", ii, default=float("nan"))
                    denom = max(1.0 - f_csf, 1e-6)
                    t_out["ConcIU_CSFcorr"][ii] = float(conc_list[ii]) / denom

            if bool(p.get("normalize_voxel_mask", True)):
                w_mask = normalize_voxel_mask(
                    mask_path,
                    t1_path=t1_path,
                    brain_path=seg_maps.get("brain"),
                    use_fnirt=use_fnirt,
                    backend=backend,
                )
                mrs_struct["mask"][vox].setdefault("outfile_norm", [])
                while len(mrs_struct["mask"][vox]["outfile_norm"]) <= ii:
                    mrs_struct["mask"][vox]["outfile_norm"].append("")
                mrs_struct["mask"][vox]["outfile_norm"][ii] = w_mask

    if n_scans >= 1 and bool(p.get("normalize_voxel_mask", True)):
        mrs_struct = voxel_mask_overlap(mrs_struct)

    dest_dir = out_dir or os.getcwd()
    if save_csv or bool(p.get("csv", False)):
        export_to_csv(mrs_struct, caller="segment", out_dir=dest_dir)
    if bool(p.get("mat", False)):
        save_mrs_mat(mrs_struct, os.path.join(dest_dir, "MRS_struct.mat"))
    if save_html:
        export_to_html(
            mrs_struct,
            out_path=os.path.join(dest_dir, "GannetSegment_report.html"),
            title="GannetSegment Tissue Segmentation Report",
        )
    return mrs_struct
