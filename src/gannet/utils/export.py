"""CSV, self-contained HTML report, and MAT export utilities for Gannet."""

from __future__ import annotations

import html
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import savemat

from gannet.config import MRSStruct
from gannet.plotting.paper_plot import paper_plot
from gannet.plotting.reports import (
    _get_pyplot,
    build_coreg_figure,
    build_fit_figure,
    build_load_figure,
    build_segment_figure,
    figure_to_base64_png,
)


def _extract_scalar(container: dict[str, Any], key: str, idx: int) -> Any:
    """Safely extract scalar value at ``idx`` from a list, array, or scalar.

    Parameters
    ------------
    container: dict[str, Any]
        Mapping containing metric arrays or lists.
    key: str
        Dictionary key to look up.
    idx: int
        0-based scan index.

    Returns
    --------
        Any:
            Python scalar value or ``float("nan")`` if unavailable.
    """
    val = container.get(key)
    if val is None:
        return float("nan")
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if isinstance(val, str):
        return val
    try:
        seq = list(val)
        if idx < len(seq):
            item = seq[idx]
            if isinstance(item, np.ndarray):
                return float(item.ravel()[0]) if item.size == 1 else item.tolist()
            if isinstance(item, np.number):
                return float(item)
            return item
    except TypeError:
        return val
    return float("nan")


def export_to_csv(
    mrs_struct: MRSStruct,
    caller: str = "fit",
    out_dir: str | None = None,
) -> str:
    """Export Gannet quantification and fit metrics to a CSV file (``ExportToCSV.m``).

    Parameters
    ------------
    mrs_struct: MRSStruct
        Processed ``MRSStruct`` from ``gannet_fit`` or ``gannet_quantify``.
    caller: str
        Originating module name: ``fit``, ``phantom``, ``segment``, or ``quantify``.
    out_dir: str | None
        Destination directory (defaults to current working directory).

    Returns
    --------
        str:
            Path to the written CSV file.
    """
    p = mrs_struct["p"]
    n_scans = int(p["numScans"])
    vox_names = p.get("Vox", ["vox1"])[: int(p["numVox"])]
    targets = list(p.get("target", ["GABA"]))
    metab_files = mrs_struct.get("metabfile", [f"scan_{i + 1}" for i in range(n_scans)])

    rows: list[dict[str, Any]] = []
    out_root = mrs_struct.get("out", {})

    for ii in range(n_scans):
        fname = (
            os.path.basename(metab_files[ii])
            if ii < len(metab_files)
            else f"scan_{ii + 1}"
        )
        for vox in vox_names:
            vox_out = out_root.get(vox, {})
            tissue = vox_out.get("tissue", {})
            cr_out = vox_out.get("Cr", {})
            cho_out = vox_out.get("Cho", {})
            water_out = vox_out.get("water", {})
            naa_out = vox_out.get("NAA", {})

            for targ in targets:
                t_out = vox_out.get(targ, {})
                row: dict[str, Any] = {
                    "Filename": fname,
                    "Voxel": vox,
                    "Metabolite": targ,
                    "LarmorFreq_MHz": _extract_scalar(p, "LarmorFreq", ii),
                    "TE_ms": _extract_scalar(p, "TE", ii),
                    "TR_ms": _extract_scalar(p, "TR", ii),
                    "Navg": _extract_scalar(p, "Navg", ii),
                    f"{targ}_Area": _extract_scalar(t_out, "Area", ii),
                    f"{targ}_FWHM_Hz": _extract_scalar(t_out, "FWHM", ii),
                    f"{targ}_SNR": _extract_scalar(t_out, "SNR", ii),
                    f"{targ}_FitError_pct": _extract_scalar(t_out, "FitError", ii),
                    f"{targ}_FitError_W_pct": _extract_scalar(t_out, "FitError_W", ii),
                    f"{targ}_FitError_Cr_pct": _extract_scalar(
                        t_out, "FitError_Cr", ii
                    ),
                    f"{targ}_ConcIU": _extract_scalar(t_out, "ConcIU", ii),
                    f"{targ}_ConcCr": _extract_scalar(t_out, "ConcCr", ii),
                    f"{targ}_ConcNAA": _extract_scalar(t_out, "ConcNAA", ii),
                    "Cr_Area": _extract_scalar(cr_out, "Area", ii),
                    "Cr_FWHM_Hz": _extract_scalar(cr_out, "FWHM", ii),
                    "Cho_Area": _extract_scalar(cho_out, "Area", ii),
                    "Water_Area": _extract_scalar(water_out, "Area", ii),
                    "Water_FWHM_Hz": _extract_scalar(water_out, "FWHM", ii),
                    "NAA_Area": _extract_scalar(naa_out, "Area", ii),
                }
                if tissue:
                    row["fGM"] = _extract_scalar(tissue, "fGM", ii)
                    row["fWM"] = _extract_scalar(tissue, "fWM", ii)
                    row["fCSF"] = _extract_scalar(tissue, "fCSF", ii)
                if caller.lower() == "quantify":
                    row[f"{targ}_ConcIU_CSFcorr"] = _extract_scalar(
                        t_out, "ConcIU_CSFcorr", ii
                    )
                    row[f"{targ}_ConcIU_TissCorr"] = _extract_scalar(
                        t_out, "ConcIU_TissCorr", ii
                    )
                    row[f"{targ}_ConcIU_AlphaTissCorr"] = _extract_scalar(
                        t_out, "ConcIU_AlphaTissCorr", ii
                    )
                    row[f"{targ}_ConcIU_AlphaTissCorr_GrpNorm"] = _extract_scalar(
                        t_out, "ConcIU_AlphaTissCorr_GrpNorm", ii
                    )
                if caller.lower() == "phantom":
                    row[f"{targ}_Conc_mM"] = _extract_scalar(t_out, "Conc", ii)
                rows.append(row)

    dest_dir = out_dir or os.getcwd()
    os.makedirs(dest_dir, exist_ok=True)
    caller_map = {
        "fit": "GannetFit_output.csv",
        "phantom": "GannetFitPhantom_output.csv",
        "segment": "GannetSegment_output.csv",
        "quantify": "GannetQuantify_output.csv",
    }
    csv_name = caller_map.get(caller.lower(), f"Gannet_{caller}_output.csv")
    csv_path = os.path.join(dest_dir, csv_name)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _fmt_num(val: Any, decimals: int = 3) -> str:
    """Format numeric scalar cleanly for HTML table cells.

    Parameters
    ------------
    val: Any
        Numeric scalar or fallback value.
    decimals: int
        Decimal precision for floats.

    Returns
    --------
        str:
            Formatted HTML-safe string.
    """
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return "—"
        return f"{val:.{decimals}f}"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, str):
        return html.escape(val)
    return html.escape(str(val))


def export_to_html(
    mrs_struct: MRSStruct,
    out_path: str | None = None,
    title: str = "Gannet MRS Analysis Report",
) -> str:
    """Write a self-contained HTML report with embedded plots and tables.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Processed ``MRSStruct`` at any stage after ``gannet_load``.
    out_path: str | None
        Destination ``.html`` file path (defaults to ``./Gannet_report.html``).
    title: str
        Page heading string displayed at the top of the report.

    Returns
    --------
        str:
            Path to the written HTML report file.
    """
    if out_path is None:
        out_path = os.path.join(os.getcwd(), "Gannet_report.html")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    p = mrs_struct["p"]
    n_scans = int(p["numScans"])
    vox_names = p.get("Vox", ["vox1"])[: int(p["numVox"])]
    targets = list(p.get("target", ["GABA"]))
    metab_files = mrs_struct.get("metabfile", [f"scan_{i + 1}" for i in range(n_scans)])
    ver = mrs_struct.get("info", {}).get("version", {})
    ver_str = str(ver.get("Gannet", "3.5.4")) if isinstance(ver, dict) else str(ver)

    summary_rows_html: list[str] = []
    scan_sections_html: list[str] = []
    out_root = mrs_struct.get("out", {})
    plt = _get_pyplot()

    for ii in range(n_scans):
        fname = (
            os.path.basename(metab_files[ii])
            if ii < len(metab_files)
            else f"scan_{ii + 1}"
        )
        for vox in vox_names:
            vox_out = out_root.get(vox, {})
            tissue = vox_out.get("tissue", {})
            for targ in targets:
                t_out = vox_out.get(targ, {})
                s_tiss = _fmt_num(_extract_scalar(t_out, "ConcIU_TissCorr", ii), 3)
                s_alpha = _fmt_num(
                    _extract_scalar(t_out, "ConcIU_AlphaTissCorr", ii), 3
                )
                summary_rows_html.append(
                    "<tr>"
                    f"<td>{html.escape(fname)}</td>"
                    f"<td>{html.escape(vox)}</td>"
                    f"<td>{html.escape(targ)}</td>"
                    f"<td>{_fmt_num(_extract_scalar(t_out, 'Area', ii), 4)}</td>"
                    f"<td>{_fmt_num(_extract_scalar(t_out, 'FWHM', ii), 2)}</td>"
                    f"<td>{_fmt_num(_extract_scalar(t_out, 'SNR', ii), 1)}</td>"
                    f"<td>{_fmt_num(_extract_scalar(t_out, 'FitError', ii), 2)}%</td>"
                    f"<td>{_fmt_num(_extract_scalar(t_out, 'ConcIU', ii), 3)}</td>"
                    f"<td>{s_tiss}</td>"
                    f"<td>{s_alpha}</td>"
                    f"<td>{_fmt_num(_extract_scalar(tissue, 'fGM', ii), 3)}</td>"
                    f"<td>{_fmt_num(_extract_scalar(tissue, 'fWM', ii), 3)}</td>"
                    f"<td>{_fmt_num(_extract_scalar(tissue, 'fCSF', ii), 3)}</td>"
                    "</tr>"
                )

            img_blocks: list[str] = []
            # Load figure
            if "spec" in mrs_struct and vox in mrs_struct["spec"]:
                fig_l = build_load_figure(mrs_struct, vox, ii)
                b64_l = figure_to_base64_png(fig_l)
                plt.close(fig_l)
                img_blocks.append(
                    '<div class="fig-card">'
                    f"<h4>Load &amp; Alignment ({html.escape(vox)})</h4>"
                    f'<img src="data:image/png;base64,{b64_l}" alt="Load QC"></div>'
                )
            # Fit figure
            if targets[0] in vox_out and vox_out[targets[0]].get("Area"):
                fig_f = build_fit_figure(mrs_struct, vox, ii)
                b64_f = figure_to_base64_png(fig_f)
                plt.close(fig_f)
                img_blocks.append(
                    f'<div class="fig-card"><h4>Spectral Fit ({html.escape(vox)})</h4>'
                    '<img src="data:image/png;base64,'
                    f'{b64_f}" alt="Spectral Fit"></div>'
                )
            # Segment or Coreg figure
            if tissue and tissue.get("fGM"):
                fig_s = build_segment_figure(mrs_struct, vox, ii)
                b64_s = figure_to_base64_png(fig_s)
                plt.close(fig_s)
                img_blocks.append(
                    f'<div class="fig-card"><h4>Segmentation ({html.escape(vox)})</h4>'
                    '<img src="data:image/png;base64,'
                    f'{b64_s}" alt="Segmentation"></div>'
                )
            elif (
                "mask" in mrs_struct
                and vox in mrs_struct["mask"]
                and mrs_struct["mask"][vox].get("img")
            ):
                fig_c = build_coreg_figure(mrs_struct, vox, ii)
                b64_c = figure_to_base64_png(fig_c)
                plt.close(fig_c)
                img_blocks.append(
                    '<div class="fig-card">'
                    f"<h4>Co-Registration ({html.escape(vox)})</h4>"
                    '<img src="data:image/png;base64,'
                    f'{b64_c}" alt="Coregistration"></div>'
                )

            scan_hdr = f"Scan {ii + 1}: {html.escape(fname)} ({html.escape(vox)})"
            scan_sections_html.append(
                f"<section class='scan-block'><h3>{scan_hdr}</h3>"
                f"<div class='fig-grid'>{''.join(img_blocks)}</div></section>"
            )

    # Group mean +/- SD spectrum figure if spectral data exist
    group_html = ""
    if (
        "freq" in mrs_struct.get("spec", {})
        and vox_names
        and vox_names[0] in mrs_struct.get("spec", {})
    ):
        fig_g = paper_plot(mrs_struct, vox=vox_names[0], target=targets[0])
        b64_g = figure_to_base64_png(fig_g)
        plt.close(fig_g)
        group_html = (
            "<section class='scan-block'><h2>Group Mean ± SD Edited Spectrum</h2>"
            '<div class="fig-card"><img src="data:image/png;base64,'
            f'{b64_g}" alt="Group Spectrum"></div>'
            "</section>"
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vendor_esc = html.escape(str(p.get("vendor", "Unknown")))
    targets_esc = html.escape(", ".join(targets))
    align_esc = html.escape(str(p.get("alignment", "RobustSpecReg")))
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 2rem; color: #1f2937; background: #f9fafb;
}}
h1 {{ color: #111827; margin-bottom: 0.25rem; }}
.meta {{ color: #6b7280; margin-bottom: 1.5rem; font-size: 0.9rem; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }}
.card {{
  background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
  padding: 0.9rem 1.2rem; min-width: 150px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.card .label {{ font-size: 0.78rem; color: #6b7280; text-transform: uppercase; }}
.card .val {{ font-size: 1.2rem; font-weight: 600; margin-top: 0.25rem; }}
table {{
  width: 100%; border-collapse: collapse; background: #ffffff;
  margin-bottom: 2rem; font-size: 0.88rem; border: 1px solid #e5e7eb;
}}
th, td {{
  border-bottom: 1px solid #e5e7eb; padding: 0.55rem 0.7rem; text-align: right;
}}
th:nth-child(-n+3), td:nth-child(-n+3) {{ text-align: left; }}
th {{ background: #f3f4f6; font-weight: 600; }}
.scan-block {{
  background: #ffffff; border: 1px solid #e5e7eb;
  border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;
}}
.fig-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  gap: 1rem;
}}
.fig-card img {{
  max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;
}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="meta">Generated {html.escape(timestamp)} &middot;
  Gannet Python {html.escape(ver_str)}</div>
<div class="cards">
  <div class="card"><div class="label">Scans</div>
    <div class="val">{n_scans}</div></div>
  <div class="card"><div class="label">Vendor</div>
    <div class="val">{vendor_esc}</div></div>
  <div class="card"><div class="label">Targets</div>
    <div class="val">{targets_esc}</div></div>
  <div class="card"><div class="label">Alignment</div>
    <div class="val">{align_esc}</div></div>
</div>
<h2>Summary Results</h2>
<table>
<thead>
<tr>
<th>File</th><th>Voxel</th><th>Metab</th><th>Area</th><th>FWHM (Hz)</th>
<th>SNR</th><th>FitError</th><th>ConcIU</th><th>ConcIU_TissCorr</th>
<th>ConcIU_AlphaTissCorr</th><th>fGM</th><th>fWM</th><th>fCSF</th>
</tr>
</thead>
<tbody>
{"".join(summary_rows_html)}
</tbody>
</table>
{group_html}
{"".join(scan_sections_html)}
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as fid:
        fid.write(html_doc)
    return out_path


def _to_matlab_compatible(obj: Any) -> Any:
    """Recursively convert ``MRSStruct`` / Python types to ``savemat`` structures.

    Parameters
    ------------
    obj: Any
        Python dict, list, numpy array, or scalar.

    Returns
    --------
        Any:
            MATLAB-serializable representation.
    """
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if v is not None:
                cleaned[str(k)] = _to_matlab_compatible(v)
        return cleaned
    if isinstance(obj, list):
        if all(isinstance(x, (int, float, np.number)) for x in obj):
            return np.asarray(obj)
        return np.asarray([_to_matlab_compatible(x) for x in obj], dtype=object)
    if obj is None:
        return []
    return obj


def save_mrs_mat(mrs_struct: MRSStruct, out_path: str) -> str:
    """Save ``MRSStruct`` to a MATLAB ``.mat`` archive using ``scipy.io.savemat``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` dictionary.
    out_path: str
        Destination ``.mat`` file path.

    Returns
    --------
        str:
            Path to the written ``.mat`` file.
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"MRS_struct": _to_matlab_compatible(dict(mrs_struct))}
    savemat(out_path, payload, do_compression=True)
    return out_path
