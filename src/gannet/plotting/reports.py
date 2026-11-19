"""Headless Matplotlib diagnostic figure generators and PDF/PNG renderers."""

from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Any

import numpy as np

from gannet.config import MRSStruct


def _get_pyplot() -> Any:
    """Import ``matplotlib.pyplot`` lazily with non-interactive ``Agg`` backend.

    Parameters
    ------------
    None

    Returns
    --------
        Any:
            Configured ``matplotlib.pyplot`` module.
    """
    if "MPLCONFIGDIR" not in os.environ:
        default_dir = os.path.expanduser("~/.matplotlib")
        writable = False
        try:
            os.makedirs(default_dir, exist_ok=True)
            probe = os.path.join(default_dir, ".probe_write")
            with open(probe, "wb"):
                pass
            os.remove(probe)
            writable = True
        except OSError:
            writable = False
        if not writable:
            cache_dir = os.path.join(tempfile.gettempdir(), "gannet_mplconfig")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = cache_dir

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    return plt


def figure_to_base64_png(fig: Any, dpi: int = 150) -> str:
    """Render a Matplotlib ``Figure`` to a base64-encoded PNG data URI string.

    Parameters
    ------------
    fig: Any
        Matplotlib figure instance to render.
    dpi: int
        Output resolution in dots per inch.

    Returns
    --------
        str:
            Base64-encoded PNG string ready for ``data:image/png;base64,...``.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return encoded


def plot_pre_post_align(
    mrs_struct: MRSStruct,
    vox: str,
    ii: int,
    ax: Any | None = None,
) -> Any:
    """Plot pre- vs. post-alignment difference spectra (``PlotPrePostAlign.m``).

    Parameters
    ------------
    mrs_struct: MRSStruct
        Loaded ``MRSStruct`` containing ``spec.freq`` and ``spec.<vox>.<target>``.
    vox: str
        Voxel name (e.g., ``vox1``).
    ii: int
        0-based scan index.
    ax: Any | None
        Optional Matplotlib Axes to draw into; created if ``None``.

    Returns
    --------
        Any:
            Matplotlib Axes containing the plot.
    """
    if ax is None:
        plt = _get_pyplot()
        _, ax = plt.subplots(figsize=(6, 4))

    freq = np.asarray(mrs_struct["spec"]["freq"], dtype=np.float64)
    targets = mrs_struct["p"].get("target", ["GABA"])
    target = targets[0] if targets else "GABA"
    spec_vox = mrs_struct["spec"].get(vox, {}).get(target, {})

    diff = np.asarray(spec_vox.get("diff", []), dtype=np.complex128)
    diff_noalign = np.asarray(spec_vox.get("diff_noalign", []), dtype=np.complex128)
    if diff.ndim == 2:
        diff_1d = np.real(diff[ii]) if ii < diff.shape[0] else np.real(diff[0])
    else:
        diff_1d = np.real(diff)
    if diff_noalign.ndim == 2:
        noalign_1d = (
            np.real(diff_noalign[ii])
            if ii < diff_noalign.shape[0]
            else np.real(diff_noalign[0])
        )
    else:
        noalign_1d = np.real(diff_noalign)

    if freq.size == noalign_1d.size and noalign_1d.size > 0:
        ax.plot(freq, noalign_1d, color="#d62728", lw=0.8, label="Pre-align")
    if freq.size == diff_1d.size and diff_1d.size > 0:
        ax.plot(freq, diff_1d, color="#1f77b4", lw=1.0, label="Post-align")

    ax.set_xlim(4.2, 0.2)
    ax.set_xlabel("Chemical Shift (ppm)")
    ax.set_title(f"{target} Pre/Post Alignment")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_pre_post_align2(
    mrs_struct: MRSStruct,
    vox: str,
    ii: int,
    ax: Any | None = None,
) -> Any:
    """Plot stacked pre- vs. post-alignment spectra across multiple targets.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Loaded ``MRSStruct``.
    vox: str
        Voxel name (e.g., ``vox1``).
    ii: int
        0-based scan index.
    ax: Any | None
        Optional Matplotlib Axes.

    Returns
    --------
        Any:
            Matplotlib Axes containing the multi-target plot.
    """
    if ax is None:
        plt = _get_pyplot()
        _, ax = plt.subplots(figsize=(6, 4))

    freq = np.asarray(mrs_struct["spec"]["freq"], dtype=np.float64)
    targets = mrs_struct["p"].get("target", ["GABA"])
    offset = 0.0
    for targ in targets:
        spec_vox = mrs_struct["spec"].get(vox, {}).get(targ, {})
        diff = np.asarray(spec_vox.get("diff", []), dtype=np.complex128)
        diff_no = np.asarray(spec_vox.get("diff_noalign", []), dtype=np.complex128)
        if diff.size == 0 or freq.size == 0:
            continue
        d_post = np.real(diff[ii] if diff.ndim == 2 else diff)
        d_pre = (
            np.real(diff_no[ii] if diff_no.ndim == 2 else diff_no)
            if diff_no.size > 0
            else d_post
        )
        scale = max(float(np.ptp(d_post)), 1e-12)
        ax.plot(freq, d_pre + offset, color="#d62728", lw=0.7, alpha=0.8)
        ax.plot(freq, d_post + offset, color="#1f77b4", lw=0.9, label=targ)
        offset += 1.2 * scale

    ax.set_xlim(4.2, 0.2)
    ax.set_xlabel("Chemical Shift (ppm)")
    ax.set_title("Edited Subspectra Comparison")
    ax.legend(frameon=False, fontsize=8)
    return ax


def build_load_figure(mrs_struct: MRSStruct, vox: str, ii: int) -> Any:
    """Build the GannetLoad diagnostic QC figure for one scan.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Processed ``MRSStruct`` from ``gannet_load``.
    vox: str
        Voxel name (e.g., ``vox1``).
    ii: int
        0-based scan index.

    Returns
    --------
        Any:
            Matplotlib figure containing edited spectra, frequency drift, and summary.
    """
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    fig.patch.set_facecolor("white")

    plot_pre_post_align(mrs_struct, vox, ii, ax=axes[0, 0])

    ax_off = axes[0, 1]
    freq = np.asarray(mrs_struct["spec"]["freq"], dtype=np.float64)
    targets_l = mrs_struct["p"].get("target", ["GABA"])
    first_t = targets_l[0] if targets_l else "GABA"
    vox_spec = mrs_struct["spec"].get(vox, {})
    off_src = vox_spec.get(first_t, {}).get("off", vox_spec.get("OFF", []))
    off = np.asarray(off_src, dtype=np.complex128)
    if off.size > 0 and freq.size > 0:
        off_1d = np.real(off[ii] if off.ndim == 2 else off)
        ax_off.plot(freq, off_1d, color="#2ca02c", lw=0.9, label="OFF")
        ax_off.set_xlim(4.2, 0.2)
        ax_off.set_title("OFF Spectrum")
        ax_off.set_xlabel("Chemical Shift (ppm)")
    else:
        ax_off.axis("off")

    ax_f0 = axes[1, 0]
    f0_list = mrs_struct["spec"].get("F0freq", [])
    if ii < len(f0_list) and f0_list[ii] is not None:
        f0_arr = np.asarray(f0_list[ii], dtype=np.float64).ravel()
        ax_f0.plot(np.arange(1, f0_arr.size + 1), f0_arr, color="#1f77b4", lw=1.0)
        ax_f0.set_xlabel("Transient")
        ax_f0.set_ylabel("F0 (ppm)")
        ax_f0.set_title("Frequency Drift")
    else:
        ax_f0.text(0.5, 0.5, "No drift curve available", ha="center", va="center")
        ax_f0.axis("off")

    ax_txt = axes[1, 1]
    ax_txt.axis("off")
    p = mrs_struct["p"]
    metab_files = mrs_struct.get("metabfile", [""])
    fname = (
        os.path.basename(metab_files[ii]) if ii < len(metab_files) else f"Scan {ii + 1}"
    )
    lines = [
        f"File: {fname}",
        f"Vendor: {p.get('vendor', 'Unknown')}",
        f"LarmorFreq: {p['LarmorFreq'][ii]:.2f} MHz",
        f"Points: {int(p['npoints'][ii])} | Navg: {int(p['Navg'][ii])}",
        f"SW: {p['sw'][ii]:.1f} Hz | TE/TR: {p['TE'][ii]:.1f}/{p['TR'][ii]:.1f} ms",
        f"Alignment: {p.get('alignment', 'RobustSpecReg')}",
    ]
    ax_txt.text(
        0.05,
        0.9,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )
    fig.tight_layout()
    return fig


def build_fit_figure(mrs_struct: MRSStruct, vox: str, ii: int) -> Any:
    """Build the GannetFit diagnostic figure showing spectral models and residuals.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Processed ``MRSStruct`` from ``gannet_fit`` or ``gannet_fit_phantom``.
    vox: str
        Voxel name.
    ii: int
        0-based scan index.

    Returns
    --------
        Any:
            Matplotlib figure displaying primary edited fit, reference fits, and stats.
    """
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    fig.patch.set_facecolor("white")

    out_vox = mrs_struct.get("out", {}).get(vox, {})
    spec_vox = mrs_struct.get("spec", {}).get(vox, {})
    targets = mrs_struct["p"].get("target", ["GABA"])
    main_t = targets[0] if targets else "GABA"
    t_out = out_vox.get(main_t, {})
    t_spec = spec_vox.get(main_t, {})
    freq = np.asarray(mrs_struct["spec"].get("freq", []), dtype=np.float64)

    # Panel 1: primary edited target fit
    ax0 = axes[0, 0]
    diff = np.asarray(t_spec.get("diff", []), dtype=np.complex128)
    if diff.size > 0 and freq.size > 0:
        d1d = np.real(diff[ii] if diff.ndim == 2 else diff)
        ax0.plot(freq, d1d, color="#222222", lw=0.8, label="Data")
    fit_x = np.asarray(t_out.get("fit_freq", []), dtype=np.float64)
    fit_y = np.asarray(t_out.get("fit_spec", []), dtype=np.float64)
    if fit_x.size > 0 and fit_y.size > 0:
        ax0.plot(fit_x, fit_y, color="#d62728", lw=1.3, label="Model")
    ax0.set_xlim(4.2, 0.2)
    ax0.set_title(f"{main_t} Edited Fit")
    ax0.set_xlabel("Chemical Shift (ppm)")
    ax0.legend(frameon=False, fontsize=8)

    # Panel 2: water fit
    ax1 = axes[0, 1]
    w_out = out_vox.get("water", {})
    w_freq = np.asarray(w_out.get("fit_freq", []), dtype=np.float64)
    w_data = np.asarray(w_out.get("fit_data", []), dtype=np.float64)
    w_mod = np.asarray(w_out.get("fit_spec", []), dtype=np.float64)
    if w_freq.size > 0 and w_data.size > 0:
        ax1.plot(w_freq, w_data, color="#222222", lw=0.8, label="Water")
        if w_mod.size == w_freq.size:
            ax1.plot(w_freq, w_mod, color="#1f77b4", lw=1.2, label="Fit")
        ax1.invert_xaxis()
        ax1.set_title("Water Reference Fit")
        ax1.set_xlabel("Chemical Shift (ppm)")
        ax1.legend(frameon=False, fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No water reference", ha="center", va="center")
        ax1.axis("off")

    # Panel 3: Cr / Cho / NAA reference
    ax2 = axes[1, 0]
    cr_out = out_vox.get("Cr", {})
    cr_freq = np.asarray(cr_out.get("fit_freq", []), dtype=np.float64)
    cr_data = np.asarray(cr_out.get("fit_data", []), dtype=np.float64)
    cr_mod = np.asarray(cr_out.get("fit_spec", []), dtype=np.float64)
    if cr_freq.size > 0 and cr_data.size > 0:
        ax2.plot(cr_freq, cr_data, color="#222222", lw=0.8, label="Cr Data")
        if cr_mod.size == cr_freq.size:
            ax2.plot(cr_freq, cr_mod, color="#2ca02c", lw=1.2, label="Cr Fit")
        ax2.invert_xaxis()
        ax2.set_title("Cr Reference Fit")
        ax2.set_xlabel("Chemical Shift (ppm)")
        ax2.legend(frameon=False, fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No Cr reference fit", ha="center", va="center")
        ax2.axis("off")

    # Panel 4: metrics summary
    ax3 = axes[1, 1]
    ax3.axis("off")
    lines: list[str] = []
    for targ in targets:
        to = out_vox.get(targ, {})
        area = _get_idx(to.get("Area"), ii)
        fwhm = _get_idx(to.get("FWHM"), ii)
        snr = _get_idx(to.get("SNR"), ii)
        err = _get_idx(to.get("FitError"), ii)
        conc = _get_idx(to.get("ConcIU"), ii)
        lines.append(
            f"{targ}:\n"
            f"  Area={area:.3g} | FWHM={fwhm:.1f}Hz\n"
            f"  SNR={snr:.1f} | Err={err:.1f}% | IU={conc:.2f}"
        )
    ax3.text(
        0.05,
        0.92,
        "\n\n".join(lines),
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )
    fig.set_layout_engine("constrained")
    return fig


def _get_idx(seq: Any, ii: int, default: float = float("nan")) -> float:
    """Safely extract scalar float at index ``ii`` from list/ndarray or scalar.

    Parameters
    ------------
    seq: Any
        List, array, or scalar.
    ii: int
        0-based index.
    default: float
        Fallback value when missing or non-numeric.

    Returns
    --------
        float:
            Extracted floating-point scalar.
    """
    if seq is None:
        return default
    arr = np.asarray(seq, dtype=np.float64).ravel()
    if arr.size == 0:
        return default
    if ii < arr.size:
        return float(arr[ii])
    return float(arr[0])


def build_coreg_figure(mrs_struct: MRSStruct, vox: str, ii: int) -> Any:
    """Build the GannetCoRegister 3-plane MRI + MRS voxel overlay figure.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` populated by ``gannet_co_register``.
    vox: str
        Voxel name.
    ii: int
        0-based scan index.

    Returns
    --------
        Any:
            Matplotlib figure displaying the sagittal/coronal/axial voxel overlay.
    """
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("white")
    imgs = mrs_struct.get("mask", {}).get(vox, {}).get("img", [])
    overlay = imgs[ii] if ii < len(imgs) else None
    if overlay is not None:
        ax.imshow(overlay)
        ax.axis("off")
        t1s = mrs_struct["mask"][vox].get("T1image", [""])
        t1_name = os.path.basename(t1s[ii]) if ii < len(t1s) else ""
        ax.set_title(f"Co-registration ({vox}): {t1_name}")
    else:
        ax.text(
            0.5, 0.5, "No co-registration overlay available", ha="center", va="center"
        )
        ax.axis("off")
    fig.tight_layout()
    return fig


def build_segment_figure(mrs_struct: MRSStruct, vox: str, ii: int) -> Any:
    """Build the GannetSegment figure showing voxel overlay and GM/WM/CSF fractions.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` populated by ``gannet_segment``.
    vox: str
        Voxel name.
    ii: int
        0-based scan index.

    Returns
    --------
        Any:
            Matplotlib figure with co-registration montage and tissue composition bars.
    """
    plt = _get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.patch.set_facecolor("white")

    imgs = mrs_struct.get("mask", {}).get(vox, {}).get("img", [])
    overlay = imgs[ii] if ii < len(imgs) else None
    if overlay is not None:
        axes[0].imshow(overlay)
        axes[0].set_title(f"Voxel Placement ({vox})")
    axes[0].axis("off")

    f_gm = _get_idx(mrs_struct["out"][vox]["tissue"]["fGM"], ii, 0.0)
    f_wm = _get_idx(mrs_struct["out"][vox]["tissue"]["fWM"], ii, 0.0)
    f_csf = _get_idx(mrs_struct["out"][vox]["tissue"]["fCSF"], ii, 0.0)
    axes[1].bar(
        ["GM", "WM", "CSF"],
        [f_gm, f_wm, f_csf],
        color=["#e377c2", "#1f77b4", "#17becf"],
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Volume Fraction")
    axes[1].set_title(
        f"Tissue Fractions (GM={f_gm:.2f}, WM={f_wm:.2f}, CSF={f_csf:.2f})"
    )
    fig.tight_layout()
    return fig


def save_pdf(
    fig: Any,
    out_dir: str,
    prefix: str,
    append: bool = False,
) -> str:
    """Save a Matplotlib figure as a PDF (or multi-page PDF via ``PdfPages``).

    Parameters
    ------------
    fig: Any
        Matplotlib figure to save.
    out_dir: str
        Destination output directory.
    prefix: str
        Output filename stem (``.pdf`` appended if missing).
    append: bool
        Reserved flag controlling multi-page document creation helper semantics.

    Returns
    --------
        str:
            Path to the written PDF file.
    """
    _get_pyplot()
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(out_dir, exist_ok=True)
    name = prefix if prefix.lower().endswith(".pdf") else f"{prefix}.pdf"
    out_path = os.path.join(out_dir, name)
    if append:
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
    else:
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
    return out_path
