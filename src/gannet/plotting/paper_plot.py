"""Publication-style group mean +/- SD spectrum plotting (``PaperPlot.m``)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct
from gannet.plotting.reports import _get_pyplot


def paper_plot(
    mrs_struct: MRSStruct,
    vox: str | None = None,
    target: str | None = None,
    signal: str = "diff",
    ppm_range: tuple[float, float] = (0.5, 4.2),
    save_path: str | None = None,
) -> Any:
    """Generate a publication-ready mean +/- SD spectrum figure across scans.

    Ports ``PaperPlot.m`` without launching interactive MATLAB UI windows.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Processed ``MRSStruct`` containing group spectra in ``spec.<vox>.<target>``.
    vox: str | None
        Voxel key (defaults to first entry in ``mrs_struct.p.Vox``).
    target: str | None
        Target metabolite key (defaults to first entry in ``mrs_struct.p.target``).
    signal: str
        Spectral signal component: ``diff``, ``diff_noalign``, ``off``, ``on``,
        or ``off_off``.
    ppm_range: tuple[float, float]
        Chemical shift display limits ``(ppm_min, ppm_max)``.
    save_path: str | None
        Optional image file path (``.png``, ``.pdf``, ``.svg``) to write.

    Returns
    --------
        Any:
            Matplotlib ``Figure`` containing mean spectrum and +/-1 SD ribbon.
    """
    plt = _get_pyplot()
    p = mrs_struct["p"]
    vox_name = vox or (p.get("Vox", ["vox1"])[0])
    targ_name = target or (p.get("target", ["GABA"])[0])

    freq = np.asarray(mrs_struct.get("spec", {}).get("freq", []), dtype=np.float64)
    targ_spec = mrs_struct["spec"].get(vox_name, {}).get(targ_name, {})
    if signal.lower() in ("off", "on", "sum"):
        arr: Any = targ_spec.get(
            signal.lower(),
            mrs_struct["spec"].get(vox_name, {}).get(signal.upper(), []),
        )
    else:
        arr = targ_spec.get(signal, [])

    spec_arr = np.real(np.asarray(arr, dtype=np.complex128))
    if spec_arr.ndim == 1:
        spec_arr = spec_arr[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor("white")

    if spec_arr.size > 0 and freq.size == spec_arr.shape[1]:
        mean_spec: NDArray[np.float64] = np.mean(spec_arr, axis=0)
        sd_spec: NDArray[np.float64] = np.std(spec_arr, axis=0)
        ax.fill_between(
            freq,
            mean_spec - sd_spec,
            mean_spec + sd_spec,
            color="#4c72b0",
            alpha=0.25,
            label="±1 SD",
        )
        ax.plot(
            freq,
            mean_spec,
            color="#1f4e78",
            lw=1.4,
            label=f"Mean {targ_name} ({signal}, N={spec_arr.shape[0]})",
        )
        ax.set_xlim(max(ppm_range), min(ppm_range))
        ax.set_xlabel("Chemical Shift (ppm)", fontsize=11)
        ax.set_ylabel("Signal Intensity (a.u.)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            f"No spectra available for {vox_name}/{targ_name}/{signal}",
            ha="center",
            va="center",
        )
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        out_dir = os.path.dirname(os.path.abspath(save_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
