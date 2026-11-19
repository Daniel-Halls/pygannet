"""MRS signal processing, filtering, alignment, and averaging modules."""

from gannet.processing.alignment import (
    align_using_h2o,
    align_using_peak,
    freq_phase_shift_nest,
    robust_spectral_registration,
    spec_reg,
    spectral_registration,
    spectral_registration_hermes,
)
from gannet.processing.averaging import signal_averaging, wacfm
from gannet.processing.corrections import (
    eddy_current_correction,
    phase_correction,
)
from gannet.processing.hsvd import water_removal_hsvd
from gannet.processing.noise import calc_noise
from gannet.processing.on_off import specify_on_off_order
from gannet.processing.signal_filter import (
    baseline_modeling,
    cwt2,
    signal_filter,
    whittaker,
)
from gannet.processing.subspectra_align import (
    align_sub_spectra,
    align_sub_spectra_pre_align_ref,
)

__all__ = [
    "align_sub_spectra",
    "align_sub_spectra_pre_align_ref",
    "align_using_h2o",
    "align_using_peak",
    "baseline_modeling",
    "calc_noise",
    "cwt2",
    "eddy_current_correction",
    "freq_phase_shift_nest",
    "phase_correction",
    "robust_spectral_registration",
    "signal_averaging",
    "signal_filter",
    "spec_reg",
    "specify_on_off_order",
    "spectral_registration",
    "spectral_registration_hermes",
    "wacfm",
    "water_removal_hsvd",
    "whittaker",
]
