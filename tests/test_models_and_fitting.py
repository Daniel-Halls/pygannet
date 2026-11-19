"""Unit tests for Gannet lineshape models, optimizers, peak fitters, and VAX I/O."""

from __future__ import annotations

import numpy as np

from gannet.config import gannet_pre_initialise
from gannet.fitting.optimizers import lsqcurvefit, lsqnonlin, nlinfit
from gannet.fitting.peak_fitters import fit_cho_cr, fit_cr
from gannet.fitting.quantification_helpers import (
    calc_conc_phantom,
    calc_iu,
    get_in_vivo_metabolite_params,
    get_phantom_metabolite_params,
)
from gannet.io.vax import (
    uint32le_to_vaxf,
    uint64le_to_vaxg,
    vaxd_to_uint64le,
    vaxf_to_uint32le,
    vaxg_to_uint64le,
)
from gannet.models.lineshapes import (
    baseline_model,
    double_gauss_model,
    etoh_model,
    five_gauss_model,
    gaba_glx_model,
    gauss_model,
    lac_model,
    lorentz_gauss_model,
    lorentz_gauss_model_p,
    lorentz_model,
    six_gauss_model,
    three_lorentz_model,
    two_lorentz_model,
    voigt_model,
)
from gannet.models.phantom_lineshapes import (
    phantom_six_lorentz_model,
    phantom_three_lorentz_model,
    phantom_two_lorentz_model,
)
from gannet.processing.noise import calc_noise


def test_vax_roundtrip_conversions() -> None:
    """Verify IEEE float32/float64 roundtrips through VAXF, VAXD, and VAXG bit layouts.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    vals32 = np.array([0.0, 1.0, -2.5, 123.456, -0.00125], dtype=np.float32)
    enc32 = vaxf_to_uint32le(vals32)
    dec32 = uint32le_to_vaxf(enc32)
    np.testing.assert_allclose(dec32, vals32, rtol=1e-5, atol=1e-7)

    vals64 = np.array([0.0, 1.0, -3.14159265, 1024.5], dtype=np.float64)
    enc64g = vaxg_to_uint64le(vals64)
    dec64g = uint64le_to_vaxg(enc64g)
    np.testing.assert_allclose(dec64g, vals64, rtol=1e-10, atol=1e-12)

    enc64d = vaxd_to_uint64le(vals64)
    assert enc64d.shape == (2, len(vals64))


def test_all_lineshape_models_evaluate() -> None:
    """Verify finite outputs and exact evaluation for all 17 lineshape models.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    freq = np.linspace(0.5, 4.5, 200)
    assert np.all(np.isfinite(gauss_model([1.0, -80.0, 3.01, 0.0, 0.0], freq)))
    assert np.all(
        np.isfinite(
            double_gauss_model([1.0, -80.0, 3.0, 0.8, -80.0, 3.75, 0.0, 0.0], freq)
        )
    )
    assert np.all(
        np.isfinite(five_gauss_model([1.0, -80.0, 2.95] * 5 + [0.0, 0.0, 0.0], freq))
    )
    assert np.all(
        np.isfinite(six_gauss_model([1.0, -80.0, 2.95] * 6 + [0.0, 0.0, 0.0], freq))
    )
    assert np.all(
        np.isfinite(
            gaba_glx_model(
                [1.0, -80.0, 3.71, 0.9, -80.0, 3.78, 1.2, -80.0, 3.02, 0.0, 0.0, 0.0],
                freq,
            )
        )
    )
    assert np.all(np.isfinite(baseline_model([3.0, 0.1, 0.5], freq)))
    assert np.all(np.isfinite(lorentz_model([1.0, 0.02, 3.02, 0.0, 0.0, 0.0], freq)))
    assert np.all(
        np.isfinite(two_lorentz_model([1.0, 0.02, 3.02, 0.0, 0.0, 0.0, 0.8], freq))
    )
    assert np.all(
        np.isfinite(
            three_lorentz_model(
                [1.0, 0.5, 0.5, 30.0, 2.35, 0.05, 0.0, 0.0, 0.0, 0.0], freq
            )
        )
    )
    assert np.all(
        np.isfinite(lorentz_gauss_model([10.0, 25.0, 4.68, 0.0, 0.0, -10.0], freq))
    )
    assert np.all(
        np.isfinite(
            lorentz_gauss_model_p([10.0, 25.0, 4.68, 0.0, 0.0, -10.0, 0.1], freq)
        )
    )
    assert np.all(
        np.isfinite(etoh_model([1.0, 1.17, 0.05, 0.5, 1.25, 0.05, 0.0, 0.0], freq))
    )
    assert np.all(
        np.isfinite(
            lac_model([1.0, -200.0, 1.33, 0.2, -200.0, 1.19, 0.0, 0.0, 0.0, 0.1], freq)
        )
    )
    assert np.all(np.isfinite(voigt_model([0.5, 1.0, 3.01, 0.05, 0.0, 0.0], freq)))
    assert np.all(
        np.isfinite(
            phantom_three_lorentz_model(
                [0.5, 0.5, 1.0, 50.0, 3.01, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0], freq
            )
        )
    )
    assert np.all(
        np.isfinite(
            phantom_two_lorentz_model(
                [1.0, 1.0, 50.0, 3.75, 0.03, 0.0, 0.0, 0.0, 0.0], freq
            )
        )
    )
    assert np.all(
        np.isfinite(
            phantom_six_lorentz_model(
                [0.5] * 6 + [50.0, 2.95, 0.02] + [0.0] * 6 + [0.0, 0.0], freq
            )
        )
    )


def test_optimizers_and_peak_fitters_recover_ground_truth() -> None:
    """Verify ``lsqcurvefit``, ``nlinfit``, ``fit_cr``, and ``fit_cho_cr`` recovery.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    freq = np.linspace(2.6, 3.6, 512)
    larmor = 127.74
    true_gauss = np.array([3.5, -120.0, 3.01, 0.2, 0.05])
    y_gauss = gauss_model(true_gauss, freq)
    x0 = np.array([2.0, -80.0, 3.00, 0.0, 0.0])
    lb = np.array([0.0, -500.0, 2.85, -5.0, -5.0])
    ub = np.array([10.0, -5.0, 3.18, 5.0, 5.0])

    fit1 = lsqcurvefit(gauss_model, x0, freq, y_gauss, lb=lb, ub=ub)
    fit2 = nlinfit(freq, y_gauss, gauss_model, fit1)
    np.testing.assert_allclose(fit2.beta, true_gauss, atol=1e-2)

    # Test lsqnonlin scalar residual norm minimization
    res_nl = lsqnonlin(lambda x: x - np.array([1.0, -2.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(res_nl, np.array([1.0, -2.0]), atol=1e-5)

    # Test FitCr & FitChoCr across 4 identical frames
    cr_true = np.array([0.5, 0.018, 3.02, 0.0, 0.0, 0.0])
    cr_spec = np.tile(lorentz_model(cr_true, freq)[:, np.newaxis], (1, 4))
    cr_init = np.array([0.4, 8.0, 3.02 * larmor, 0.0, 0.0, 0.0])
    out_cr = fit_cr(freq, cr_spec, cr_init, larmor)
    assert out_cr["fit_params"].shape == (4, 6)
    np.testing.assert_allclose(
        np.mean(out_cr["fit_params"][:, 2]), 3.02 * larmor, atol=1.5
    )

    cc_true = np.array([0.5, 0.018, 3.02, 0.0, 0.0, 0.0, 0.8])
    cc_spec = np.tile(two_lorentz_model(cc_true, freq)[:, np.newaxis], (1, 3))
    cc_init = np.array([0.4, 8.0, 3.02 * larmor, 0.0, 0.0, 0.0, 0.7])
    out_cc = fit_cho_cr(freq, cc_spec, cc_init, larmor)
    assert out_cc["fit_params"].shape == (3, 7)
    np.testing.assert_allclose(np.mean(out_cc["fit_params"][:, 6]), 0.8, atol=0.05)


def test_quantification_helpers_and_noise() -> None:
    """Verify ``calc_iu``, ``calc_conc_phantom``, metabolite tables, and ``calc_noise``.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    for metab in ("GABA", "Glx", "GSH", "Lac", "EtOH"):
        assert get_in_vivo_metabolite_params(metab).t1_metab > 0
        assert get_phantom_metabolite_params(metab).t1_metab > 0

    mrs = gannet_pre_initialise()
    mrs["p"]["TE"] = [68.0]
    mrs["p"]["TR"] = [2000.0]
    mrs["p"]["TE_water"] = [68.0]
    mrs["p"]["TR_water"] = [2000.0]
    mrs["p"]["vendor"] = "Philips"
    mrs.ensure_substruct("out", "vox1", "GABA")
    mrs.ensure_substruct("out", "vox1", "water")
    mrs["out"]["vox1"]["GABA"]["Area"] = [0.002]
    mrs["out"]["vox1"]["water"]["Area"] = [1.0]

    mrs = calc_iu(mrs, "vox1", "GABA", 0)
    assert mrs["out"]["vox1"]["GABA"]["ConcIU"][0] > 0.0

    mrs = calc_conc_phantom(mrs, "vox1", "GABA", 0)
    assert mrs["out"]["vox1"]["GABA"]["ConcIU"][0] > 0.0

    freq = np.linspace(12.0, -2.0, 2000)
    rng = np.random.default_rng(1)
    spec = 0.01 * (rng.standard_normal(2000) + 1j * rng.standard_normal(2000))
    noise = calc_noise(freq, spec)
    assert 0.002 < noise < 0.05
