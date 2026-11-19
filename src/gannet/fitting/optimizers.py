"""Optimization wrappers mirroring MATLAB lsqcurvefit, nlinfit, and lsqnonlin."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares


@dataclass(frozen=True)
class NlinfitResult:
    """Result container returned by ``nlinfit``.

    Attributes
    ----------
    beta: NDArray[np.float64]
        Optimized parameter vector.
    resid: NDArray[np.float64]
        Residual vector defined as ``ydata - model(beta, xdata)``.
    mse: float
        Mean squared error ``sum(resid**2) / max(n - p, 1)``.
    """

    beta: NDArray[np.float64]
    resid: NDArray[np.float64]
    mse: float


def _sanitize_bounds(
    x0: NDArray[np.float64],
    lb: ArrayLike | None,
    ub: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Normalize bounds and clip initial guess inside ``[lb, ub]``.

    Parameters
    ------------
    x0: NDArray[np.float64]
        Initial parameter vector.
    lb: ArrayLike | None
        Lower bounds or None for unbounded.
    ub: ArrayLike | None
        Upper bounds or None for unbounded.

    Returns
    --------
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
            Tuple containing sanitized ``(x0_clipped, lower, upper)``.
    """
    n_params = len(x0)
    lower = (
        np.full(n_params, -np.inf, dtype=np.float64)
        if lb is None or np.asarray(lb).size == 0
        else np.asarray(lb, dtype=np.float64).ravel()
    )
    upper = (
        np.full(n_params, np.inf, dtype=np.float64)
        if ub is None or np.asarray(ub).size == 0
        else np.asarray(ub, dtype=np.float64).ravel()
    )
    swapped_lo = np.minimum(lower, upper)
    swapped_hi = np.maximum(lower, upper)
    equal_mask = np.isfinite(swapped_lo) & (swapped_lo == swapped_hi)
    swapped_hi = np.where(equal_mask, swapped_hi + 1e-12, swapped_hi)
    clipped = np.clip(x0, swapped_lo, swapped_hi)
    result = (clipped, swapped_lo, swapped_hi)
    return result


def lsqcurvefit(
    model_fn: Callable[[NDArray[np.float64], object], NDArray[np.float64]],
    x0: ArrayLike,
    xdata: object,
    ydata: ArrayLike,
    lb: ArrayLike | None = None,
    ub: ArrayLike | None = None,
    max_iter: int = 800,
    tol_x: float = 1e-4,
    tol_fun: float = 1e-4,
) -> NDArray[np.float64]:
    """Solve bounded or unbounded nonlinear curve-fitting least squares.

    Mirrors MATLAB's ``lsqcurvefit`` using Trust-Region Reflective when
    bounds are present or Levenberg-Marquardt when unbounded.

    Parameters
    ------------
    model_fn: Callable[[NDArray[np.float64], object], NDArray[np.float64]]
        Model callable ``f(params, xdata) -> y_pred``.
    x0: ArrayLike
        Initial parameter guess.
    xdata: object
        Independent variable passed to ``model_fn``.
    ydata: ArrayLike
        Target observed array.
    lb: ArrayLike | None
        Lower parameter bounds. Defaults to None.
    ub: ArrayLike | None
        Upper parameter bounds. Defaults to None.
    max_iter: int
        Maximum number of iterations/function evaluations scaling. Defaults to 800.
    tol_x: float
        Step tolerance. Defaults to 1e-4.
    tol_fun: float
        Function value tolerance. Defaults to 1e-4.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Optimized parameter vector.
    """
    x0_arr = np.asarray(x0, dtype=np.float64).ravel()
    y_arr = np.asarray(ydata, dtype=np.float64).ravel()
    has_bounds = (lb is not None and np.asarray(lb).size > 0) or (
        ub is not None and np.asarray(ub).size > 0
    )

    def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute curve-fitting residual vector."""
        pred = np.asarray(model_fn(params, xdata), dtype=np.float64).ravel()
        return pred - y_arr

    max_nfev = max(max_iter * (len(x0_arr) + 1), 1000)
    if has_bounds:
        x_init, lower, upper = _sanitize_bounds(x0_arr, lb, ub)
        res = least_squares(
            residuals,
            x_init,
            bounds=(lower, upper),
            method="trf",
            xtol=tol_x,
            ftol=tol_fun,
            max_nfev=max_nfev,
        )
    else:
        method = "lm" if y_arr.size >= x0_arr.size else "trf"
        res = least_squares(
            residuals,
            x0_arr,
            method=method,
            xtol=tol_x,
            ftol=tol_fun,
            max_nfev=max_nfev,
        )
    params_out: NDArray[np.float64] = np.asarray(res.x, dtype=np.float64)
    return params_out


def nlinfit(
    xdata: object,
    ydata: ArrayLike,
    model_fn: Callable[[NDArray[np.float64], object], NDArray[np.float64]],
    beta0: ArrayLike,
    max_iter: int = 400,
    tol_x: float = 1e-6,
    tol_fun: float = 1e-6,
) -> NlinfitResult:
    """Perform unconstrained Levenberg-Marquardt nonlinear regression.

    Mirrors MATLAB's ``nlinfit`` return conventions where residuals are
    ``ydata - model_fn(beta, xdata)``.

    Parameters
    ------------
    xdata: object
        Independent variable passed to ``model_fn``.
    ydata: ArrayLike
        Observed dependent data vector.
    model_fn: Callable[[NDArray[np.float64], object], NDArray[np.float64]]
        Model callable ``f(params, xdata) -> y_pred``.
    beta0: ArrayLike
        Initial parameter guess.
    max_iter: int
        Maximum iteration budget. Defaults to 400.
    tol_x: float
        Parameter step tolerance. Defaults to 1e-6.
    tol_fun: float
        Cost function tolerance. Defaults to 1e-6.

    Returns
    --------
        NlinfitResult: NlinfitResult dataclass
            Optimized parameters, residuals (``ydata - y_pred``), and MSE.
    """
    b0 = np.asarray(beta0, dtype=np.float64).ravel()
    y_arr = np.asarray(ydata, dtype=np.float64).ravel()

    def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute ydata minus model prediction."""
        pred = np.asarray(model_fn(params, xdata), dtype=np.float64).ravel()
        return y_arr - pred

    max_nfev = max(max_iter * (len(b0) + 1), 1000)
    method = "lm" if y_arr.size >= b0.size else "trf"
    res = least_squares(
        residuals,
        b0,
        method=method,
        xtol=tol_x,
        ftol=tol_fun,
        max_nfev=max_nfev,
    )
    beta_opt = np.asarray(res.x, dtype=np.float64)
    resid = residuals(beta_opt)
    dof = max(len(resid) - len(beta_opt), 1)
    mse = float(np.sum(resid**2) / dof)
    result = NlinfitResult(beta=beta_opt, resid=resid, mse=mse)
    return result


def lsqnonlin(
    fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: ArrayLike,
    lb: ArrayLike | None = None,
    ub: ArrayLike | None = None,
    max_iter: int = 400,
    tol_x: float = 1e-6,
    tol_fun: float = 1e-6,
) -> NDArray[np.float64]:
    """Minimize sum of squares of a vector-valued residual function.

    Mirrors MATLAB's ``lsqnonlin`` with Levenberg-Marquardt when unbounded.

    Parameters
    ------------
    fun: Callable[[NDArray[np.float64]], NDArray[np.float64]]
        Residual vector function ``fun(x) -> r``.
    x0: ArrayLike
        Initial parameter guess.
    lb: ArrayLike | None
        Optional lower bounds. Defaults to None.
    ub: ArrayLike | None
        Optional upper bounds. Defaults to None.
    max_iter: int
        Maximum iteration budget. Defaults to 400.
    tol_x: float
        Step tolerance. Defaults to 1e-6.
    tol_fun: float
        Function tolerance. Defaults to 1e-6.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Optimized parameter vector.
    """
    x0_arr = np.asarray(x0, dtype=np.float64).ravel()
    has_bounds = (lb is not None and np.asarray(lb).size > 0) or (
        ub is not None and np.asarray(ub).size > 0
    )
    max_nfev = max(max_iter * (len(x0_arr) + 1), 1000)

    def wrapped(params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ensure residual vector is 1D float64."""
        return np.asarray(fun(params), dtype=np.float64).ravel()

    if has_bounds:
        x_init, lower, upper = _sanitize_bounds(x0_arr, lb, ub)
        res = least_squares(
            wrapped,
            x_init,
            bounds=(lower, upper),
            method="trf",
            xtol=tol_x,
            ftol=tol_fun,
            max_nfev=max_nfev,
        )
    else:
        probe = wrapped(x0_arr)
        method = "lm" if probe.size >= x0_arr.size else "trf"
        res = least_squares(
            wrapped,
            x0_arr,
            method=method,
            xtol=tol_x,
            ftol=tol_fun,
            max_nfev=max_nfev,
        )
    params_out: NDArray[np.float64] = np.asarray(res.x, dtype=np.float64)
    return params_out


def zscore(values: ArrayLike) -> NDArray[np.float64]:
    """Standardize a vector using sample standard deviation (ddof=1).

    Matches MATLAB's ``zscore`` function.

    Parameters
    ------------
    values: ArrayLike
        1D input numeric array.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Standardized array with zero mean and unit sample std.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        zeros: NDArray[np.float64] = np.zeros_like(arr, dtype=np.float64)
        return zeros
    std_val = float(np.std(arr, ddof=1))
    if std_val == 0.0:
        zeros = np.zeros_like(arr, dtype=np.float64)
        return zeros
    result: NDArray[np.float64] = (arr - float(np.mean(arr))) / std_val
    return result


def mad_median(values: ArrayLike) -> float:
    """Compute median absolute deviation from the median (MATLAB ``mad(x, 1)``).

    Parameters
    ------------
    values: ArrayLike
        Input numeric array.

    Returns
    --------
        float: float object
            Median of ``|x - median(x)|``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    result = float(np.median(np.abs(arr - med)))
    return result
