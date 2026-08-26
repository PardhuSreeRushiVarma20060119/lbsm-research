"""
robust_fitting.py
==================
LBSM — Robust Full-Covariance HMM Fitting
-------------------------------------------
Mitigation stack for the numerical instability documented in
``LBSM-ISSUE-NB07-001`` (``outputs/reports/issues/lbsm_covariance_issue.pdf``):
switching a Gaussian HMM from ``covariance_type="diag"`` to ``"full"`` for
NB07's robustness grid introduces ill-conditioning risk whenever the
effective per-state sample size is small relative to the ``d(d+1)/2`` free
covariance parameters per state (d=6 features -> 21 parameters/state).

Layers implemented here (numbering matches the issue document):
  1. Z-scoring                              -> reuses ``src.telemetry.normalization``
  2. Tikhonov regularisation (min_covar) sweep
  3. Multiple random restarts, best-log-likelihood selection
  4. KMeans-warmed initialisation (first restart only)
  5. Covariance-type fallback hierarchy: full -> tied -> diag
  6. Pre-fit data-sufficiency warning (warns, never blocks the fit)
  7. Post-fit covariance health check (condition number / min eigenvalue)

Design principle (from the issue document): *prevent-the-fit is the wrong
call*. This module always attempts the fit, catches failures post-hoc, and
records what happened rather than silently discarding or blocking a
configuration — callers (NB07's robustness grid) decide what to do with an
unreliable result using the returned :class:`RobustHMMResult`.

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 6.2 — Model Order Selection and Identifiability
LBSM-ISSUE-NB07-001 — Full Covariance HMM Numerical Instability
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HMMLEARN_AVAILABLE = False

from sklearn.cluster import KMeans

from ..telemetry.normalization import ZScoreParams, apply_zscore, fit_zscore


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class RobustHMMResult:
    """Outcome of :func:`fit_hmm_robust`.

    Attributes
    ----------
    model                : fitted GaussianHMM (``None`` only in the
        essentially-unreachable case where even diagonal covariance fails)
    zscore_params        : :class:`ZScoreParams` used to standardise ``X_raw``
        before fitting — needed to transform any new data into the same space
    cov_type_used        : ``"full"``, ``"tied"``, or ``"diag"`` — the
        covariance structure that actually produced ``model``, after any
        fallback (may differ from the requested ``covariance_type``)
    min_covar_used        : the Tikhonov regularisation value that produced
        the winning fit
    n_restarts_succeeded  : how many of the attempted restarts (at the
        winning covariance type) converged to a finite, healthy model
    n_restarts_attempted  : total restarts attempted at the winning
        covariance type (``len(min_covar_grid) * n_restarts`` for ``"full"``)
    converged             : whether any restart produced a usable model
    ll_per_obs             : final log-likelihood per observation
    max_condition_number  : worst covariance condition number across states
    min_eigenvalue         : smallest covariance eigenvalue across states
    tried_fallback         : whether the covariance-type fallback hierarchy
        (full -> tied -> diag) had to be used
    reliable               : ``True`` iff no fallback was needed AND the
        pre-fit sufficiency warning was not triggered — the "trust this
        result without caveats" flag for the NB07 results table
    sufficiency_warning    : the data-sufficiency warning text, if any was
        raised for the *requested* covariance type before fitting
    attempt_log            : per-attempt diagnostic log (empty unless
        ``fit_hmm_robust(..., collect_attempt_log=True)``) — one dict per
        (min_covar, restart) attempt at the *requested* covariance type
        (fallback attempts, if any, are not logged here), each with keys
        ``min_covar``, ``restart``, ``ll_per_obs``, ``n_iter``,
        ``em_converged`` (Baum-Welch's own convergence flag),
        ``health_ok`` (passed :func:`check_covariance_health`),
        ``kept`` (finite score and healthy, i.e. eligible to be the winner),
        ``is_winner`` (this attempt is the one selected as ``model``),
        ``max_condition_number``, ``min_eigenvalue`` — recorded for
        *every* attempt, not just the winner, so failed/discarded attempts
        are retained rather than silently dropped.
    """
    model                : object
    zscore_params        : ZScoreParams
    cov_type_used        : str
    min_covar_used       : float
    n_restarts_succeeded : int
    n_restarts_attempted : int
    converged            : bool
    ll_per_obs           : float
    max_condition_number : float
    min_eigenvalue       : float
    tried_fallback       : bool
    reliable              : bool
    sufficiency_warning   : Optional[str]
    attempt_log           : List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 6 — pre-fit data-sufficiency warning (warns, never blocks)
# ---------------------------------------------------------------------------
def check_data_sufficiency(
    n_obs           : int,
    n_components    : int,
    n_features      : int,
    covariance_type : str = "full",
    safety_factor   : int = 10,
) -> Optional[str]:
    """Warn (never block) if avg obs/state is below the reliability rule of
    thumb: ``safety_factor * d(d+1)/2`` per state (Eq. 3 of
    ``LBSM-ISSUE-NB07-001``). A no-op for non-``"full"`` covariance, which
    doesn't require estimating off-diagonal terms.

    Returns
    -------
    warning : the warning message if triggered, else ``None``.
    """
    if covariance_type != "full":
        return None
    n_cov_params  = n_features * (n_features + 1) // 2
    avg_per_state = n_obs / n_components
    min_reliable  = n_cov_params * safety_factor
    if avg_per_state < min_reliable:
        msg = (
            f"Full covariance requested. Average obs/state ({avg_per_state:.0f}) "
            f"< recommended minimum ({min_reliable}). Results may be numerically "
            f"unreliable. Consider covariance_type='tied' or increasing N*T."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        return msg
    return None


# ---------------------------------------------------------------------------
# Layer 7 — post-fit covariance health check
# ---------------------------------------------------------------------------
def check_covariance_health(
    model,
    kappa_thresh : float = 1e8,
    eigval_thresh: float = 1e-8,
) -> List[str]:
    """Inspect a fitted GaussianHMM's covariance matrices for ill-conditioning.

    Handles all of hmmlearn's ``covariance_type`` storage conventions
    (``"full"``: one ``(d,d)`` matrix per state; ``"diag"``/``"spherical"``:
    a ``(d,)`` variance vector per state; ``"tied"``: a single shared
    ``(d,d)`` matrix).

    Returns
    -------
    issues : list of issue strings, one per (state, problem) found. An empty
        list means healthy.
    """
    issues: List[str] = []
    for k in range(model.n_components):
        eigvals = np.linalg.eigvalsh(_per_state_cov_matrix(model, k))
        kappa = eigvals.max() / max(eigvals.min(), 1e-300)
        if eigvals.min() < eigval_thresh:
            issues.append(f"state {k}: min_eigval={eigvals.min():.2e}")
        if kappa > kappa_thresh:
            issues.append(f"state {k}: condition={kappa:.2e}")
    return issues


def _per_state_cov_matrix(model, k: int) -> np.ndarray:
    """Return state ``k``'s covariance as a dense ``(d, d)`` matrix, correctly
    dispatching on hmmlearn's storage convention for each ``covariance_type``
    (``full``: ``(K,d,d)``; ``diag``/``spherical``: ``(K,d)`` per-state
    variances; ``tied``: a single shared ``(d,d)`` matrix, same for every
    ``k``) — distinguishing these by shape alone is ambiguous (a ``"tied"``
    ``(d,d)`` matrix and a single-feature ``"diag"`` ``(K,1)`` array can both
    have ``ndim==2``), so ``model.covariance_type`` is used directly.
    """
    covars = model.covars_
    if model.covariance_type == "full":
        return covars[k]
    if model.covariance_type == "tied":
        return covars
    return np.diag(covars[k])  # diag / spherical


def _covariance_stats(model) -> Tuple[float, float]:
    """Return ``(max_condition_number, min_eigenvalue)`` across all states."""
    max_kappa, min_eig = 0.0, np.inf
    for k in range(model.n_components):
        eigvals = np.linalg.eigvalsh(_per_state_cov_matrix(model, k))
        kappa = eigvals.max() / max(eigvals.min(), 1e-300)
        max_kappa = max(max_kappa, kappa)
        min_eig = min(min_eig, eigvals.min())
    return float(max_kappa), float(min_eig)


# ---------------------------------------------------------------------------
# Layers 2-4 — one covariance-type attempt: min_covar sweep x restarts
# ---------------------------------------------------------------------------
def _attempt(
    X_z             : np.ndarray,
    lengths         : List[int],
    n_components    : int,
    cov_type        : str,
    min_covar_grid  : Sequence[float],
    n_restarts      : int,
    max_iter        : int,
    tol             : float,
    random_state    : int,
    km_centers      : Optional[np.ndarray],
    log             : Optional[List[dict]] = None,
) -> Tuple[Optional[object], float, Optional[float], int, int]:
    """Sweep ``min_covar_grid`` x ``n_restarts`` for one covariance type.

    If ``log`` is given (a list, mutated in place), every attempt — including
    ones that raised, diverged, or failed the health check — appends one
    diagnostic dict to it, so a caller collecting the full attempt log
    retains failed/discarded attempts rather than only the winner. The
    winning attempt's entry (if any) is marked ``is_winner=True`` in-place
    once the winner is known, after the sweep completes.

    Returns
    -------
    (best_model, best_score, best_min_covar, n_succeeded, n_attempted)
    """
    best_model, best_score, best_lam = None, -np.inf, None
    n_succeeded, n_attempted = 0, 0
    winner_entry = None

    for lam in min_covar_grid:
        for rs in range(n_restarts):
            n_attempted += 1
            warm_start = (rs == 0 and km_centers is not None)
            entry = {"min_covar": lam, "restart": rs, "ll_per_obs": float("nan"),
                     "n_iter": None, "em_converged": False, "health_ok": False,
                     "kept": False, "is_winner": False,
                     "max_condition_number": float("nan"), "min_eigenvalue": float("nan")}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = _GaussianHMM(
                        n_components    = n_components,
                        covariance_type = cov_type,
                        n_iter          = max_iter,
                        tol             = tol,
                        min_covar       = lam,
                        init_params     = "stc" if warm_start else "stmc",
                        params          = "stmc",
                        random_state    = random_state + rs,
                    )
                    if warm_start:
                        m.means_ = km_centers.copy()
                    m.fit(X_z, lengths)

                score = m.score(X_z, lengths)
                n_iter = len(m.monitor_.history)
                entry["n_iter"] = n_iter
                entry["em_converged"] = bool(m.monitor_.converged)
                entry["ll_per_obs"] = float(score) / len(X_z) if np.isfinite(score) else float("nan")

                if not np.isfinite(score):
                    if log is not None:
                        log.append(entry)
                    continue
                issues = check_covariance_health(m)
                entry["health_ok"] = not issues
                if issues:
                    kappa, eig = _covariance_stats(m)
                    entry["max_condition_number"], entry["min_eigenvalue"] = kappa, eig
                    if log is not None:
                        log.append(entry)
                    continue

                kappa, eig = _covariance_stats(m)
                entry["max_condition_number"], entry["min_eigenvalue"] = kappa, eig
                entry["kept"] = True
                n_succeeded += 1
                if score > best_score:
                    best_score, best_model, best_lam = score, m, lam
                    winner_entry = entry
            except (ValueError, np.linalg.LinAlgError):
                pass
            if log is not None:
                log.append(entry)

    if winner_entry is not None:
        winner_entry["is_winner"] = True

    return best_model, best_score, best_lam, n_succeeded, n_attempted


# ---------------------------------------------------------------------------
# Full mitigation-stack fitting function
# ---------------------------------------------------------------------------
def fit_hmm_robust(
    X_raw           : np.ndarray,
    lengths         : List[int],
    n_components    : int = 4,
    covariance_type : str = "full",
    min_covar_grid  : Sequence[float] = (1e-2, 1e-3, 1e-4),
    n_restarts      : int = 10,
    max_iter        : int = 300,
    tol             : float = 1e-4,
    random_state    : int = 42,
    use_kmeans_init : bool = True,
    verbose         : bool = False,
    collect_attempt_log : bool = False,
) -> RobustHMMResult:
    """Fit a covariance HMM through the complete ``LBSM-ISSUE-NB07-001``
    mitigation stack.

    Never raises on numerical failure (short of the essentially-unreachable
    case where even diagonal covariance fails) — always returns a usable
    model, falling back to a simpler covariance structure if necessary, and
    records exactly what happened in the returned :class:`RobustHMMResult`
    so the caller can flag unreliable configurations rather than have them
    silently discarded or block the run.

    Parameters
    ----------
    X_raw            : raw (unstandardised) observation matrix (N_total, d)
    lengths          : per-sequence lengths (see :func:`prepare_sequences`)
    n_components     : number of hidden states
    covariance_type  : requested covariance structure — ``"full"`` is the
        motivating case; ``"tied"``/``"diag"`` skip the min_covar sweep
        (a single fixed value is used) since they don't need it
    min_covar_grid   : Tikhonov regularisation values to sweep for
        ``covariance_type="full"``, low regularisation to high
    n_restarts       : random restarts per ``min_covar`` value
    max_iter, tol    : Baum-Welch EM controls
    random_state     : base seed (restart ``rs`` uses ``random_state + rs``)
    use_kmeans_init  : warm-start the first restart's means from KMeans
        centroids fit on the z-scored data (subsequent restarts use
        hmmlearn's own random initialisation, for diversity)
    verbose          : print restart/fallback progress
    collect_attempt_log : if True, populate the returned result's
        ``attempt_log`` with one diagnostic dict per (min_covar, restart)
        attempt at the requested covariance type (see
        :class:`RobustHMMResult`), including failed/discarded attempts, not
        only the winner. ``False`` by default (small overhead, skipped
        unless a caller specifically wants the full audit trail).

    Returns
    -------
    result : RobustHMMResult
    """
    if not _HMMLEARN_AVAILABLE:
        raise ImportError("hmmlearn is required.  pip install hmmlearn")

    n_obs, d = X_raw.shape

    # Layer 1: z-score
    zparams = fit_zscore(X_raw)
    X_z     = apply_zscore(X_raw, zparams)

    # Layer 6: pre-fit sufficiency warning (never blocks)
    sufficiency_warning = check_data_sufficiency(
        n_obs, n_components, d, covariance_type=covariance_type,
    )

    # Layer 4: KMeans warm start (computed once, reused across attempts)
    km_centers = None
    if use_kmeans_init:
        km = KMeans(n_clusters=n_components, n_init=10, random_state=random_state)
        km.fit(X_z)
        km_centers = km.cluster_centers_

    attempt_log: List[dict] = []

    def run(cov_type: str, grid: Sequence[float], restarts: int, log: Optional[List[dict]] = None):
        return _attempt(
            X_z, lengths, n_components, cov_type, grid, restarts,
            max_iter, tol, random_state, km_centers, log=log,
        )

    # Layers 2+3+4+7 at the requested covariance type
    grid = min_covar_grid if covariance_type == "full" else [1e-3]
    model, score, lam, n_succeeded, n_attempted = run(
        covariance_type, grid, n_restarts,
        log=attempt_log if collect_attempt_log else None,
    )
    cov_type_used, tried_fallback = covariance_type, False

    # Layer 5: fallback hierarchy (full -> tied -> diag), single fit each,
    # matching the issue document's simpler fallback protocol
    if model is None and covariance_type == "full":
        if verbose:
            print("  Full covariance failed all restarts -- falling back to 'tied'.")
        tried_fallback = True
        model, score, lam, n_succeeded, n_attempted = run("tied", [1e-3], 1)
        cov_type_used = "tied"
    if model is None:
        if verbose:
            print("  Falling back to 'diag'.")
        tried_fallback = True
        model, score, lam, n_succeeded, n_attempted = run("diag", [1e-6], 1)
        cov_type_used = "diag"

    if model is None:
        # Essentially unreachable -- diagonal covariance's only failure mode
        # is every observation in a state being bit-identical -- but never
        # raise; return a clearly-flagged failure result instead.
        return RobustHMMResult(
            model=None, zscore_params=zparams, cov_type_used="none",
            min_covar_used=float("nan"), n_restarts_succeeded=0,
            n_restarts_attempted=n_attempted, converged=False,
            ll_per_obs=float("-inf"), max_condition_number=float("inf"),
            min_eigenvalue=0.0, tried_fallback=True, reliable=False,
            sufficiency_warning=sufficiency_warning, attempt_log=attempt_log,
        )

    max_kappa, min_eig = _covariance_stats(model)
    reliable = (not tried_fallback) and (sufficiency_warning is None)
    return RobustHMMResult(
        model=model, zscore_params=zparams, cov_type_used=cov_type_used,
        min_covar_used=float(lam), n_restarts_succeeded=n_succeeded,
        n_restarts_attempted=n_attempted, converged=True,
        ll_per_obs=float(score) / n_obs, max_condition_number=max_kappa,
        min_eigenvalue=min_eig, tried_fallback=tried_fallback,
        reliable=reliable, sufficiency_warning=sufficiency_warning,
        attempt_log=attempt_log,
    )
