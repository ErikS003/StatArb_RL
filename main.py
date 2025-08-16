from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

#Empirical Mean Reversion Time 

def _important_extrema_indices(
    x: np.ndarray,
    C: float = 2.0
) -> List[int]:
    """
    Detect "important" local extremes per Sec. 3.1 in Ning & Lee (2024).
    Let s be the sample std deviation of the series {X_t}.
    A point m is an important minimum if it is the minimum on a segment [i, j] (i <= m <= j)
    and both segment endpoints are at least C*s ABOVE X_m.
    Similarly for important maxima with X_m at least C*s ABOVE both endpoints.

    We approximate this with prominence-like logic:
    - Require m to be a strict local min/max
    - On each side, the series must move by >= C*s away from X_m before we accept it
    - Also check m is the min/max on [i, j].

    Parameters
    ----------
    x : np.ndarray
        1D array of spread values.
    C : float
        Threshold in units of sample std (default 2.0, as in the paper's OU experiments).

    Returns
    -------
    List[int]
        Indices of important local extremes (minima and maxima), ordered.
    """
    n = len(x)
    if n < 3:
        return []

    s = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if s == 0.0:
        return []

    extremes = []

    # Helper to find the first index to the left/right where the deviation criterion is met
    def first_left_ge(idx: int, thresh: float, is_min: bool) -> Optional[int]:
        # For minima: need X[left] - X[idx] >= thresh
        # For maxima: need X[idx] - X[left] >= thresh
        left = idx - 1
        while left >= 0:
            if is_min:
                if x[left] - x[idx] >= thresh:
                    return left
            else:
                if x[idx] - x[left] >= thresh:
                    return left
            left -= 1
        return None

    def first_right_ge(idx: int, thresh: float, is_min: bool) -> Optional[int]:
        right = idx + 1
        while right < n:
            if is_min:
                if x[right] - x[idx] >= thresh:
                    return right
            else:
                if x[idx] - x[right] >= thresh:
                    return right
            right += 1
        return None

    for m in range(1, n - 1):
        is_local_min = x[m] < x[m - 1] and x[m] < x[m + 1]
        is_local_max = x[m] > x[m - 1] and x[m] > x[m + 1]

        if not (is_local_min or is_local_max):
            continue

        thresh = C * s
        if is_local_min:
            Li = first_left_ge(m, thresh, is_min=True)
            Rj = first_right_ge(m, thresh, is_min=True)
            if Li is None or Rj is None:
                continue
            # check minimum on [Li, Rj]
            if x[m] <= np.min(x[Li:Rj + 1]):
                extremes.append(m)

        elif is_local_max:
            Li = first_left_ge(m, thresh, is_min=False)
            Rj = first_right_ge(m, thresh, is_min=False)
            if Li is None or Rj is None:
                continue
            # check maximum on [Li, Rj]
            if x[m] >= np.max(x[Li:Rj + 1]):
                extremes.append(m)

    extremes.sort()
    return extremes


def compute_emrt(
    x: np.ndarray,
    C: float = 2.0
) -> Tuple[float, List[int]]:
    """
    Compute EMRT (empirical mean reversion time) per Eq. on p.4–5.
    Build {tau_n}: tau_1 first important extreme; tau_2 first crossing of mean after tau_1;
    tau_3 next important extreme after tau_2; tau_4 next mean crossing; ...

    r = (2/N) * sum_{i even} (tau_i - tau_{i-1})

    Returns inf if sequence is too short to define r.

    Parameters
    ----------
    x : np.ndarray
        Spread values.
    C : float
        Threshold used in "important extremes" definition.

    Returns
    -------
    (r, taus) : (float, List[int])
        EMRT value and the tau index sequence used.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return float("inf"), []

    theta = float(np.mean(x))
    extremes = _important_extrema_indices(x, C=C)
    if not extremes:
        return float("inf"), []

    taus = []
    # tau1: first important extreme
    t = extremes[0]
    taus.append(t)

    # helper: crossing index after t where series crosses the sample mean
    def first_cross_after(start_idx: int, mean_val: float) -> Optional[int]:
        # find u >= start_idx where (x[u]-mean)*(x[u-1]-mean) <= 0 and sign changes
        prev = start_idx
        for u in range(start_idx + 1, n):
            if (x[u] - mean_val) == 0.0:
                return u
            if (x[prev] - mean_val) == 0.0:
                prev = u
                continue
            if (x[u] - mean_val) * (x[prev] - mean_val) < 0.0:
                return u
            prev = u
        return None

    # helper: next important extreme AFTER idx
    def next_extreme_after(idx: int) -> Optional[int]:
        for e in extremes:
            if e > idx:
                return e
        return None

    while True:
        # tau2k: crossing after tau_{2k-1}
        cross = first_cross_after(taus[-1], theta)
        if cross is None:
            break
        taus.append(cross)
        # tau2k+1: important extreme after crossing
        nxt = next_extreme_after(taus[-1])
        if nxt is None:
            break
        taus.append(nxt)

    if len(taus) < 2:
        return float("inf"), taus

    # r = (2/N) * sum over even indices of (tau_i - tau_{i-1})
    intervals = [taus[i] - taus[i - 1] for i in range(1, len(taus), 2)]
    if not intervals:
        return float("inf"), taus
    N = len(taus)
    r = (2.0 / N) * float(np.sum(intervals))
    return r, taus


def best_coefficient_emrt(
    s1: np.ndarray,
    s2: np.ndarray,
    C: float = 2.0,
    grid_min: float = -3.0,
    grid_max: float = 3.0,
    grid_step: float = 0.05,
    var_cap: Optional[float] = None
) -> Tuple[float, float]:
    """
    Grid-search B in X = S1 - B*S2 that MINIMIZES EMRT on the formation window.
    Returns (best_B, best_emrt).

    Parameters
    ----------
    s1, s2 : np.ndarray
        Price series (aligned) for the two assets.
    C : float
        Important-extrema threshold (std units). Default 2.0 per paper's OU examples.
    grid_* : float
        Grid search params. Paper uses [-3,3] in 0.01 steps; we default to 0.05 for speed.
    var_cap : Optional[float]
        If provided, only consider spreads with variance < var_cap (paper mentions S^2(Y) < M).

    Returns
    -------
    (B, r)
    """
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    assert len(s1) == len(s2) and len(s1) > 10, "Need aligned price arrays with length > 10"

    best_B, best_r = None, float("inf")
    grid = np.arange(grid_min, grid_max + 1e-9, grid_step)

    for B in grid:
        x = s1 - B * s2
        if var_cap is not None and np.var(x) >= var_cap:
            continue
        r, _ = compute_emrt(x, C=C)
        if r < best_r:
            best_r = r
            best_B = float(B)

    if best_B is None:
        # fallback
        best_B = 1.0
        best_r = float("inf")
    return best_B, best_r
