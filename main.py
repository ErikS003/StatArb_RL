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


# 2) RL: Tabular Q-learning agent (Sec. 4.2)

@dataclass
class RLConfig:
    lookback_l: int = 4
    # OLD: k_thresh_pct -> keep for backwards compat but unused when use_zscore_state=True
    k_thresh_pct: float = 3.0
    # NEW: use z-score state & threshold in sigmas
    use_zscore_state: bool = True
    k_std: float = 0.25        # bin threshold in σ-units for Δz
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    trans_cost: float = 0.0
    seed: Optional[int] = None

class MeanReversionEnv:
    """
    X_t is the spread; we standardize it to z_t = (X_t - theta)/std for states.
    Actions: -1 (sell/short or reduce), 0 (hold), +1 (buy/long or reduce).
    Position pos ∈ {-1, 0, +1}, one unit max.
    Reward: R_{t+1} = A_t * (theta - X_t) - c*|A_t|
    """

    def __init__(self, x: np.ndarray, theta: float, cfg: RLConfig):
        self.x = np.asarray(x, dtype=float)
        self.theta = float(theta)
        self.cfg = cfg
        self.t = None
        self.pos = None  # -1, 0, +1
        self.done = None

        # Standardization for state features
        std = float(np.std(self.x, ddof=1))
        self.z = (self.x - self.theta) / (std + 1e-8)

    def reset(self):
        self.t = self.cfg.lookback_l
        self.pos = 0
        self.done = False
        return self._state()

    def _state_vector(self, t: int) -> np.ndarray:
        l = self.cfg.lookback_l
        if self.cfg.use_zscore_state:
            # Use Δz bins in σ-units
            window = self.z[t - l + 1 : t + 1]
            prev   = self.z[t - l     : t]
            dz = window - prev
            k = float(self.cfg.k_std)
            di = np.zeros(l, dtype=int)
            di[dz >  k] =  2
            di[(dz > 0) & (dz <= k)]  =  1
            di[(dz < 0) & (dz >= -k)] = -1
            di[dz < -k] = -2
            return di
        else:
            # (legacy) % change binning (not recommended for spreads near 0)
            k = self.cfg.k_thresh_pct
            window = self.x[t - l + 1 : t + 1]
            prev   = self.x[t - l     : t]
            pct = (window - prev) / (np.where(prev != 0, prev, 1.0)) * 100.0
            di = np.zeros(l, dtype=int)
            di[pct >  k] =  2
            di[(pct > 0) & (pct <= k)]  =  1
            di[(pct < 0) & (pct >= -k)] = -1
            di[pct < -k] = -2
            return di

    def _state_index(self, di: np.ndarray) -> int:
        mapping = {-2:0, -1:1, 1:2, 2:3}
        idx = 0
        for v in di:
            idx = idx * 4 + mapping[int(v)]
        return idx

    def _state(self) -> int:
        return self._state_index(self._state_vector(self.t))

    def allowed_actions(self) -> List[int]:
        # one-step inventory changes only
        if self.pos == 0:
            return [-1, 0, +1]
        elif self.pos == +1:
            return [0, -1]      # can only reduce/close
        else:  # pos == -1
            return [0, +1]      # can only reduce/close

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode already done")

        if action not in self.allowed_actions():
            action = 0

        # Reward at t
        R = float(action) * (self.theta - self.x[self.t]) - self.cfg.trans_cost * abs(int(action))

        # Inventory transition
        if action == +1 and self.pos < +1:
            self.pos += 1
        elif action == -1 and self.pos > -1:
            self.pos -= 1

        # Advance
        self.t += 1
        if self.t >= len(self.x) - 1:
            self.done = True

        s_next = self._state() if not self.done else None
        return (s_next if s_next is not None else -1), R, self.done, {"pos": self.pos, "t": self.t}

class QLearningAgent:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg
        self.num_states = 4 ** cfg.lookback_l
        self.num_actions = 3  # [-1, 0, +1] (we'll mask illegal actions)
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
        self.Q = np.zeros((self.num_states, self.num_actions), dtype=float)

    @staticmethod
    def _action_to_index(a: int) -> int:
        return { -1:0, 0:1, +1:2 }[a]

    @staticmethod
    def _index_to_action(i: int) -> int:
        return { 0:-1, 1:0, 2:+1 }[i]

    def policy_action(self, state: int, allowed: List[int], explore: bool) -> int:
        if explore and np.random.rand() < self.cfg.epsilon:
            return int(np.random.choice(allowed))

        # Exploit: pick best allowed action
        q_row = self.Q[state]
        # Mask illegal actions by -inf
        mask = np.full(self.num_actions, -np.inf)
        for a in allowed:
            mask[self._action_to_index(a)] = 0.0
        q_masked = q_row + mask
        a_idx = int(np.argmax(q_masked))
        return self._index_to_action(a_idx)

    def update(self, s: int, a: int, r: float, s_next: Optional[int], allowed_next: List[int]):
        a_idx = self._action_to_index(a)
        q_sa = self.Q[s, a_idx]

        if s_next is None or len(allowed_next) == 0:
            target = r
        else:
            # max over allowed actions
            q_next = self.Q[s_next]
            mask = np.full(self.num_actions, -np.inf)
            for an in allowed_next:
                mask[self._action_to_index(an)] = 0.0
            best_next = np.max(q_next + mask)
            target = r + self.cfg.gamma * best_next

        self.Q[s, a_idx] = (1 - self.cfg.alpha) * q_sa + self.cfg.alpha * target

