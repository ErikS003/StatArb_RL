import numpy as np
import random
from collections import defaultdict

class MeanReversionEnv:
    def __init__(self, price_series, lookback=4, k=0.03, transaction_cost=0.001, theta=None):
        self.price = price_series
        self.L = lookback
        self.k = k                    #threshold
        self.c = transaction_cost
        self.theta = float(theta) if theta is not None else float(np.mean(price_series))
        self.action_map = {-1: 0, 0: 1, 1: 2}
        self.reset()

    # --- helpers ----------------------------------------------------------
    def _discretise_change(self, pct):
        if pct >  self.k * 100: return  2
        if pct >  0:            return  1
        if pct < -self.k * 100: return -2
        if pct <  0:            return -1
        return 0  # extremely small move

    def _get_state(self):
        # Last L discrete return symbols (oldest => newest) + current position
        symbols = [self._discretise_change(
            (self.price[self.t - j] - self.price[self.t - j - 1]) / self.price[self.t - j - 1] * 100
        ) for j in range(self.L, 0, -1)]
        return tuple(symbols) + (self.position,)

    def reset(self):
        # start after we can compute L changes
        self.t = self.L
        self.position = 0
        return self._get_state()

    def step(self, action):
        """
        action in {-1,0,+1}; availability checked externally
        reward  = A_t · (θ - X_t) - c |A_t|
        """
        reward = 0.0
        if action ==  1 and self.position == 0:  # buy to open
            self.position = 1
            reward =  (self.theta - self.price[self.t]) - self.c
        elif action == -1 and self.position == 1:  # sell to close
            self.position = 0
            reward = -(self.theta - self.price[self.t]) - self.c

        self.t += 1
        done = self.t >= len(self.price)
        next_state = None if done else self._get_state()
        return next_state, reward, done, {}

