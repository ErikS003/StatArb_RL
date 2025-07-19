import numpy as np
import random

class OUProcess:
    """
    (Euler-Maruyama discretisation)
    """
    def __init__(self, mu=1.0, theta=1.0, sigma=0.1, x0=None, dt=1.0):
        self.mu, self.theta, self.sigma, self.dt = mu, theta, sigma, dt
        self.x0 = theta if x0 is None else x0

    def sample_path(self, n_steps: int) -> np.ndarray:
        x = np.empty(n_steps + 1)
        x[0] = self.x0
        sqrt_dt_sigma = self.sigma * np.sqrt(self.dt)
        for t in range(1, n_steps + 1):
            dx = self.mu * (self.theta - x[t - 1]) * self.dt
            dx += sqrt_dt_sigma * np.random.randn()
            x[t] = x[t - 1] + dx
        return x

