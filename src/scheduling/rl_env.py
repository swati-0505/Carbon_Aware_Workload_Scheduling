import numpy as np
import pandas as pd

N_CI, N_SLACK, N_HOUR = 5, 5, 4
N_STATES = N_CI * N_SLACK * N_HOUR
N_ACTIONS = 2  # 0 = wait, 1 = start now

def ci_percentile(ci, window=168):
    """Percentile rank (0..1) of each hour within the previous `window` hours.
    Hours with too little history get 0.5."""
    s = pd.Series(np.asarray(ci, dtype=float))
    pct = s.rolling(window + 1, min_periods=25).apply(
        lambda w: (w[:-1] <= w[-1]).mean(), raw=True)
    return pct.fillna(0.5).values


def encode_state(pct, slack_left, hour):
    ci_bin = min(int(pct * N_CI), N_CI - 1)
    if slack_left <= 0:
        sl_bin = 0
    elif slack_left <= 3:
        sl_bin = 1
    elif slack_left <= 8:
        sl_bin = 2
    elif slack_left <= 16:
        sl_bin = 3
    else:
        sl_bin = 4
    return (ci_bin * N_SLACK + sl_bin) * N_HOUR + (hour % 24) // 6


class StoppingEnv:
    n_states = N_STATES
    n_actions = N_ACTIONS

    def __init__(self, ci, slack_hours=24, duration=4, seed=0):
        self.ci = np.asarray(ci, dtype=float)
        self.pct = ci_percentile(self.ci)
        self.cs = np.concatenate([[0.0], np.cumsum(self.ci)])
        self.slack = int(slack_hours)
        self.dur = int(duration)
        self.scale = float(self.ci.std())
        self.t_min = 168
        self.t_max = len(self.ci) - self.slack - self.dur - 1
        if self.t_max <= self.t_min:
            raise ValueError("series too short for this slack and duration")
        self.rng = np.random.default_rng(seed)
        self.t0 = self.t = 0

    def _run_cost(self, t):
        return (self.cs[t + self.dur] - self.cs[t]) / self.dur

    def _state(self):
        return encode_state(self.pct[self.t], self.t0 + self.slack - self.t, self.t % 24)

    def reset(self):
        self.t0 = int(self.rng.integers(self.t_min, self.t_max + 1))
        self.t = self.t0
        return self._state()

    def step(self, action):
        """Returns (next_state, reward, done)."""
        forced = self.t >= self.t0 + self.slack
        if action == 1 or forced:
            reward = (self._run_cost(self.t0) - self._run_cost(self.t)) / self.scale
            return self._state(), float(reward), True
        self.t += 1
        return self._state(), 0.0, False