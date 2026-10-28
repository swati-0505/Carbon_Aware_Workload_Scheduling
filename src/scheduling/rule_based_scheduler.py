import numpy as np
import pandas as pd
from .baseline_scheduler import earliest_feasible, feasible_starts

FORECAST_KINDS = ("oracle", "noisy_oracle", "seasonal24", "seasonal168", "persistence")

def make_forecast(ci, kind, t, horizon, rng=None, noise_pct=0.0):
    if kind == "oracle":
        return ci[t:t + horizon].copy()
    if kind == "noisy_oracle":
        f = ci[t:t + horizon].copy()
        noise = rng.normal(0.0, noise_pct / 100.0, size=len(f))
        return np.maximum(f * (1.0 + noise), 1.0)
    k = np.arange(horizon)
    if kind == "persistence":
        return np.full(horizon, ci[t - 1])
    if kind == "seasonal24":
        return ci[t - 24 + (k % 24)]
    if kind == "seasonal168":
        return ci[t - 168 + (k % 168)]
    raise ValueError(f"unknown forecast_kind: {kind}")

def schedule(jobs, ci, capacity_kw, policy="forecast", forecast_kind="oracle",
             threshold_quantile=0.3, noise_pct=0.0, seed=0):
    ci = np.asarray(ci, dtype=float)
    n = len(ci)
    load = np.zeros(n)
    arr = jobs["arrival"].values
    dur = jobs["duration"].values
    pw = jobs["power_kw"].values
    sl = jobs["slack"].values
    starts = np.full(len(jobs), -1, dtype=int)
    rng = np.random.default_rng(seed)

    if policy == "threshold":
        thr = (pd.Series(ci).rolling(168, min_periods=24)
               .quantile(threshold_quantile).shift(1).values)
    elif policy != "forecast":
        raise ValueError(f"unknown policy: {policy}")
    elif forecast_kind not in FORECAST_KINDS:
        raise ValueError(f"unknown forecast_kind: {forecast_kind}")

    for i in range(len(jobs)):
        a, d, p, s = int(arr[i]), int(dur[i]), pw[i], int(sl[i])
        if s == 0:
            st = earliest_feasible(load, capacity_kw, p, d, a)
        else:
            cand = feasible_starts(load, capacity_kw, p, d, a, a + s)
            if len(cand) == 0:
                st = earliest_feasible(load, capacity_kw, p, d, a + s + 1)
            elif policy == "threshold":
                ok = cand[ci[cand] <= thr[cand]]   
                st = int(ok[0]) if len(ok) else int(cand[-1])
            else:
                f = make_forecast(ci, forecast_kind, a, s + d, rng, noise_pct)
                cs = np.concatenate([[0.0], np.cumsum(f)])
                rel = cand - a
                cost = cs[rel + d] - cs[rel]
                best = np.flatnonzero(cost <= cost.min() + 1e-6)[0]
                st = int(cand[best])
        if st >= 0:
            load[st:st + d] += p
            starts[i] = st
    return starts, load