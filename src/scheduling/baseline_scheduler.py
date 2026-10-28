import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-9

def immediate_load(jobs, n_hours):
    p = jobs["power_kw"].values
    a = jobs["arrival"].values
    d = jobs["duration"].values
    diff = np.zeros(n_hours + 1)
    np.add.at(diff, a, p)
    np.add.at(diff, a + d, -p)
    return np.cumsum(diff)[:n_hours]

def feasible_starts(load, capacity_kw, power_kw, dur, lo, hi):
    hi = min(hi, len(load) - dur)
    if hi < lo:
        return np.empty(0, dtype=int)
    seg = load[lo:hi + dur]
    peak = sliding_window_view(seg, dur).max(axis=1)
    return lo + np.flatnonzero(peak + power_kw <= capacity_kw + EPS)

def earliest_feasible(load, capacity_kw, power_kw, dur, lo, step=48):
    a = lo
    last = len(load) - dur
    while a <= last:
        s = feasible_starts(load, capacity_kw, power_kw, dur, a, a + step - 1)
        if len(s):
            return int(s[0])
        a += step
    return -1

def schedule(jobs, n_hours, capacity_kw):
    load = np.zeros(n_hours)
    arr = jobs["arrival"].values
    dur = jobs["duration"].values
    pw = jobs["power_kw"].values
    starts = np.full(len(jobs), -1, dtype=int)
    for i in range(len(jobs)):
        st = earliest_feasible(load, capacity_kw, pw[i], dur[i], arr[i])
        if st >= 0:
            load[st:st + dur[i]] += pw[i]
            starts[i] = st
    return starts, load