import numpy as np
import pandas as pd

DURATIONS = np.array([1, 2, 3, 4, 6, 8, 12, 24])
DURATION_P = np.array([0.30, 0.20, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03])

POWERS_KW = np.array([5, 10, 20, 40, 80])
POWER_P = np.array([0.30, 0.30, 0.20, 0.15, 0.05])
_PROFILE = np.array([0.4] * 7 + [0.8] * 2 + [1.4] * 9 + [1.0] * 4 + [0.6] * 2)
HOURLY_PROFILE = _PROFILE / _PROFILE.mean()

MAX_SLACK = 48          
RESERVE_HOURS = 24 + MAX_SLACK + 24   


def generate_jobs(n_hours, jobs_per_hour=1.0, flexible_fraction=0.6,
                  slack_hours=24, seed=42, warmup_hours=168):
    rng = np.random.default_rng(seed)
    hours = np.arange(warmup_hours, n_hours - RESERVE_HOURS)
    lam = jobs_per_hour * HOURLY_PROFILE[hours % 24]
    counts = rng.poisson(lam)
    arrival = np.repeat(hours, counts)
    n = len(arrival)

    duration = rng.choice(DURATIONS, n, p=DURATION_P)
    power = rng.choice(POWERS_KW, n, p=POWER_P)
    u = rng.random(n)

    flexible = u < flexible_fraction
    slack = np.where(flexible, int(slack_hours), 0)

    return pd.DataFrame({
        "job_id": np.arange(n),
        "arrival": arrival.astype(int),
        "duration": duration.astype(int),
        "power_kw": power.astype(float),
        "flexible": flexible,
        "slack": slack.astype(int),
    })