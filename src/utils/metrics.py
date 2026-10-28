import numpy as np
import pandas as pd

def prefix_sum(ci):
    """Prefix sums of the carbon-intensity series, so that the sum over any
    window [s, s+d) is prefix[s+d] - prefix[s]."""
    ci = np.asarray(ci, dtype=float)
    return np.concatenate([[0.0], np.cumsum(ci)])

def job_emissions_kg(power_kw, start, duration, ci_prefix):
    start = np.asarray(start, dtype=int)
    duration = np.asarray(duration, dtype=int)
    ci_sum = ci_prefix[start + duration] - ci_prefix[start]
    return np.asarray(power_kw, dtype=float) * ci_sum / 1000.0

def load_profile(start, duration, power_kw, n_hours):
    diff = np.zeros(n_hours + 1)
    np.add.at(diff, start, power_kw)
    np.add.at(diff, start + duration, -np.asarray(power_kw, dtype=float))
    return np.cumsum(diff)[:n_hours]

def summarize(jobs, starts, ci):
    ci = np.asarray(ci, dtype=float)
    starts = np.asarray(starts, dtype=int)
    ok = starts >= 0
    j = jobs[ok]
    s = starts[ok]
    d = j["duration"].values
    p = j["power_kw"].values
    a = j["arrival"].values
    sl = j["slack"].values

    kg = job_emissions_kg(p, s, d, prefix_sum(ci))
    energy_kwh = float((p * d).sum())
    delay = s - a
    load = load_profile(s, d, p, len(ci))
    return {
        "n_jobs": int(len(jobs)),
        "unscheduled": int((~ok).sum()),
        "energy_kwh": energy_kwh,
        "emissions_kg": float(kg.sum()),
        "avg_intensity": float(kg.sum() * 1000.0 / energy_kwh) if energy_kwh else float("nan"),
        "mean_delay_h": float(delay.mean()) if len(delay) else float("nan"),
        "p95_delay_h": float(np.percentile(delay, 95)) if len(delay) else float("nan"),
        "violations": int((s > a + sl).sum() + (~ok).sum()),
        "peak_kw": float(load.max()),
    }

def savings_pct(baseline_kg, other_kg):
    return 100.0 * (baseline_kg - other_kg) / baseline_kg

def coefficient_of_variation(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x) / np.mean(x))


def variability(ci):
    ci = pd.Series(ci).astype(float)
    daily = ci.groupby(ci.index.date)
    daily_cv = (daily.std(ddof=0) / daily.mean()).mean()
    daily_range = (daily.max() - daily.min()).mean()
    return {
        "mean": float(ci.mean()),
        "std": float(ci.std(ddof=0)),
        "cv": coefficient_of_variation(ci),
        "daily_cv_mean": float(daily_cv),
        "daily_range_mean": float(daily_range),
        "p10": float(ci.quantile(0.10)),
        "p90": float(ci.quantile(0.90)),
        "min": float(ci.min()),
        "max": float(ci.max()),
    }

def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[keep], y_pred[keep]

def mae(y_true, y_pred):
    t, p = _clean(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))

def rmse(y_true, y_pred):
    t, p = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))

def mape(y_true, y_pred, eps=1.0):
    t, p = _clean(y_true, y_pred)
    return float(100.0 * np.mean(np.abs(t - p) / np.maximum(np.abs(t), eps)))

def forecast_scores(y_true, y_pred):
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "MAPE_%": mape(y_true, y_pred)}