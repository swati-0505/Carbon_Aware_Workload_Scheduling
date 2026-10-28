import argparse
import os
import time
import numpy as np
import pandas as pd
from src.forecasting import features
from src.scheduling import baseline_scheduler, job_generator, rule_based_scheduler
from src.scheduling.baseline_scheduler import earliest_feasible, feasible_starts
from src.utils import metrics

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
TRAIN_END = pd.Timestamp("2022-12-31 23:00")
REGION_NAMES = {"india": "India", "uk": "UK", "gb": "UK"}
HORIZON_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36, 42, 48]
MAX_H = 48                    
SCENARIOS = [(0.6, 24), (0.6, 12), (0.6, 6)]   

def _load(region):
    path = os.path.join(PROCESSED_DIR, f"{region.lower()}_intensity.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python main.py clean")
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    s = df.set_index("timestamp")["carbon_intensity"].astype(float)
    if s.isna().any() or len(s.index.to_series().diff().dropna().unique()) != 1:
        raise ValueError(f"{path} has gaps. Run: python src\\data\\fix_gaps.py")
    return s

def _make_model():
    try:
        import lightgbm as lgb
        return "lightgbm", lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return "hist_gbm(sklearn)", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, random_state=0)

def build_forecasts(ci, train_end, pos):
    n = len(ci)
    P = np.full((n, len(HORIZON_GRID)), np.nan)
    name = None
    for j, h in enumerate(HORIZON_GRID):
        X, y = features.make_supervised(ci, h)
        train = X.index <= train_end - pd.Timedelta(hours=h) 
        name, model = _make_model()
        model.fit(X[train], y[train])
        Xo = features.make_features(ci, h).iloc[pos:].dropna()
        P[ci.index.get_indexer(Xo.index), j] = model.predict(Xo)
        print(f"      horizon {h:2d} h trained", end="\r")
    print(" " * 40, end="\r")

    hs = np.arange(1, MAX_H + 1)
    F = np.full((n, MAX_H), np.nan)
    for t in range(pos, n):
        if not np.isnan(P[t]).any():
            F[t] = np.maximum(np.interp(hs, HORIZON_GRID, P[t]), 1.0)
    return F, name

def schedule_with_forecast(jobs, ci, capacity_kw, F):
    n = len(ci)
    load = np.zeros(n)
    arr = jobs["arrival"].values
    dur = jobs["duration"].values
    pw = jobs["power_kw"].values
    sl = jobs["slack"].values
    starts = np.full(len(jobs), -1, dtype=int)

    for i in range(len(jobs)):
        a, d, p, s = int(arr[i]), int(dur[i]), pw[i], int(sl[i])
        if s == 0:
            st = earliest_feasible(load, capacity_kw, p, d, a)
        else:
            if s + d > MAX_H:
                raise ValueError(f"slack {s} + duration {d} exceeds MAX_H={MAX_H}")
            cand = feasible_starts(load, capacity_kw, p, d, a, a + s)
            if len(cand) == 0:
                st = earliest_feasible(load, capacity_kw, p, d, a + s + 1)
            else:
                f = F[a - 1][:s + d]
                if np.isnan(f).any():
                    raise ValueError("forecast missing for an arrival hour")
                cs = np.concatenate([[0.0], np.cumsum(f)])
                rel = cand - a
                cost = cs[rel + d] - cs[rel]
                best = np.flatnonzero(cost <= cost.min() + 1e-6)[0]
                st = int(cand[best])
        if st >= 0:
            load[st:st + d] += p
            starts[i] = st
    return starts, load

def run(regions=None, headroom=1.25, seed=42):
    if not regions:
        regions = ["India", "UK"]
    if isinstance(regions, str):
        regions = [regions]
    regions = [REGION_NAMES[str(r).strip().lower()] for r in regions]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for region in regions:
        t0 = time.time()
        s = _load(region)
        train_end = TRAIN_END
        if s.index[-1] <= TRAIN_END + pd.Timedelta(days=90):
            train_end = s.index[int(0.8 * len(s))]
        pos = s.index.get_loc(train_end)

        print(f"\n[{region}] training forecast models on data up to {train_end} ...")
        F, model_name = build_forecasts(s, train_end, pos)
        gbm_label = f"forecast_{model_name}"

        offset = pos + 1 - 168
        ci_test = s.values[offset:]
        F_test = F[offset:]
        print(f"  test hours: {len(ci_test) - 168}  model: {model_name}  "
              f"({time.time() - t0:.0f}s)")

        for flex, slack in SCENARIOS:
            jobs = job_generator.generate_jobs(
                len(ci_test), flexible_fraction=flex, slack_hours=slack, seed=seed)
            cap = headroom * baseline_scheduler.immediate_load(jobs, len(ci_test)).max()
            base_starts, _ = baseline_scheduler.schedule(jobs, len(ci_test), cap)
            res = {"baseline": metrics.summarize(jobs, base_starts, ci_test)}
            for label, pol, kind in [("threshold", "threshold", "oracle"),
                                     ("forecast_seasonal24", "forecast", "seasonal24"),
                                     ("forecast_oracle", "forecast", "oracle")]:
                st, _ = rule_based_scheduler.schedule(
                    jobs, ci_test, cap, policy=pol, forecast_kind=kind, seed=seed)
                res[label] = metrics.summarize(jobs, st, ci_test)
            st, _ = schedule_with_forecast(jobs, ci_test, cap, F_test)
            res[gbm_label] = metrics.summarize(jobs, st, ci_test)

            base_kg = res["baseline"]["emissions_kg"]
            sav = {k: metrics.savings_pct(base_kg, v["emissions_kg"]) for k, v in res.items()}
            gap = sav["forecast_oracle"] - sav["forecast_seasonal24"]
            closed = 100.0 * (sav[gbm_label] - sav["forecast_seasonal24"]) / gap if gap > 0 else float("nan")

            print(f"\n  {region} | {int(flex * 100)}% flexible, {slack} h slack | "
                  f"{len(jobs)} jobs")
            for label, v in res.items():
                print(f"    {label:24s} saving {sav[label]:6.2f}%   "
                      f"mean delay {v['mean_delay_h']:5.2f} h   violations {v['violations']}")
                rows.append({"region": region, "flexible_fraction": flex,
                             "slack_hours": slack, "policy": label,
                             "savings_pct": round(sav[label], 3),
                             "emissions_kg": round(v["emissions_kg"], 1),
                             "mean_delay_h": round(v["mean_delay_h"], 2),
                             "violations": v["violations"], "n_jobs": v["n_jobs"]})
            print(f"    share of the (seasonal24 -> oracle) gap closed by "
                  f"{model_name}: {closed:.0f}%")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "forecast_scheduling_results.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/forecast_scheduling_results.csv")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None)
    args = ap.parse_args()
    run(args.regions)