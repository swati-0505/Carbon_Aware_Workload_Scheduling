import argparse
import os
import numpy as np
import pandas as pd
from src.utils import metrics
from . import baselines, features, lstm_model

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
HORIZONS = [1, 6, 24]
TRAIN_END = pd.Timestamp("2022-12-31 23:00")
REGION_NAMES = {"india": "India", "uk": "UK", "gb": "UK"}


def _load(region):
    path = os.path.join(PROCESSED_DIR, f"{region.lower()}_intensity.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python main.py clean")
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    s = df.set_index("timestamp")["carbon_intensity"].astype(float)
    steps = s.index.to_series().diff().dropna().unique()
    if len(steps) != 1 or s.isna().any():
        raise ValueError(f"{path} has gaps. Run: python src\\data\\fix_gaps.py")
    return s


def split_point(ci):
    if ci.index[-1] > TRAIN_END + pd.Timedelta(days=90):
        return TRAIN_END
    return ci.index[int(0.8 * len(ci))]


def gbm_forecast(ci, horizon, train_end):
    """Gradient boosting, one model per horizon. Returns (name, Series) or None."""
    X, y = features.make_supervised(ci, horizon)
    train = X.index <= train_end - pd.Timedelta(hours=horizon)   # targets inside train period
    test = X.index > train_end
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=0, verbose=-1)
        name = "lightgbm"
    except ImportError:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                  random_state=0)
            name = "hist_gbm(sklearn)"
        except ImportError:
            return None
    model.fit(X[train], y[train])
    return name, pd.Series(model.predict(X[test]), index=X.index[test])


def evaluate_region(region, use_lstm=True):
    ci = _load(region)
    train_end = split_point(ci)
    print(f"\n[{region}] {len(ci)} hours | train up to {train_end} | "
          f"test {(ci.index > train_end).sum()} hours")

    lstm_preds = {}
    if use_lstm and lstm_model.available():
        print("    training LSTM (this can take a few minutes)...")
        lstm_preds = lstm_model.fit_predict(ci, HORIZONS, train_end)
    elif use_lstm:
        print("    PyTorch not installed: LSTM skipped (pip install torch to include it)")

    rows = []
    for h in HORIZONS:
        preds = baselines.all_baselines(ci, h)
        gbm = gbm_forecast(ci, h, train_end)
        if gbm:
            preds[gbm[0]] = gbm[1]
        else:
            print("    no gradient-boosting library found (pip install lightgbm)")
        if h in lstm_preds:
            preds["lstm"] = lstm_preds[h]

        frame = pd.DataFrame(preds)
        frame["y_true"] = ci.shift(-h)
        frame = frame[frame.index > train_end].dropna()      # same rows for every model
        for model in preds:
            sc = metrics.forecast_scores(frame["y_true"], frame[model])
            rows.append({"region": region, "horizon_h": h, "model": model,
                         "n_test": len(frame), **sc})
    res = pd.DataFrame(rows)
    base = res[res.model == "persistence"].set_index("horizon_h")["MAE"]
    res["MAE_vs_persistence_%"] = (
        100 * (res["horizon_h"].map(base) - res["MAE"]) / res["horizon_h"].map(base))
    return res


def run(regions=None, use_lstm=True):
    if not regions:
        regions = ["India", "UK"]
    if isinstance(regions, str):
        regions = [regions]
    regions = [REGION_NAMES[str(r).strip().lower()] for r in regions]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_res = []
    for r in regions:
        res = evaluate_region(r, use_lstm)
        all_res.append(res)
        print(f"\n  MAE (gCO2/kWh), {r}:")
        print(res.pivot(index="model", columns="horizon_h", values="MAE").round(2).to_string())
    out = pd.concat(all_res, ignore_index=True)
    out.round(3).to_csv(os.path.join(RESULTS_DIR, "forecast_metrics.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/forecast_metrics.csv")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--no-lstm", action="store_true")
    args = ap.parse_args()
    run(args.regions, use_lstm=not args.no_lstm)