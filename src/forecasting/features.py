import numpy as np
import pandas as pd

LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 168]

def make_features(ci, horizon):
    ci = ci.astype(float)
    X = pd.DataFrame(index=ci.index)
    for k in LAGS:
        X[f"lag_{k}"] = ci.shift(k)
    X["roll_mean_24"] = ci.rolling(24).mean()
    X["roll_std_24"] = ci.rolling(24).std()
    X["roll_mean_168"] = ci.rolling(168).mean()
    X["trend_6"] = ci - ci.shift(6)
    k24 = -(-horizon // 24) * 24 - horizon
    k168 = -(-horizon // 168) * 168 - horizon
    X["same_hour_yesterday"] = ci.shift(k24)
    X["same_hour_last_week"] = ci.shift(k168)

    target_time = ci.index + pd.Timedelta(hours=horizon)
    hour = target_time.hour.values
    dow = target_time.dayofweek.values
    month = target_time.month.values
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    return X

def make_supervised(ci, horizon):
    X = make_features(ci, horizon)
    y = ci.shift(-horizon).rename("y")
    data = X.join(y).dropna()
    return data.drop(columns="y"), data["y"]