import math
import pandas as pd

def persistence(ci, horizon):
    return ci.copy()

def seasonal_naive(ci, horizon, period=24):
    k = math.ceil(horizon / period)
    return ci.shift(k * period - horizon)

def moving_average(ci, horizon, window=24):
    return ci.rolling(window).mean()

def all_baselines(ci, horizon):
    return {
        "persistence": persistence(ci, horizon),
        "seasonal_naive_24h": seasonal_naive(ci, horizon, 24),
        "seasonal_naive_168h": seasonal_naive(ci, horizon, 168),
        "moving_average_24h": moving_average(ci, horizon, 24),
    }