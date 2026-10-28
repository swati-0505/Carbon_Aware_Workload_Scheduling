import os
import shutil
import pandas as pd

PROCESSED = os.path.join("data", "processed")
FILES = ["india_intensity.csv", "uk_intensity.csv"]
START = pd.Timestamp("2021-09-01 00:00")
END = pd.Timestamp("2023-12-31 23:00")
BAD_BELOW = 5.0  

def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

def fix(name):
    path = os.path.join(PROCESSED, name)
    if not os.path.exists(path):
        print(f"[{name}] NOT FOUND at {path} - skipped")
        return

    backup = path.replace(".csv", "_before_fix.csv")
    if not os.path.exists(backup):
        shutil.copy(path, backup)

    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    num_cols = [c for c in df.columns if c != ts_col
                and pd.api.types.is_numeric_dtype(df[c])]
    val_col = "carbon_intensity" if "carbon_intensity" in df.columns else num_cols[-1]
    extra = [c for c in df.columns if c not in (ts_col, val_col)]
    if extra:
        print(f"[{name}] note: dropping extra columns {extra}")

    ts = pd.to_datetime(df[ts_col])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    s = pd.Series(df[val_col].values, index=ts).sort_index()
    s = s.groupby(level=0).mean()  # collapse duplicate timestamps
    rows_in = len(s)

    s = s[(s.index >= START) & (s.index <= END)]
    bad = int((s < BAD_BELOW).sum())
    s[s < BAD_BELOW] = float("nan")

    full = pd.date_range(START, END, freq="h")
    s = s.reindex(full)
    missing = s.isna()
    n_missing = int(missing.sum())
    gap = longest_run(missing.values)

    s = s.interpolate(method="time", limit_direction="both")

    out = pd.DataFrame({ts_col: s.index, val_col: s.values.round(2)})
    out.to_csv(path, index=False)

    print(f"[{name}] column '{val_col}'")
    print(f"   rows in: {rows_in} -> rows out: {len(out)} "
          f"({START} -> {END})")
    print(f"   bad readings (<{BAD_BELOW}) set to missing: {bad}")
    print(f"   missing hours filled by interpolation: {n_missing} "
          f"({100 * n_missing / len(out):.2f}%), longest gap: {gap} h")
    print(f"   now: mean {s.mean():.0f}, min {s.min():.0f}, "
          f"max {s.max():.0f} gCO2/kWh")
    print(f"   backup: {backup}")

if __name__ == "__main__":
    for f in FILES:
        fix(f)