import pandas as pd
from src.utils import config

def _find_files(region):
    keywords = config.REGION_KEYWORDS.get(region, [region.lower()])
    files = []
    for f in sorted(config.DATA_RAW.glob("*")):
        if f.suffix.lower() not in (".csv", ".xlsx", ".xls"):
            continue
        if any(k in f.name.lower() for k in keywords):
            files.append(f)
    return files

def _read(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _detect_time(df):
    lower = {c.lower().strip(): c for c in df.columns}

    if "year" in lower and "month" in lower:
        try:
            t = pd.to_datetime(
                {"year": df[lower["year"]], "month": df[lower["month"]], "day": 1},
                errors="coerce",
            )
            if t.notna().mean() > 0.8:
                return t, [lower["year"], lower["month"]]
        except Exception:
            pass

    preferred = ["timestamp", "datetime", "time", "date", "from", "start", "period"]
    ordered = [c for c in df.columns if any(p in c.lower() for p in preferred)]
    ordered += [c for c in df.columns if c not in ordered]

    for col in ordered:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        t = pd.to_datetime(df[col], errors="coerce", utc=True)
        if t.notna().mean() > 0.8:
            return t.dt.tz_convert(None), [col]
    return None, []


def _detect_intensity(columns):
    cols = list(columns)
    low = {c: c.lower() for c in cols}
    for key in ["actual", "carbon_intensity", "carbon intensity", "gco2", "co2 intensity", "intensity"]:
        for c in cols:
            if key in low[c]:
                return c
    return None


def clean_region(region):
    files = _find_files(region)
    if not files:
        print(f"  [{region}] no files found in data/raw (name must contain: "
              f"{config.REGION_KEYWORDS.get(region)})")
        return False

    frames = []
    for f in files:
        df = _read(f)
        t, used = _detect_time(df)
        if t is None:
            print(f"  [{region}] {f.name}: could not find a time column. Columns: {list(df.columns)}")
            continue
        df = df.drop(columns=used)
        df.insert(0, "timestamp", t)
        frames.append(df)
        print(f"  [{region}] loaded {f.name}: {len(df)} rows")

    if not frames:
        return False

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["timestamp"])

    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    low = {c.lower(): c for c in df.columns}
    if "actual" in low and "forecast" in low:
        df[low["actual"]] = df[low["actual"]].fillna(df[low["forecast"]])

    step = df.index.to_series().diff().median()
    if pd.notna(step) and step > pd.Timedelta(days=1):
        print(f"  [{region}] WARNING: data is {step.days}-day resolution (monthly/yearly). "
              "Not suitable for hourly scheduling; saved as-is.")
    else:
        df = df.resample("1h").mean()
        df = df.interpolate(limit=3)

    intensity_col = _detect_intensity(df.columns)
    if intensity_col is not None:
        df = df.rename(columns={intensity_col: "carbon_intensity"})
        print(f"  [{region}] carbon intensity column: {intensity_col}")
    else:
        print(f"  [{region}] no carbon intensity column; will derive from generation mix")

    out = config.DATA_PROCESSED / f"{region.lower()}_clean.csv"
    df.to_csv(out)
    print(f"  [{region}] saved {out.name}: {len(df)} rows, "
          f"{df.index.min()} -> {df.index.max()}")
    return True


def run(regions=None):
    config.ensure_dirs()
    for region in regions or config.REGIONS:
        clean_region(region)


if __name__ == "__main__":
    run()