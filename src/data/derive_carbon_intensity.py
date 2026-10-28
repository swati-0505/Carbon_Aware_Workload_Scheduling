import pandas as pd
from src.utils import config

SKIP_WORDS = ["total", "demand", "emission", "intensity", "forecast", "capacity"]

def _factor_for(column):
    name = column.lower()
    if any(w in name for w in SKIP_WORDS):
        return None
    for key, factor in config.EMISSION_FACTORS.items():
        if key in name:
            return factor
    return None

def derive_region(region):
    path = config.DATA_PROCESSED / f"{region.lower()}_clean.csv"
    if not path.exists():
        print(f"  [{region}] {path.name} not found. Run: python main.py clean")
        return False

    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")

    if "carbon_intensity" in df.columns:
        series = df["carbon_intensity"]
        print(f"  [{region}] using existing carbon_intensity column")
    else:
        factors = {c: _factor_for(c) for c in df.columns}
        matched = {c: f for c, f in factors.items() if f is not None}
        if not matched:
            print(f"  [{region}] no generation columns matched emission factors. "
                  f"Columns: {list(df.columns)}")
            return False
        print(f"  [{region}] deriving from columns: {matched}")
        gen = df[list(matched)].clip(lower=0)
        total = gen.sum(axis=1)
        weighted = sum(gen[c] * f for c, f in matched.items())
        series = (weighted / total).where(total > 0)

    series = series.interpolate(limit=6).dropna()
    out = config.DATA_PROCESSED / f"{region.lower()}_intensity.csv"
    series.rename("carbon_intensity").to_csv(out, header=True)
    print(f"  [{region}] saved {out.name}: {len(series)} rows | "
          f"mean {series.mean():.0f}, min {series.min():.0f}, max {series.max():.0f} gCO2/kWh")
    return True

def run(regions=None):
    config.ensure_dirs()
    for region in regions or config.REGIONS:
        derive_region(region)

if __name__ == "__main__":
    run()