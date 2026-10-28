import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "india_monthly"
CACHE = ROOT / "data" / "processed" / "india_hourly_cache"
OUT_MIX = ROOT / "data" / "raw" / "india_mix_hourly.csv"
OUT_VRE = ROOT / "data" / "processed" / "india_2024_2025_vre_hourly.csv"

PATTERNS = {
    "thermal_mw": r"THM",
    "hydro_mw": r"HYD",
    "gas_mw": r"GAS",
    "nuclear_mw": r"NPC|NUC",
    "wind_mw": r"ALL_IND\w*_WIND",
    "solar_mw": r"ALL_IND\w*_SOLAR",
    "demand_mw": r"NLDC_DEMAND",
}
COMPONENTS = ["thermal_mw", "hydro_mw", "gas_mw", "nuclear_mw", "wind_mw", "solar_mw"]
FACTORS = {"thermal_mw": 820, "gas_mw": 490, "hydro_mw": 24, "nuclear_mw": 12,
           "wind_mw": 11, "solar_mw": 45}  # gCO2/kWh, same as config.py

def read_mix_month(xl):
    sheets = [s for s in xl.sheet_names if not s.startswith("_")]
    sheet = "Sheet1" if "Sheet1" in sheets else sheets[0]
    df = xl.parse(sheet)
    labels = df.iloc[0]
    data = df.iloc[1:].copy()
    data.index = pd.to_datetime(data[df.columns[0]], errors="coerce")
    data = data[data.index.notna()]

    found = {}
    for col in df.columns[1:]:
        label = str(labels[col]).strip()
        for name, pattern in PATTERNS.items():
            if name not in found and re.search(pattern, label, re.I):
                found[name] = col
        if label.lower() == "total" and "total_mw" not in found:
            found["total_mw"] = col

    out = pd.DataFrame(index=data.index)
    for name in list(PATTERNS) + ["total_mw"]:
        if name in found:
            out[name] = pd.to_numeric(data[found[name]], errors="coerce")
        else:
            out[name] = float("nan")
            print(f"    WARNING: no column found for {name}")
    out = out[~out.index.duplicated(keep="first")]
    return out.resample("1h").mean()

def read_vre_file(xl):
    df = xl.parse("Report")
    time_col = next(c for c in df.columns if "time" in str(c).lower())
    out = pd.DataFrame(index=pd.to_datetime(df[time_col], errors="coerce", dayfirst=True))
    for name, key in [("demand_mw", "demand"), ("wind_mw", "wind"),
                      ("solar_mw", "solar"), ("total_mw", "total")]:
        col = next((c for c in df.columns if key in str(c).lower()), None)
        out[name] = pd.to_numeric(df[col], errors="coerce").to_numpy() if col else float("nan")
    out = out[out.index.notna()]
    out = out[~out.index.duplicated(keep="first")]
    return out.resample("1h").mean()

def main():
    files = sorted(list(SRC.glob("*.xlsx")) + list(SRC.glob("*.xls")))
    if not files:
        print(f"No Excel files in {SRC}")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_MIX.parent.mkdir(parents=True, exist_ok=True)

    mix_parts, vre_parts = [], []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        cached = list(CACHE.glob(f"*__{f.stem}.csv"))
        if cached:
            kind = cached[0].name.split("__")[0]
            part = pd.read_csv(cached[0], index_col=0, parse_dates=True)
            print("    loaded from cache")
        else:
            xl = pd.ExcelFile(f)
            kind = "vre" if "Report" in xl.sheet_names else "mix"
            part = read_vre_file(xl) if kind == "vre" else read_mix_month(xl)
            part.to_csv(CACHE / f"{kind}__{f.stem}.csv")
            print(f"    {len(part)} hourly rows, {part.index.min()} -> {part.index.max()}")
        (vre_parts if kind == "vre" else mix_parts).append(part)

    if mix_parts:
        mix = pd.concat(mix_parts).sort_index()
        mix = mix[~mix.index.duplicated(keep="first")]
        mix = mix.dropna(subset=["thermal_mw", "total_mw"])
        mix.index.name = "timestamp"
        mix.to_csv(OUT_MIX)

        hours = int((mix.index.max() - mix.index.min()) / pd.Timedelta(hours=1)) + 1
        print(f"\nMIX FILE: {OUT_MIX}")
        print(f"  range: {mix.index.min()} -> {mix.index.max()}")
        print(f"  rows: {len(mix)} | expected hours: {hours} | missing hours: {hours - len(mix)}")
        gap = (mix[COMPONENTS].sum(axis=1) - mix["total_mw"]).abs().mean()
        print(f"  mean |sum of sources - total|: {gap:.0f} MW")
        shares = mix[COMPONENTS].mean() / mix[COMPONENTS].mean().sum() * 100
        print("  average generation share (%):", shares.round(1).to_dict())
        ci = sum(mix[c] * FACTORS[c] for c in COMPONENTS) / mix[COMPONENTS].sum(axis=1)
        print(f"  estimated carbon intensity: mean {ci.mean():.0f}, min {ci.min():.0f}, "
              f"max {ci.max():.0f} gCO2/kWh")

    if vre_parts:
        vre = pd.concat(vre_parts).sort_index()
        vre = vre[~vre.index.duplicated(keep="first")]
        vre.index.name = "timestamp"
        vre.to_csv(OUT_VRE)
        print(f"\nVRE FILE: {OUT_VRE}")
        print(f"  range: {vre.index.min()} -> {vre.index.max()} | rows: {len(vre)}")


if __name__ == "__main__":
    main()