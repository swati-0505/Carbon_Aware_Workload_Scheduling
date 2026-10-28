import argparse
import os
import time
import numpy as np
import pandas as pd
from src.scheduling import baseline_scheduler, job_generator, rule_based_scheduler
from src.utils import metrics

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
REGION_NAMES = {"india": "India", "uk": "UK", "gb": "UK"}
DEFAULT_REGIONS = ["India", "UK"]
MAIN_SCENARIO = (0.6, 24)
ALL_SCENARIOS = [(0.6, 24), (0.6, 6), (0.6, 12), (0.6, 48),
                 (0.2, 24), (0.4, 24), (0.8, 24)]
POLICIES = [
    ("threshold", "threshold", None),
    ("forecast_oracle", "forecast", "oracle"),
    ("forecast_seasonal24", "forecast", "seasonal24"),
    ("forecast_seasonal168", "forecast", "seasonal168"),
    ("forecast_persistence", "forecast", "persistence"),
]

def normalise_regions(regions):
    if not regions:
        return list(DEFAULT_REGIONS)
    if isinstance(regions, str):
        regions = [regions]
    out = []
    for r in regions:
        key = str(r).strip().lower()
        if key not in REGION_NAMES:
            raise ValueError(f"unknown region '{r}'. Use: India, UK")
        name = REGION_NAMES[key]
        if name not in out:
            out.append(name)
    return out


def load_intensity(region):
    path = os.path.join(PROCESSED_DIR, f"{region.lower()}_intensity.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python main.py clean")
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    s = df.set_index("timestamp")["carbon_intensity"].astype(float)
    steps = s.index.to_series().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(hours=1) or s.isna().any():
        raise ValueError(
            f"{path} is not a complete hourly series (gaps or NaN). "
            "Run: python src\\data\\fix_gaps.py")
    return s

def run(regions=None, quick=False, jobs_per_hour=1.0, headroom=1.25, seed=42):
    regions = normalise_regions(regions)
    t0 = time.time()
    series = {r: load_intensity(r) for r in regions}
    n_hours = min(len(s) for s in series.values())
    ci = {r: s.values[:n_hours] for r, s in series.items()}
    print(f"\nSimulation | regions: {', '.join(regions)} | hours: {n_hours}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    var_rows = []
    for r in regions:
        v = metrics.variability(series[r].iloc[:n_hours])
        v["region"] = r
        var_rows.append(v)
    var_df = pd.DataFrame(var_rows).set_index("region")
    var_df.round(3).to_csv(os.path.join(RESULTS_DIR, "grid_variability.csv"))
    print("\nGrid variability (gCO2/kWh):")
    print(var_df[["mean", "std", "cv", "daily_cv_mean", "daily_range_mean"]]
          .round(3).to_string())

    scenarios = [MAIN_SCENARIO] if quick else ALL_SCENARIOS
    rows = []
    for flex, slack in scenarios:
        ts = time.time()
        jobs = job_generator.generate_jobs(
            n_hours, jobs_per_hour=jobs_per_hour, flexible_fraction=flex,
            slack_hours=slack, seed=seed)
        capacity = headroom * baseline_scheduler.immediate_load(jobs, n_hours).max()
        base_starts, _ = baseline_scheduler.schedule(jobs, n_hours, capacity)
        if not np.array_equal(base_starts, jobs["arrival"].values):
            print("  warning: baseline had to queue some jobs (capacity too tight)")

        for r in regions:
            base = metrics.summarize(jobs, base_starts, ci[r])
            rows.append(_row(r, flex, slack, "baseline", base, base, capacity))
            for label, policy, kind in POLICIES:
                starts, _ = rule_based_scheduler.schedule(
                    jobs, ci[r], capacity, policy=policy,
                    forecast_kind=kind or "oracle", seed=seed)
                res = metrics.summarize(jobs, starts, ci[r])
                rows.append(_row(r, flex, slack, label, res, base, capacity))
        print(f"  scenario flex={flex}, slack={slack}h done "
              f"({len(jobs)} jobs, {time.time() - ts:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "simulation_results.csv"), index=False)

    main = out[(out.flexible_fraction == MAIN_SCENARIO[0]) &
               (out.slack_hours == MAIN_SCENARIO[1])]
    print(f"\nMain scenario: {int(MAIN_SCENARIO[0] * 100)}% flexible jobs, "
          f"{MAIN_SCENARIO[1]} h slack")
    table = main.pivot(index="policy", columns="region", values="savings_pct")
    order = ["baseline"] + [p[0] for p in POLICIES]
    print("Emission saving vs baseline (%):")
    print(table.loc[order].round(2).to_string())
    viol = main.pivot(index="policy", columns="region", values="violations")
    print("\nDeadline violations (jobs):")
    print(viol.loc[order].to_string())
    print(f"\nSaved: {RESULTS_DIR}/simulation_results.csv, "
          f"{RESULTS_DIR}/grid_variability.csv  ({time.time() - t0:.0f}s total)")
    return out

def _row(region, flex, slack, policy, res, base, capacity):
    return {
        "region": region,
        "flexible_fraction": flex,
        "slack_hours": slack,
        "policy": policy,
        "emissions_kg": round(res["emissions_kg"], 1),
        "baseline_kg": round(base["emissions_kg"], 1),
        "savings_pct": round(metrics.savings_pct(base["emissions_kg"], res["emissions_kg"]), 3),
        "avg_intensity": round(res["avg_intensity"], 2),
        "mean_delay_h": round(res["mean_delay_h"], 2),
        "p95_delay_h": round(res["p95_delay_h"], 1),
        "violations": res["violations"],
        "peak_kw": round(res["peak_kw"], 1),
        "capacity_kw": round(capacity, 1),
        "n_jobs": res["n_jobs"],
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(args.regions, quick=args.quick)