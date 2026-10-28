import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")

REGIONS = ["India", "UK"]
REGION_COLORS = {"India": "#D55E00", "UK": "#0072B2"}  

POLICY_LABELS = {
    "baseline": "Baseline (run now)",
    "threshold": "Threshold rule",
    "forecast_seasonal24": "Seasonal-24h guess",
    "forecast_seasonal168": "Seasonal-168h guess",
    "forecast_persistence": "Persistence",
    "forecast_oracle": "Oracle (perfect)",
}
POLICY_ORDER = ["Threshold rule", "Seasonal-24h guess", "Seasonal-168h guess",
                "ML forecast", "Oracle (perfect)"]

MODEL_ORDER = ["Persistence", "Seasonal naive 24h", "Seasonal naive 168h",
               "Moving average 24h", "ML forecast"]


def policy_label(p):
    if p in POLICY_LABELS:
        return POLICY_LABELS[p]
    if "gbm" in p or "lightgbm" in p:
        return "ML forecast"
    return p


def model_label(m):
    names = {"persistence": "Persistence", "seasonal_naive_24h": "Seasonal naive 24h",
             "seasonal_naive_168h": "Seasonal naive 168h",
             "moving_average_24h": "Moving average 24h", "lstm": "LSTM"}
    if m in names:
        return names[m]
    if "gbm" in m:
        return "ML forecast"
    return m


def _read_csv(path, **kw):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    return pd.read_csv(path, **kw)


def _save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure {name}")


def _save_table(df, name):
    df.to_csv(os.path.join(TAB_DIR, name))
    print(f"  saved table  {name}")


def load_series():
    out = {}
    for r in REGIONS:
        df = _read_csv(os.path.join(PROCESSED_DIR, f"{r.lower()}_intensity.csv"),
                       parse_dates=["timestamp"])
        if df is not None:
            out[r] = df.set_index("timestamp")["carbon_intensity"]
    return out

def fig_daily_profile(series):
    long = pd.concat([pd.DataFrame({"hour": s.index.hour, "ci": s.values, "region": r})
                      for r, s in series.items()])
    order = [r for r in REGIONS if r in series]
    g = sns.relplot(data=long, x="hour", y="ci", col="region", hue="region", kind="line",
                    errorbar="sd", palette=REGION_COLORS, legend=False, col_order=order,
                    hue_order=order, height=3.6, aspect=1.25,
                    facet_kws={"sharey": False})
    g.set_axis_labels("Hour of day (each grid's own clock)", "Carbon intensity (gCO2/kWh)")
    g.set_titles("{col_name}")
    g.figure.suptitle("Average daily profile (line = mean, band = ±1 std)", y=1.03)
    g.set(xticks=range(0, 24, 4))
    _save_fig(g.figure, "fig01_daily_profile.png")


def fig_distribution(series):
    long = pd.concat([pd.DataFrame({"ci": s.values, "region": r}) for r, s in series.items()])
    order = [r for r in REGIONS if r in series]
    g = sns.displot(data=long, x="ci", col="region", hue="region", kde=True, bins=50,
                    palette=REGION_COLORS, legend=False, col_order=order, hue_order=order,
                    height=3.6, aspect=1.25,
                    facet_kws={"sharex": False, "sharey": False})
    g.set_axis_labels("Carbon intensity (gCO2/kWh)", "Hours")
    g.set_titles("{col_name}")
    g.figure.suptitle("Distribution of hourly carbon intensity", y=1.03)
    _save_fig(g.figure, "fig02_distribution.png")


def fig_monthly_cv(series):
    rows = []
    for r, s in series.items():
        g = s.resample("MS")
        cv = (g.std() / g.mean()).dropna()
        rows.append(pd.DataFrame({"month": cv.index, "cv": cv.values, "region": r}))
    d = pd.concat(rows)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sns.lineplot(data=d, x="month", y="cv", hue="region", hue_order=[r for r in REGIONS if r in series],
                 palette=REGION_COLORS, marker="o", ax=ax)
    ax.set(xlabel="", ylabel="Coefficient of variation (std / mean)",
           title="Monthly variability of carbon intensity")
    ax.legend(title="")
    fig.autofmt_xdate()
    _save_fig(fig, "fig03_monthly_cv.png")


def fig_savings_by_policy(main):
    d = main[~main.policy.isin(["baseline", "forecast_persistence"])].copy()
    d["policy_label"] = d.policy.map(policy_label)
    order = [p for p in POLICY_ORDER if p in set(d.policy_label)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.barplot(data=d, x="policy_label", y="savings_pct", hue="region", order=order,
                hue_order=[r for r in REGIONS if r in set(d.region)],
                palette=REGION_COLORS, errorbar=None, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", padding=2, fontsize=8)
    ax.set(xlabel="", ylabel="Emission saving vs baseline (%)",
           title="Main scenario: 60% flexible jobs, 24 h slack")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(title="")
    _save_fig(fig, "fig04_savings_by_policy.png")


def _line_facets(d, x, xlabel, title, name, xticks=None):
    d = d.copy()
    d["policy_label"] = d.policy.map(policy_label)
    hue_order = [p for p in POLICY_ORDER if p in set(d.policy_label)]
    g = sns.relplot(data=d, x=x, y="savings_pct", hue="policy_label", style="policy_label",
                    hue_order=hue_order, style_order=hue_order, kind="line", markers=True,
                    dashes=False, col="region", col_order=[r for r in REGIONS if r in set(d.region)],
                    height=3.8, aspect=1.15, facet_kws={"sharey": False})
    g.set_axis_labels(xlabel, "Emission saving vs baseline (%)")
    g.set_titles("{col_name}")
    if xticks is not None:
        g.set(xticks=xticks)
    g._legend.set_title("")
    g.figure.suptitle(title, y=1.03)
    _save_fig(g.figure, name)


def fig_savings_vs_slack(sim):
    d = sim[(sim.flexible_fraction == 0.6) &
            sim.policy.isin(["threshold", "forecast_seasonal24", "forecast_oracle"])]
    _line_facets(d, "slack_hours", "Slack: how long a job may wait (hours)",
                 "Effect of slack (60% flexible jobs)", "fig05_savings_vs_slack.png",
                 sorted(d.slack_hours.unique()))


def fig_savings_vs_flex(sim):
    d = sim[(sim.slack_hours == 24) &
            sim.policy.isin(["threshold", "forecast_seasonal24", "forecast_oracle"])].copy()
    d["flexible_pct"] = (d.flexible_fraction * 100).round().astype(int)
    _line_facets(d, "flexible_pct", "Share of flexible jobs (%)",
                 "Effect of job flexibility (24 h slack)", "fig06_savings_vs_flexibility.png",
                 sorted(d.flexible_pct.unique()))


def fig_forecast_mae(fm):
    d = fm.copy()
    d["model_label"] = d.model.map(model_label)
    order = [m for m in MODEL_ORDER + ["LSTM"] if m in set(d.model_label)]
    g = sns.catplot(data=d, kind="bar", x="horizon_h", y="MAE", hue="model_label",
                    hue_order=order, col="region",
                    col_order=[r for r in REGIONS if r in set(d.region)],
                    errorbar=None, height=3.9, aspect=1.2, sharey=False)
    g.set_axis_labels("Forecast horizon (hours ahead)", "MAE (gCO2/kWh), lower is better")
    g.set_titles("{col_name}")
    g._legend.set_title("")
    g.figure.suptitle("Forecast error by model and horizon (test year 2023)", y=1.03)
    _save_fig(g.figure, "fig07_forecast_mae.png")


def fig_forecast_scheduling(fs):
    d = fs[fs.policy != "baseline"].copy()
    d["policy_label"] = d.policy.map(policy_label)
    order = [p for p in POLICY_ORDER if p in set(d.policy_label) and p != "Seasonal-168h guess"]
    g = sns.catplot(data=d, kind="bar", x="slack_hours", y="savings_pct", hue="policy_label",
                    hue_order=order, col="region",
                    col_order=[r for r in REGIONS if r in set(d.region)],
                    errorbar=None, height=3.9, aspect=1.2, sharey=False)
    g.set_axis_labels("Slack (hours)", "Emission saving vs baseline (%)")
    g.set_titles("{col_name}")
    g._legend.set_title("")
    g.figure.suptitle("Scheduling with forecasts (jobs arriving in 2023)", y=1.03)
    _save_fig(g.figure, "fig08_forecast_scheduling.png")





def table_variability(var):
    keep = ["mean", "std", "cv", "daily_cv_mean", "daily_range_mean", "p10", "p90", "min", "max"]
    t = var.set_index("region")[[c for c in keep if c in var.columns]].round(3)
    _save_table(t, "table1_grid_variability.csv")


def table_main_savings(main):
    d = main.copy()
    d["policy_label"] = d.policy.map(policy_label)
    parts = []
    for r in REGIONS:
        x = d[d.region == r].set_index("policy_label")[["savings_pct", "mean_delay_h", "violations"]]
        x.columns = [f"{r}: saving (%)", f"{r}: mean delay (h)", f"{r}: violations"]
        parts.append(x)
    t = pd.concat(parts, axis=1)
    order = [p for p in ["Baseline (run now)", "Threshold rule", "Seasonal-24h guess",
                         "Seasonal-168h guess", "Persistence", "Oracle (perfect)"] if p in t.index]
    _save_table(t.loc[order].round(2), "table2_main_savings.csv")


def table_forecast_mae(fm):
    d = fm.copy()
    d["model"] = d.model.map(model_label)
    t = d.pivot_table(index=["region", "model"], columns="horizon_h", values="MAE").round(2)
    t.columns = [f"{h} h ahead" for h in t.columns]
    _save_table(t, "table3_forecast_mae.csv")


def table_forecast_scheduling(fs):
    d = fs.copy()
    d["policy"] = d.policy.map(policy_label)
    t = d.pivot_table(index="policy", columns=["region", "slack_hours"], values="savings_pct").round(2)
    t.columns = [f"{r}, {s} h slack" for r, s in t.columns]
    order = [p for p in ["Baseline (run now)", "Threshold rule", "Seasonal-24h guess",
                         "ML forecast", "Oracle (perfect)"] if p in t.index]
    _save_table(t.loc[order], "table4_forecast_scheduling.csv")

    
def run(regions=None):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    print("\nFigures and tables -> results/figures, results/tables")
    series = load_series()
    if series:
        fig_daily_profile(series)
        fig_distribution(series)
        fig_monthly_cv(series)

    sim = _read_csv(os.path.join(RESULTS_DIR, "simulation_results.csv"))
    if sim is not None:
        main = sim[(sim.flexible_fraction == 0.6) & (sim.slack_hours == 24)]
        fig_savings_by_policy(main)
        fig_savings_vs_slack(sim)
        fig_savings_vs_flex(sim)
        table_main_savings(main)

    var = _read_csv(os.path.join(RESULTS_DIR, "grid_variability.csv"))
    if var is not None:
        table_variability(var)

    fm = _read_csv(os.path.join(RESULTS_DIR, "forecast_metrics.csv"))
    if fm is not None:
        fig_forecast_mae(fm)
        table_forecast_mae(fm)

    fs = _read_csv(os.path.join(RESULTS_DIR, "forecast_scheduling_results.csv"))
    if fs is not None:
        fig_forecast_scheduling(fs)
        table_forecast_scheduling(fs)
    print("\nDone.")


if __name__ == "__main__":
    run()