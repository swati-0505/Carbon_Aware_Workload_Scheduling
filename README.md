# Carbon-Aware Workload Scheduling: India vs UK

Data centres and cloud workloads consume large amounts of electricity, and the carbon intensity of that electricity (gCO2/kWh) changes hour by hour depending on which power plants are running. Many jobs, such as model training, batch analytics and backups, do not have to run immediately. If such jobs are delayed to cleaner hours, the same work produces less CO2.

This project studies carbon-aware workload scheduling on two very different grids: the **Indian grid**, which depends heavily on thermal (coal) generation, and the **UK grid**, which has a lower and much more variable carbon intensity driven by wind. Using hourly data from September 2021 to December 2023, the project simulates a cloud job trace under several scheduling policies and compares their emissions with a carbon-unaware baseline. It also builds forecasting models for carbon intensity and tests whether better forecasts lead to better scheduling decisions.

## Research questions

1. How much CO2 can carbon-aware scheduling save on a grid that is almost always carbon-intensive (India) compared with a highly variable, lower-carbon grid (UK)?
2. Do savings grow with the variability of the grid (coefficient of variation, CV)?
3. How much does forecast quality matter, and how much of the gap between a simple guess and perfect knowledge can a machine-learning forecast close?

## What is in the pipeline

| Stage | What it does | Main files |
|---|---|---|
| Data | Builds an hourly India generation-mix file, downloads UK data, cleans both, derives India carbon intensity from the generation mix | `src/data/` |
| Gap fixing | Trims both series to the same window, replaces bad readings, interpolates missing hours | `src/data/fix_gaps.py` |
| Simulation | Schedules the same job trace on both grids with several policies and compares emissions with a baseline | `src/simulation/simulator.py`, `src/scheduling/` |
| Forecasting | Compares simple baselines with gradient boosting (LightGBM) at 1, 6 and 24 hours ahead | `src/forecasting/` |
| Forecast-informed scheduling | Uses the ML forecast inside the scheduler | `src/simulation/forecast_scheduling.py` |
| Figures and tables | Produces the figures and tables for the report | `src/analysis/make_figures.py` |

### Scheduling policies

All policies see the same jobs and the same cluster capacity. Only flexible jobs may wait, and never longer than their slack.

- **Baseline**: start every job as soon as it arrives (first come, first served).
- **Threshold rule**: start when the current carbon intensity is in the cleanest 30% of the previous 168 hours (uses only past data).
- **Forecast-based**: start in the window hour with the lowest forecast carbon intensity over the job's run time. Forecasts: `seasonal24` (same hour yesterday), `seasonal168` (same hour last week), `persistence` (no skill, used as a sanity check), `oracle` (perfect knowledge, upper bound), and LightGBM.
- **Reinforcement learning** (optional extension): a tabular Q-learning scheduler in `src/scheduling/rl_scheduler.py`.

Emissions are always computed with the true carbon intensity, never with the forecast.

### Job model (synthetic)

About one job per hour (Poisson arrivals, busier in working hours), duration 1 to 24 hours, power 5 to 80 kW. Default scenario: 60% of jobs flexible with 24 hours of slack. The cluster capacity is 1.25 times the peak load of the baseline. The same trace (same seed) is used on both grids.

## Data

Data files are **not** stored in this repository (see `.gitignore`).

- **India**: hourly generation by source (thermal, hydro, gas, nuclear, wind, solar) from Grid-India monthly files, Mendeley Data. Mukherjee, Kalita and Kumar (2025), Mendeley Data V2, DOI: 10.17632/y58jknpgs8.2. Place the monthly Excel files in `data/india_monthly/`. The main study uses September 2021 to December 2023, where the full generation mix is available. The January 2024 to June 2025 file has only demand, wind, solar and total, so it is kept for later robustness checks.
- **UK**: half-hourly carbon intensity from the Carbon Intensity API (`api.carbonintensity.org.uk`), downloaded by `src/data/fetch_uk.py`.

India carbon intensity is **estimated** from the generation mix using fixed emission factors per source (gCO2/kWh): thermal 820, gas 490, hydro 24, nuclear 12, wind 11, solar 45. It is not a measured value. India timestamps are in IST and UK timestamps in UTC; each grid is simulated on its own clock.

Please check the licence terms of the dataset and of the API before redistributing any data.

## Setup

Python 3.11 on Windows (PowerShell) was used.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Main packages: `pandas`, `numpy`, `openpyxl`, `requests`, `scikit-learn`, `lightgbm`, `matplotlib`, `seaborn`. `torch` is optional (only for the LSTM model; it is skipped if not installed).

## How to run

Run everything from the project root, with the virtual environment active.

```
# 1. India: monthly Excel files -> hourly mix file
python src\data\build_india_hourly.py

# 2. UK: download the same dates as India
python src\data\fetch_uk.py --start 2021-09-01 --end 2023-12-31
copy data\uk_raw\uk_carbon_intensity.csv data\raw\uk_carbon_intensity.csv

# 3. Clean both grids and derive carbon intensity
python main.py clean

# 4. Fill missing hours, remove bad readings (needed before simulating)
python src\data\fix_gaps.py

# 5. Scheduling simulation
python main.py simulate

# 6. Forecast evaluation (add --no-lstm if PyTorch is not installed)
python -m src.forecasting.evaluate_forecast --no-lstm

# 7. Scheduling with the ML forecast
python -m src.simulation.forecast_scheduling

# 8. Figures and tables
python -m src.analysis.make_figures
```

Optional: `python -m src.scheduling.rl_scheduler` (Q-learning) and `src/data/fetch_electricitymaps.py` (needs an API token, not used in the main study).

## Project structure

```
.
├── data/              input data (not in the repository)
├── notebooks/         data_cleaning, eda_patterns, forecaster_training, scheduler_evaluation
├── report/            abstract, literature review, final report
├── results/
│   ├── figures/       figures (PNG)
│   ├── tables/        tables (CSV)
│   └── *.csv          raw outputs of the experiments
├── src/
│   ├── data/          build_india_hourly, fetch_uk, clean_data, derive_carbon_intensity, fix_gaps, fetch_electricitymaps
│   ├── forecasting/   features, baselines, evaluate_forecast, lstm_model
│   ├── scheduling/    job_generator, baseline_scheduler, rule_based_scheduler, rl_env, rl_scheduler
│   ├── simulation/    simulator, forecast_scheduling
│   ├── analysis/      make_figures
│   └── utils/         config, metrics
├── main.py            entry point (python main.py clean | simulate)
└── requirements.txt
```

## Outputs

| File | Content |
|---|---|
| `results/grid_variability.csv` | Mean, standard deviation, CV and daily range of each grid |
| `results/simulation_results.csv` | Savings, delay and deadline violations for every region, scenario and policy |
| `results/forecast_metrics.csv` | MAE, RMSE and MAPE of each forecaster and horizon |
| `results/forecast_scheduling_results.csv` | Scheduling results with the ML forecast |
| `results/figures/`, `results/tables/` | Figures and tables for the report |

## Limitations

- The job trace is synthetic, not taken from a real data centre.
- India carbon intensity is estimated from the generation mix with fixed emission factors.
- Missing hours were filled by interpolation, and readings close to 0 gCO2/kWh were treated as bad values.
- Forecasts use only past carbon intensity; no weather or wind forecast is used.
- A few jobs may start after their deadline when the cluster is full. These are reported as `violations`.
- Only two grids are compared, so conclusions about variability are descriptive rather than statistical.

## License

To be decided.