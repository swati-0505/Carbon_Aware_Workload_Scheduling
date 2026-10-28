from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
TABLES = RESULTS / "tables"

REGIONS = ["India", "UK"]

REGION_KEYWORDS = {
    "India": ["india", "ember", "carbontracker"],
    "UK": ["uk", "gb", "britain"],
}

EMISSION_FACTORS = {
    "coal": 820,
    "lignite": 820,
    "thermal": 820,
    "gas": 490,
    "oil": 650,
    "diesel": 650,
    "nuclear": 12,
    "hydro": 24,
    "wind": 11,
    "solar": 45,
    "biomass": 230,
    "geothermal": 38,
    "renewable": 30,
}

RANDOM_SEED = 42
N_JOBS = 200
JOB_POWER_KW = (0.2, 2.0)          
JOB_DURATION_H = (1, 6)            
DEADLINE_SLACK_H = (6, 24)         

FORECAST_HORIZON_H = 24
TRAIN_FRACTION = 0.8


def ensure_dirs():
    for p in [DATA_RAW, DATA_PROCESSED, FIGURES, MODELS, TABLES]:
        p.mkdir(parents=True, exist_ok=True)