import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DEFAULT_REGIONS = ["India", "UK"]

FOLDERS = [
    "data/raw",
    "data/processed",
    "results/figures",
    "results/models",
    "results/tables",
]

STEPS = {
    "clean": [
        "src.data.clean_data",
        "src.data.derive_carbon_intensity",
    ],
    "forecast": [
        "src.forecasting.evaluate_forecast",
    ],
    "simulate": [
        "src.simulation.simulator",
    ],
}


def ensure_folders():
    for folder in FOLDERS:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)


def run_module(module_name, regions):
    print(f"\n>>> Running {module_name}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        print(f"    [skipped] cannot import {module_name}: {e}")
        return False
    if not hasattr(module, "run"):
        print(f"    [skipped] {module_name} has no run(regions) function yet")
        return False
    module.run(regions)
    print(f"    [done] {module_name}")
    return True


def run_step(step, regions):
    print(f"\n=== Step: {step} | regions: {', '.join(regions)} ===")
    ok = True
    for module_name in STEPS[step]:
        ok = run_module(module_name, regions) and ok
    return ok


def check():
    ensure_folders()
    print("Folders OK.\n")
    print("Module status:")
    for step, modules in STEPS.items():
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                status = "ready" if hasattr(module, "run") else "missing run()"
            except ModuleNotFoundError:
                status = "not found"
            except Exception as e:  # syntax errors etc.
                status = f"error: {e}"
            print(f"  [{step}] {module_name}: {status}")

    raw = list((ROOT / "data" / "raw").glob("*"))
    print(f"\nFiles in data/raw: {len(raw)}")
    for f in raw[:10]:
        print(f"  - {f.name}")
    if not raw:
        print("  (empty - download the India and UK datasets here first)")


def launch_app():
    app_path = ROOT / "app.py"
    if not app_path.exists():
        print("app.py not found yet. Create it first (Streamlit UI for the agent).")
        return
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def main():
    parser = argparse.ArgumentParser(description="Carbon-Aware Workload Scheduling")
    parser.add_argument(
        "command",
        choices=["check", "clean", "forecast", "simulate", "all", "app"],
        help="what to run",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=DEFAULT_REGIONS,
        help="regions to process, e.g. --regions India UK",
    )
    args = parser.parse_args()

    ensure_folders()

    if args.command == "check":
        check()
    elif args.command == "app":
        launch_app()
    elif args.command == "all":
        for step in ["clean", "forecast", "simulate"]:
            run_step(step, args.regions)
    else:
        run_step(args.command, args.regions)


if __name__ == "__main__":
    main()