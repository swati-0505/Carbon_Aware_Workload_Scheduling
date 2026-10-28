import argparse
import os
from datetime import datetime, timedelta
import pandas as pd
import requests

BASE = "https://api.electricitymaps.com"
OUT_DIR = os.path.join("data", "electricitymaps")
ZONES = {"UK": ["GB"], "India": []}  


def _get(path, params=None):
    token = os.environ.get("ELECTRICITYMAPS_TOKEN")
    headers = {"auth-token": token} if token else {}
    r = requests.get(BASE + path, headers=headers, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {path}: {r.text[:200]}")
    return r.json()


def list_zones():
    zones = _get("/v3/zones")
    for key in sorted(zones):
        name = zones[key].get("zoneName", "") if isinstance(zones[key], dict) else ""
        print(f"{key:12s} {name}")
    return zones


def _records(payload):
    for key in ("history", "data"):
        if key in payload:
            return payload[key]
    raise RuntimeError(f"unexpected response, keys: {list(payload)}")


def _to_frame(records):
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return (df[["timestamp", "carbonIntensity"]]
            .rename(columns={"carbonIntensity": "carbon_intensity"})
            .drop_duplicates("timestamp").sort_values("timestamp"))


def fetch_history(zone):
    return _to_frame(_records(_get("/v3/carbon-intensity/history", {"zone": zone})))


def fetch_range(zone, start, end):
    frames, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(days=10), end)
        payload = _get("/v3/carbon-intensity/past-range", {
            "zone": zone, "start": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": nxt.strftime("%Y-%m-%dT%H:%M:%SZ")})
        frames.append(_to_frame(_records(payload)))
        cur = nxt
    return pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")


def run(regions=None):
    """Pipeline entry point: fetches the last 24 h for each known zone.
    Does nothing (and does not fail) when no token is set."""
    if not os.environ.get("ELECTRICITYMAPS_TOKEN"):
        print("  [electricitymaps] ELECTRICITYMAPS_TOKEN not set; skipping (optional step)")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    names = regions or list(ZONES)
    for r in names:
        for zone in ZONES.get(r, []):
            df = fetch_history(zone)
            path = os.path.join(OUT_DIR, f"em_{zone}_history.csv")
            df.to_csv(path, index=False)
            print(f"  [electricitymaps] {zone}: {len(df)} rows -> {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-zones", action="store_true")
    ap.add_argument("--zone")
    ap.add_argument("--mode", choices=["history", "range"], default="history")
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    if a.list_zones:
        list_zones()
    elif a.zone:
        os.makedirs(OUT_DIR, exist_ok=True)
        if a.mode == "history":
            df = fetch_history(a.zone)
        else:
            df = fetch_range(a.zone, datetime.strptime(a.start, "%Y-%m-%d"),
                             datetime.strptime(a.end, "%Y-%m-%d"))
        path = os.path.join(OUT_DIR, f"em_{a.zone}_{a.mode}.csv")
        df.to_csv(path, index=False)
        print(f"{len(df)} rows -> {path}")
    else:
        ap.print_help()