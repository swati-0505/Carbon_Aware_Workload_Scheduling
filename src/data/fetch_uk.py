import argparse
import os
import time
from datetime import date, datetime, timedelta
import pandas as pd
import requests

BASE = "https://api.carbonintensity.org.uk"
HEADERS = {"Accept": "application/json"}
CHUNK_DAYS = 13
OUT_DIR = os.path.join("data", "uk_raw")
OUT_FILE = os.path.join(OUT_DIR, "uk_carbon_intensity.csv")

def get_json(url, tries=5):
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} for {url} (try {attempt}/{tries})")
        except requests.RequestException as e:
            print(f"  network error: {e} (try {attempt}/{tries})")
        time.sleep(3 * attempt)
    raise RuntimeError(f"Failed after {tries} tries: {url}")

def fmt(d):
    return d.strftime("%Y-%m-%dT%H:%MZ")

def fetch_chunk(start_dt, end_dt):
    s, e = fmt(start_dt), fmt(end_dt)
    inten = get_json(f"{BASE}/intensity/{s}/{e}")["data"]
    gen = get_json(f"{BASE}/generation/{s}/{e}")["data"]

    rows = {}
    for item in inten:
        rows[item["from"]] = {
            "datetime_utc": item["from"],
            "forecast": item["intensity"].get("forecast"),
            "actual": item["intensity"].get("actual"),
            "index": item["intensity"].get("index"),
        }
    for item in gen:
        row = rows.setdefault(item["from"], {"datetime_utc": item["from"]})
        for m in item["generationmix"]:
            row[f"{m['fuel']}_perc"] = m["perc"]
    return list(rows.values())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), end)
        print(f"Fetching {cur.date()} -> {nxt.date()}")
        all_rows.extend(fetch_chunk(cur, nxt))
        cur = nxt
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = (df.drop_duplicates("datetime_utc")
            .sort_values("datetime_utc")
            .reset_index(drop=True))
    df.to_csv(OUT_FILE, index=False)

    print("\n=== UK SUMMARY ===")
    print("rows:", len(df))
    print("from:", df["datetime_utc"].min(), "to:", df["datetime_utc"].max())
    print("missing 'actual':", int(df["actual"].isna().sum()))
    print("saved to:", OUT_FILE)

if __name__ == "__main__":
    main()