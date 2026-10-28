import sys
from pathlib import Path
import pandas as pd

FOLDER = Path(__file__).resolve().parent / "data" / "india_monthly"

def find_time_col(df):
    named = [c for c in df.columns if any(k in str(c).lower() for k in ("date", "time", "stamp"))]
    for c in named + [c for c in df.columns if c not in named]:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        best, ratio = None, 0.0
        for dayfirst in (True, False):
            t = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst)
            r = t.notna().mean()
            if r > ratio:
                best, ratio = t, r
        if ratio > 0.8:
            return c, best
    return None, None

def main():
    args = sys.argv[1:]
    summary_only = "--summary" in args
    filters = [a.lower() for a in args if not a.startswith("--")]

    files = sorted(list(FOLDER.glob("*.xlsx")) + list(FOLDER.glob("*.xls")) + list(FOLDER.glob("*.csv")))
    if filters:
        files = [f for f in files if any(k in f.name.lower() for k in filters)]
    if not files:
        print(f"No matching files in {FOLDER}")
        return

    rows = []
    for f in files:
        try:
            xl = None if f.suffix.lower() == ".csv" else pd.ExcelFile(f)
            sheets = ["(csv)"] if xl is None else [s for s in xl.sheet_names if not s.startswith("_")]
        except Exception as e:
            print(f"{f.name}: ERROR opening file: {e}")
            continue

        for sheet in sheets:
            try:
                df = pd.read_csv(f) if xl is None else xl.parse(sheet)
            except Exception as e:
                print(f"{f.name} [{sheet}]: ERROR reading sheet: {e}")
                continue

            col, t = find_time_col(df)
            first = t.min() if col is not None else pd.NaT
            last = t.max() if col is not None else pd.NaT
            dups = missing = None
            if col is not None:
                t = t.dropna()
                dups = int(t.duplicated().sum())
                expected = int((t.max() - t.min()) / pd.Timedelta(hours=1)) + 1
                missing = expected - t.nunique()
            rows.append({"file": f.name, "sheet": sheet, "rows": len(df), "cols": df.shape[1],
                         "first": first, "last": last, "dup_hours": dups, "missing_hours": missing})

            if not summary_only:
                print("=" * 70)
                print(f"{f.name}  |  sheet: {sheet}  |  {df.shape[0]} rows x {df.shape[1]} cols")
                print("columns:", [str(c) for c in df.columns][:15])
                print(df.head(3).to_string(max_colwidth=22))
                print("time column:", col, "| first:", first, "| last:", last,
                      "| duplicate hours:", dups, "| missing hours:", missing)

    print("\n" + "=" * 70 + "\nSUMMARY")
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if len(table) and table["first"].notna().any():
        print(f"\nOverall range: {table['first'].min()}  ->  {table['last'].max()}")


if __name__ == "__main__":
    main()