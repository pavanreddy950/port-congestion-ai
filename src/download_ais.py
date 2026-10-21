"""Download daily MarineCadastre AIS files and keep only cargo/tanker
positions inside the configured port areas.

Each raw daily file is ~320 MB zipped (~7M rows for all US waters); the
filtered output is a few MB, so the zip is deleted after processing.

Usage:
    python -m src.download_ais --start 2023-01-01 --end 2023-03-31 --workers 4
"""
import argparse
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from src.config import AIS_COLUMNS, AIS_URL, PORTS, RAW_DIR, VESSEL_TYPE_RANGE


def assign_port(lat: pd.Series, lon: pd.Series) -> pd.Series:
    port = pd.Series(pd.NA, index=lat.index, dtype="object")
    for name, cfg in PORTS.items():
        lat_min, lat_max, lon_min, lon_max = cfg["bbox"]
        inside = lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max)
        port[inside] = name
    return port


def filter_chunk(df: pd.DataFrame) -> pd.DataFrame:
    lo, hi = VESSEL_TYPE_RANGE
    df = df[df["VesselType"].between(lo, hi)]
    df = df.assign(port=assign_port(df["LAT"], df["LON"]))
    return df[df["port"].notna()]


def process_day(day: date) -> str:
    out_path = RAW_DIR / f"ais_{day:%Y_%m_%d}.parquet"
    if out_path.exists():
        return f"{day} skipped (exists)"

    url = AIS_URL.format(year=day.year, month=day.month, day=day.day)
    zip_path = RAW_DIR / f"tmp_{day:%Y_%m_%d}.zip"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)

    parts = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as fh:
            for chunk in pd.read_csv(fh, usecols=AIS_COLUMNS, chunksize=250_000,
                                     dtype={"VesselName": "string", "IMO": "string"}):
                parts.append(filter_chunk(chunk))
    zip_path.unlink()

    df = pd.concat(parts, ignore_index=True)
    # some daily files mix text and numeric MMSI values
    df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce")
    df = df[df["MMSI"].notna()].astype({"MMSI": "int64"})
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df["Status"] = df["Status"].astype("float32")
    df.to_parquet(out_path, index=False)
    return f"{day} done: {len(df):,} rows, {df['MMSI'].nunique()} vessels"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2023-03-31")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    days = [start + timedelta(d) for d in range((end - start).days + 1)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_day, d): d for d in days}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # keep going; failed days can be re-run
                print(f"{futures[fut]} FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
