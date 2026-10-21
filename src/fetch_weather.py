"""Fetch hourly historical weather (wind, gusts, rain) and wave height for
each port from the free Open-Meteo archive APIs (no API key needed).

Usage:
    python -m src.fetch_weather --start 2023-01-01 --end 2023-03-31
"""
import argparse

import pandas as pd
import requests

from src.config import PORTS, PROCESSED_DIR

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"


def _hourly(url: str, lat: float, lon: float, start: str, end: str, variables: list[str]) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": ",".join(variables), "timezone": "UTC"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch(start: str, end: str) -> pd.DataFrame:
    frames = []
    for port, cfg in PORTS.items():
        wx = _hourly(WEATHER_URL, cfg["lat"], cfg["lon"], start, end,
                     ["wind_speed_10m", "wind_gusts_10m", "precipitation"])
        waves = _hourly(MARINE_URL, cfg["wave_lat"], cfg["wave_lon"], start, end, ["wave_height"])
        frames.append(wx.merge(waves, on="time", how="left").assign(port=port))
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2023-03-31")
    args = p.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch(args.start, args.end)
    out = PROCESSED_DIR / "weather_hourly.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {len(df):,} rows to {out}")
    print(df.groupby("port")[["wind_speed_10m", "wind_gusts_10m", "wave_height"]].mean().round(1))


if __name__ == "__main__":
    main()
