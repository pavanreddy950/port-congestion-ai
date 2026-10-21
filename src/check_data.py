"""Quick look at the final dataset.

Run:  python -m src.check_data
"""
import pandas as pd

from src.config import PROCESSED_DIR

calls = pd.read_csv(PROCESSED_DIR / "port_calls.csv", parse_dates=["arrival_time"])

print(f"Port calls: {len(calls):,}   Vessels: {calls['MMSI'].nunique():,}   Columns: {calls.shape[1]}")
print(f"Period: {calls['arrival_time'].min():%d %b %Y} to {calls['arrival_time'].max():%d %b %Y}")
print(f"Congested (24 h+ at anchor): {calls['congested'].mean():.0%}\n")

summary = calls.groupby("port").agg(
    calls=("congested", "size"),
    congested_share=("congested", "mean"),
    median_wait_h=("anchor_wait_h", "median"),
)
print(summary.round(2).to_string(), "\n")

print("First 5 rows (key columns):")
print(calls[["MMSI", "port", "vessel_group", "arrival_time", "anchor_wait_h", "congested",
             "queue_same_type", "draft", "wind_gusts_10m"]].head()
      .round({"anchor_wait_h": 1, "queue_same_type": 0, "draft": 1, "wind_gusts_10m": 1}).to_string(index=False))
