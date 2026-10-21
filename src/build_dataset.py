"""Turn raw AIS pings into one row per port call, with an anchorage-wait
label and features that are known at the moment the ship arrives.

Definitions
-----------
- Visit: consecutive pings of one vessel inside one port area; a gap of more
  than VISIT_GAP_H hours starts a new visit.
- Anchored ping: nav status 1 ("at anchor") and speed < 1 kn.
- Moored ping:   nav status 5 ("moored")    and speed < 1 kn.
- Arrival time:  first ping of the visit.
- Berth time:    first moored ping of the visit.
- Anchor wait:   berth time - first anchored ping (0 if it never anchored
  before berthing). This is the queueing delay caused by congestion.
- Congested:     anchor wait >= CONGESTION_THRESHOLD_H.

Usage:
    python -m src.build_dataset
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR

VISIT_GAP_H = 12
CONGESTION_THRESHOLD_H = 24
TRAIN_END = pd.Timestamp("2023-03-01")   # port capacity is estimated on data before this
SPEED_STOPPED_KN = 1.0


def load_pings() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("ais_*.parquet"))
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.sort_values(["MMSI", "port", "BaseDateTime"]).reset_index(drop=True)
    stopped = df["SOG"] < SPEED_STOPPED_KN
    df["anchored"] = stopped & (df["Status"] == 1)
    df["moored"] = stopped & (df["Status"] == 5)
    return df


def segment_visits(df: pd.DataFrame) -> pd.DataFrame:
    gap = df.groupby(["MMSI", "port"])["BaseDateTime"].diff()
    new_visit = gap.isna() | (gap > pd.Timedelta(hours=VISIT_GAP_H))
    df["visit_id"] = new_visit.cumsum()
    return df


def summarise_visits(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("visit_id")
    visits = pd.DataFrame({
        "MMSI": g["MMSI"].first(),
        "port": g["port"].first(),
        "vessel_type": g["VesselType"].first(),
        "length": g["Length"].median(),
        "width": g["Width"].median(),
        "draft": g["Draft"].median(),
        "arrival_time": g["BaseDateTime"].first(),
        "last_seen": g["BaseDateTime"].last(),
        "first_ping_moored": g["moored"].first(),
        "berth_time": df[df["moored"]].groupby("visit_id")["BaseDateTime"].first(),
    })
    # first anchored ping that happens before berthing
    anch = df[df["anchored"]].merge(visits[["berth_time"]], left_on="visit_id", right_index=True)
    anch = anch[anch["berth_time"].isna() | (anch["BaseDateTime"] < anch["berth_time"])]
    visits["anchor_time"] = anch.groupby("visit_id")["BaseDateTime"].first()
    return visits


def clean_visits(visits: pd.DataFrame, data_start, data_end) -> tuple[pd.DataFrame, dict]:
    """Drop visits where the true wait cannot be observed; return drop counts."""
    stats = {"all_visits": len(visits)}
    v = visits[~visits["first_ping_moored"]]                       # arrival not seen
    stats["drop_already_moored_at_first_ping"] = len(visits) - len(v)
    n = len(v)
    v = v[v["arrival_time"] >= data_start + pd.Timedelta(hours=VISIT_GAP_H)]  # left-censored
    stats["drop_arrived_before_data_start"] = n - len(v)
    n = len(v)
    v = v[v["arrival_time"] <= data_end - pd.Timedelta(days=5)]   # right-censored
    stats["drop_arrived_in_last_5_days"] = n - len(v)
    n = len(v)
    v = v[v["berth_time"].notna()]   # anchorage-only calls (bunkering, lightering, transit)
    stats["drop_never_berthed"] = n - len(v)
    stats["kept_port_calls"] = len(v)

    v = v.copy()
    wait = (v["berth_time"] - v["anchor_time"]).dt.total_seconds() / 3600
    v["anchor_wait_h"] = wait.fillna(0.0)
    v["time_to_berth_h"] = (v["berth_time"] - v["arrival_time"]).dt.total_seconds() / 3600
    v["vessel_group"] = np.where(v["vessel_type"] >= 80, "tanker", "cargo")
    return v.reset_index(), stats


def hourly_port_state(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct vessels anchored / moored in each port in each hour."""
    df = df.assign(hour=df["BaseDateTime"].dt.floor("h"),
                   group=np.where(df["VesselType"] >= 80, "tanker", "cargo"))
    cols = [
        df[df["anchored"]].groupby(["port", "hour"])["MMSI"].nunique().rename("queue_anchored"),
        df[df["moored"]].groupby(["port", "hour"])["MMSI"].nunique().rename("berths_occupied"),
    ]
    # Tankers and cargo ships use different berths, so count them separately too.
    for g in ("cargo", "tanker"):
        sub = df[df["group"] == g]
        cols.append(sub[sub["anchored"]].groupby(["port", "hour"])["MMSI"].nunique().rename(f"queue_{g}"))
        cols.append(sub[sub["moored"]].groupby(["port", "hour"])["MMSI"].nunique().rename(f"berths_{g}"))
    state = pd.concat(cols, axis=1).fillna(0)

    # Re-index to a full hourly grid so hours with zero ships count as zero.
    full = []
    for port, s in state.groupby(level="port"):
        s = s.droplevel("port")
        grid = pd.date_range(s.index.min(), s.index.max(), freq="h")
        s = s.reindex(grid, fill_value=0)
        s["queue_change_24h"] = s["queue_anchored"] - s["queue_anchored"].shift(24)
        full.append(s.rename_axis("hour").reset_index().assign(port=port))
    return pd.concat(full, ignore_index=True)


def recent_history(arrivals: np.ndarray, done: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean wait and congested share of calls in `done` that berthed in the
    7 days before each arrival time (only information available at arrival)."""
    done = done.sort_values("berth_time")
    bt = done["berth_time"].values
    wait = done["anchor_wait_h"].values
    wait_cum = np.concatenate([[0.0], np.cumsum(wait)])
    cong_cum = np.concatenate([[0], np.cumsum(wait >= CONGESTION_THRESHOLD_H)])
    hi = np.searchsorted(bt, arrivals, side="left")
    lo = np.searchsorted(bt, arrivals - np.timedelta64(7, "D"), side="left")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(n > 0, (wait_cum[hi] - wait_cum[lo]) / n, np.nan),
                np.where(n > 0, (cong_cum[hi] - cong_cum[lo]) / n, np.nan))


def add_features(calls: pd.DataFrame, state: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    calls = calls.copy()
    # State at the last complete hour before arrival (no look-ahead).
    calls["snap_hour"] = calls["arrival_time"].dt.floor("h") - pd.Timedelta(hours=1)
    calls = calls.merge(state, left_on=["port", "snap_hour"], right_on=["port", "hour"], how="left") \
                 .drop(columns="hour")
    count_cols = ["queue_anchored", "berths_occupied", "queue_cargo", "berths_cargo", "queue_tanker", "berths_tanker"]
    calls[count_cols] = calls[count_cols].fillna(0)
    is_tanker = calls["vessel_group"] == "tanker"
    calls["queue_same_type"] = np.where(is_tanker, calls["queue_tanker"], calls["queue_cargo"])
    calls["berths_same_type"] = np.where(is_tanker, calls["berths_tanker"], calls["berths_cargo"])

    # Capacity proxy: 95th percentile of simultaneously moored ships, measured
    # on the training period only (overall and per vessel type).
    train_state = state[state["hour"] < TRAIN_END].groupby("port")
    cap = train_state["berths_occupied"].quantile(0.95).rename("berth_capacity")
    cap_type = train_state[["berths_cargo", "berths_tanker"]].quantile(0.95)
    calls = calls.merge(cap, left_on="port", right_index=True, how="left")
    calls["occupancy_ratio"] = calls["berths_occupied"] / calls["berth_capacity"]
    calls["queue_per_berth"] = calls["queue_anchored"] / calls["berth_capacity"]
    type_cap = np.where(is_tanker, calls["port"].map(cap_type["berths_tanker"]),
                        calls["port"].map(cap_type["berths_cargo"])).clip(min=1)
    calls["occupancy_same_type"] = calls["berths_same_type"] / type_cap
    calls["queue_per_berth_same_type"] = calls["queue_same_type"] / type_cap

    # Traffic and recent-delay history, using only calls that happened earlier.
    feats = []
    for port, grp in calls.groupby("port"):
        grp = grp.sort_values("arrival_time")
        t = grp["arrival_time"].values
        arr_24h = np.searchsorted(t, t) - np.searchsorted(t, t - np.timedelta64(24, "h"))
        arr_7d = np.searchsorted(t, t) - np.searchsorted(t, t - np.timedelta64(7, "D"))
        recent_wait, recent_cong = recent_history(t, grp)

        # same history restricted to ships of the same type
        type_wait, type_cong = np.full(len(grp), np.nan), np.full(len(grp), np.nan)
        for g in ("cargo", "tanker"):
            mask = (grp["vessel_group"] == g).values
            if mask.any():
                type_wait[mask], type_cong[mask] = recent_history(t[mask], grp[mask])

        feats.append(pd.DataFrame({"visit_id": grp["visit_id"].values,
                                   "arrivals_24h": arr_24h, "arrivals_7d": arr_7d,
                                   "recent_avg_wait_7d": recent_wait,
                                   "recent_congested_share_7d": recent_cong,
                                   "recent_avg_wait_7d_same_type": type_wait,
                                   "recent_congested_share_7d_same_type": type_cong}))
    calls = calls.merge(pd.concat(feats), on="visit_id")

    # Calendar features.
    calls["hour_of_day"] = calls["arrival_time"].dt.hour
    calls["day_of_week"] = calls["arrival_time"].dt.dayofweek
    calls["is_weekend"] = (calls["day_of_week"] >= 5).astype(int)

    # Weather at arrival plus the worst gust in the next 24 h. Observed weather
    # stands in for the forecast a planner would have at arrival time.
    weather = weather.sort_values(["port", "time"]).copy()
    weather["max_gust_next_24h"] = weather.groupby("port")["wind_gusts_10m"] \
        .transform(lambda s: s[::-1].rolling(24, min_periods=1).max()[::-1])
    weather["max_wave_next_24h"] = weather.groupby("port")["wave_height"] \
        .transform(lambda s: s[::-1].rolling(24, min_periods=1).max()[::-1])
    calls["wx_hour"] = calls["arrival_time"].dt.floor("h")
    calls = calls.merge(weather.rename(columns={"time": "wx_hour"}), on=["port", "wx_hour"], how="left") \
                 .drop(columns="wx_hour")

    calls["congested"] = (calls["anchor_wait_h"] >= CONGESTION_THRESHOLD_H).astype(int)
    return calls.sort_values("arrival_time").reset_index(drop=True)


def save_map_sample(pings: pd.DataFrame, per_port: int = 8000) -> None:
    """Small sample of anchored/moored positions for the demo map (the raw
    pings are too large to ship). One ping per vessel per state per 6 h."""
    s = pings[pings["anchored"] | pings["moored"]].copy()
    s["state"] = np.where(s["anchored"], "At anchor (waiting)", "At berth (moored)")
    s["slot"] = s["BaseDateTime"].dt.floor("6h")
    s = s.drop_duplicates(["MMSI", "state", "slot"])
    s = s.sample(frac=1, random_state=0).groupby("port").head(per_port)
    s[["port", "state", "LAT", "LON", "MMSI", "VesselName", "BaseDateTime"]] \
        .to_parquet(PROCESSED_DIR / "map_sample.parquet", index=False)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pings = segment_visits(load_pings())
    save_map_sample(pings)
    data_start, data_end = pings["BaseDateTime"].min(), pings["BaseDateTime"].max()
    print(f"pings: {len(pings):,}  vessels: {pings['MMSI'].nunique()}  range: {data_start} -> {data_end}")

    visits = summarise_visits(pings)
    calls, stats = clean_visits(visits, data_start, data_end)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    state = hourly_port_state(pings)
    state.to_parquet(PROCESSED_DIR / "port_hourly_state.parquet", index=False)
    weather = pd.read_parquet(PROCESSED_DIR / "weather_hourly.parquet")
    calls = add_features(calls, state, weather)
    calls.to_csv(PROCESSED_DIR / "port_calls.csv", index=False)

    print(f"\nport calls: {len(calls)}   congested share: {calls['congested'].mean():.1%}")
    print(calls.groupby("port").agg(calls=("visit_id", "size"),
                                    congested=("congested", "mean"),
                                    median_wait_h=("anchor_wait_h", "median"),
                                    p90_wait_h=("anchor_wait_h", lambda s: s.quantile(0.9))).round(2))
    print("\nanchor wait (h) quantiles:", calls["anchor_wait_h"].quantile([.5, .6, .7, .8, .9, .95]).round(1).to_dict())


if __name__ == "__main__":
    main()
