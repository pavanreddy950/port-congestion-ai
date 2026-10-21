"""Model definition, confidence scoring and plain-English explanations.

The predictor is a bagged ensemble of LightGBM classifiers:
- probability  = mean of the ensemble members' predicted probabilities,
                 recalibrated with Platt scaling (a logistic fit on the
                 log-odds) on a held-out validation window
- confidence   = how much the members agree, reduced when the input falls
                 outside the range seen in training
- prediction set = split conformal prediction: the set of outcomes that
                 cannot be ruled out at 90% coverage
- explanation  = SHAP contributions averaged over members, turned into
                 sentences, plus a suggested operational action
"""
import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

CONFORMAL_ALPHA = 0.10   # prediction sets aim to contain the true outcome 90% of the time

NUMERIC_FEATURES = [
    "queue_anchored", "berths_occupied", "occupancy_ratio", "queue_per_berth", "queue_change_24h",
    "queue_same_type", "berths_same_type", "occupancy_same_type", "queue_per_berth_same_type",
    "arrivals_24h", "arrivals_7d", "recent_avg_wait_7d", "recent_congested_share_7d",
    "recent_avg_wait_7d_same_type", "recent_congested_share_7d_same_type",
    "length", "draft",
    "hour_of_day", "day_of_week", "is_weekend",
    "wind_speed_10m", "wind_gusts_10m", "precipitation", "wave_height",
    "max_gust_next_24h", "max_wave_next_24h",
]
CATEGORICAL_FEATURES = ["port", "vessel_group"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "congested"

# Readable names and units for explanations.
FEATURE_TEXT = {
    "queue_anchored": ("ships waiting at anchor", "{:.0f}"),
    "berths_occupied": ("ships already at berth", "{:.0f}"),
    "occupancy_ratio": ("berth occupancy vs capacity", "{:.0%}"),
    "queue_per_berth": ("queue length per berth", "{:.2f}"),
    "queue_change_24h": ("change in anchor queue over last 24 h", "{:+.0f} ships"),
    "queue_same_type": ("same-type ships waiting at anchor", "{:.0f}"),
    "berths_same_type": ("same-type ships already at berth", "{:.0f}"),
    "occupancy_same_type": ("same-type berth occupancy vs capacity", "{:.0%}"),
    "queue_per_berth_same_type": ("same-type queue per berth", "{:.2f}"),
    "recent_avg_wait_7d_same_type": ("avg wait of same-type ships, last 7 days", "{:.1f} h"),
    "recent_congested_share_7d_same_type": ("share of same-type ships delayed 24 h+, last 7 days", "{:.0%}"),
    "arrivals_24h": ("arrivals in the last 24 h", "{:.0f}"),
    "arrivals_7d": ("arrivals in the last 7 days", "{:.0f}"),
    "recent_avg_wait_7d": ("average anchor wait over the last 7 days", "{:.1f} h"),
    "recent_congested_share_7d": ("share of recent ships delayed 24 h+", "{:.0%}"),
    "length": ("vessel length", "{:.0f} m"),
    "draft": ("vessel draft", "{:.1f} m"),
    "hour_of_day": ("arrival hour (UTC)", "{:.0f}:00"),
    "day_of_week": ("arrival day of week", "{:.0f}"),
    "is_weekend": ("weekend arrival", "{:.0f}"),
    "wind_speed_10m": ("wind speed", "{:.0f} km/h"),
    "wind_gusts_10m": ("wind gusts", "{:.0f} km/h"),
    "precipitation": ("rain", "{:.1f} mm/h"),
    "wave_height": ("wave height", "{:.1f} m"),
    "max_gust_next_24h": ("strongest gust in next 24 h", "{:.0f} km/h"),
    "max_wave_next_24h": ("highest waves in next 24 h", "{:.1f} m"),
    "port": ("port", "{}"),
    "vessel_group": ("vessel type", "{}"),
}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def prepare(df: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        cats = categories[col] if categories else sorted(X[col].dropna().unique())
        X[col] = pd.Categorical(X[col], categories=cats)
    return X


@dataclass
class CongestionModel:
    n_members: int = 25
    params: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
    ))
    members: list = field(default_factory=list)
    categories: dict = field(default_factory=dict)
    feature_ranges: dict = field(default_factory=dict)
    threshold: float = 0.5   # set by training: Youden's J cut-off on validation weeks
    calibrator: LogisticRegression | None = None   # set by calibrate()
    conformal_q: float | None = None               # set by calibrate()

    def fit(self, df: pd.DataFrame):
        self.categories = {c: sorted(df[c].dropna().unique()) for c in CATEGORICAL_FEATURES}
        X, y = prepare(df, self.categories), df[TARGET].values
        self.feature_ranges = {c: (np.nanquantile(X[c], 0.01), np.nanquantile(X[c], 0.99))
                               for c in NUMERIC_FEATURES}
        rng = np.random.default_rng(0)
        self.members = []
        for i in range(self.n_members):
            idx = rng.integers(0, len(X), len(X))          # bootstrap sample
            clf = lgb.LGBMClassifier(random_state=i, **self.params)
            clf.fit(X.iloc[idx], y[idx])
            self.members.append(clf)
        return self

    def member_probs(self, df: pd.DataFrame) -> np.ndarray:
        X = prepare(df, self.categories)
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.members])

    def calibrate(self, raw_probs: np.ndarray, y: np.ndarray):
        """Fit Platt recalibration and the split-conformal cut-off on held-out
        data. `raw_probs` must come from a model that did not see `y`. Platt
        scaling is strictly increasing, so ranking and alert decisions are kept."""
        self.calibrator = LogisticRegression(C=1e6).fit(_logit(raw_probs), y)
        p = self.calibrate_probs(raw_probs)
        # nonconformity = 1 - probability given to the true outcome
        scores = np.where(y == 1, 1 - p, p)
        n = len(scores)
        self.conformal_q = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - CONFORMAL_ALPHA)) / n),
                                             method="higher"))
        return self

    def calibrate_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return raw_probs
        return self.calibrator.predict_proba(_logit(raw_probs))[:, 1]

    def prediction_sets(self, p: np.ndarray) -> np.ndarray:
        """Outcomes not ruled out at the conformal level: 'delay', 'no delay' or 'either'."""
        if self.conformal_q is None:
            return np.full(len(p), "n/a")
        delay, no_delay = (1 - p) <= self.conformal_q, p <= self.conformal_q
        return np.select([delay & no_delay, delay, no_delay], ["either", "delay", "no delay"], "either")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.member_probs(df)
        raw, spread = probs.mean(axis=1), probs.std(axis=1)
        p = self.calibrate_probs(raw)
        agreement = np.clip(1 - spread / 0.25, 0, 1)

        X = prepare(df, self.categories)
        out_of_range = np.zeros(len(X))
        for c, (lo, hi) in self.feature_ranges.items():
            out_of_range += ((X[c] < lo) | (X[c] > hi)).fillna(False).values
        support = np.clip(1 - 0.15 * out_of_range, 0.4, 1)

        confidence = agreement * support
        band = np.select([confidence >= 0.75, confidence >= 0.5], ["High", "Medium"], "Low")
        return pd.DataFrame({"probability": p, "probability_raw": raw, "spread": spread,
                             "confidence": confidence, "confidence_band": band,
                             "prediction_set": self.prediction_sets(p),
                             "unusual_inputs": out_of_range.astype(int)},
                            index=df.index)

    def shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average SHAP contributions (log-odds) across ensemble members."""
        X = prepare(df, self.categories)
        total = np.zeros((len(X), len(FEATURES)))
        for m in self.members:
            with warnings.catch_warnings():  # shap notes its LightGBM output format changed
                warnings.simplefilter("ignore", UserWarning)
                sv = shap.TreeExplainer(m).shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
            total += sv
        return pd.DataFrame(total / len(self.members), columns=FEATURES, index=df.index)


def _fmt(feature: str, value) -> str:
    name, fmt = FEATURE_TEXT[feature]
    if pd.isna(value):
        return f"{name}: unknown"
    try:
        return f"{name}: {fmt.format(value)}"
    except (ValueError, TypeError):
        return f"{name}: {value}"


def explain_row(row: pd.Series, shap_row: pd.Series, top_k: int = 4) -> list[dict]:
    """Top drivers of one prediction, as dicts with readable text."""
    order = shap_row.abs().sort_values(ascending=False).index[:top_k]
    return [{"feature": f, "value": row[f], "impact": float(shap_row[f]),
             "direction": "raises" if shap_row[f] > 0 else "lowers",
             "text": f"{_fmt(f, row[f])} -> {'raises' if shap_row[f] > 0 else 'lowers'} delay risk"}
            for f in order]


def summarise(probability: float, drivers: list[dict], threshold: float = 0.5) -> str:
    """One sentence combining the top drivers, e.g. 'Elevated risk ... (62%): driven
    by same-type queue per berth at 1.02 combined with ...'."""
    level = "Elevated" if probability >= threshold else "Some" if probability >= threshold / 2 else "Low"
    up = [_fmt(d["feature"], d["value"]).replace(": ", " at ") for d in drivers if d["direction"] == "raises"]
    down = [_fmt(d["feature"], d["value"]).replace(": ", " at ") for d in drivers if d["direction"] == "lowers"]
    text = f"{level} risk of a 24 h+ anchor wait ({probability:.0%})"
    if up:
        text += ": driven by " + " combined with ".join(up)
    if down:
        text += (", partly offset by " if up else ": kept down by ") + " and ".join(down)
    return text + "."


def recommend(probability: float, confidence_band: str, drivers: list[dict], threshold: float = 0.5) -> str:
    """Translate a prediction into a suggested planning action. `threshold` is
    the model's tuned alert cut-off; half of it marks the 'watch' zone."""
    up = {d["feature"] for d in drivers if d["direction"] == "raises"}
    weather = up & {"wind_gusts_10m", "max_gust_next_24h", "wave_height", "max_wave_next_24h", "wind_speed_10m"}
    queue = up & {f for f in NUMERIC_FEATURES
                  if f.startswith(("queue", "berths", "occupancy", "recent", "arrivals"))}

    if probability >= threshold:
        action = "Flagged: likely to wait 24 h+ at anchor. "
        if queue:
            action += ("The queue/berth situation is the main driver: consider slow-steaming to arrive "
                       "later (saves fuel, same berth time), tell the consignee to expect a delay, and "
                       "push back trucking/rail pickup slots. ")
        if weather:
            action += "Weather is adding risk: expect pilotage or berthing suspensions; add buffer time. "
    elif probability >= threshold / 2:
        action = ("Watch: some risk. Keep the current schedule but monitor the anchorage queue; "
                  "set an alert if the queue grows before arrival. ")
    else:
        action = "Low risk. Plan for normal berthing; no action needed. "

    if confidence_band == "Low":
        action += "Confidence is low (models disagree or conditions are unusual), so treat this as a rough signal."
    return action.strip()
