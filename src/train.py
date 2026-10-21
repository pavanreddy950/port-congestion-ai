"""Train and evaluate congestion models with a time-based split.

Train: arrivals before TRAIN_END (Jan-Feb 2023). Test: arrivals after (March).
Compares the ensemble against three baselines and writes metrics, plots and
the fitted model.

Usage:
    python -m src.train
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.build_dataset import TRAIN_END

VAL_START = pd.Timestamp("2023-02-15")
from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.model import CATEGORICAL_FEATURES, CONFORMAL_ALPHA, FEATURES, NUMERIC_FEATURES, TARGET, CongestionModel


def metrics(y, p, threshold=0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def best_threshold(y, p) -> float:
    """Probability cut-off maximising Youden's J (recall + specificity - 1) on
    a validation set: it weighs missed delays and false alarms equally, where
    maximising F1 on a small window tended to flag nearly every ship."""
    grid = np.unique(np.round(p, 3))
    if len(grid) <= 2:   # binary rule baseline
        return 0.5
    pos, neg = y == 1, y == 0
    return float(max(grid, key=lambda t: (p[pos] >= t).mean() + (p[neg] < t).mean() - 1))


def port_rate_baseline(train, test):
    rate = train.groupby("port")[TARGET].mean()
    return test["port"].map(rate).fillna(train[TARGET].mean()).values


def logistic_baseline(train, test):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    clf = make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(train[FEATURES], train[TARGET])
    return clf.predict_proba(test[FEATURES])[:, 1]


def rule_baseline(train, test):
    """Hand rule: congested when same-type queue per berth exceeds the cut-off
    that gives the best F1 on the training data."""
    q = train["queue_per_berth_same_type"]
    cut = max(np.linspace(q.min(), q.quantile(0.99), 50),
              key=lambda c: f1_score(train[TARGET], (q >= c).astype(int), zero_division=0))
    return (test["queue_per_berth_same_type"] >= cut).astype(float).values


BASELINES = {
    "Baseline: port historical rate": port_rate_baseline,
    "Baseline: rule on same-type queue per berth": rule_baseline,
    "Logistic regression": logistic_baseline,
}
OURS = "LightGBM ensemble (ours)"


def expected_calibration_error(y, p, bins=10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def interaction_grid(model, test, n_rows=300) -> pd.DataFrame:
    """Two-way partial dependence: average predicted risk when the same-type
    queue per berth and the strongest gust in the next 24 h are set to grid
    values for a sample of real March arrivals."""
    sample = test.sample(min(n_rows, len(test)), random_state=0)
    queues = np.round(np.linspace(0, test["queue_per_berth_same_type"].quantile(0.95), 6), 2)
    gusts = np.array([20, 35, 50, 65, 80])
    rows = []
    for q in queues:
        for g in gusts:
            probe = sample.assign(queue_per_berth_same_type=q, max_gust_next_24h=g)
            rows.append({"queue_per_berth_same_type": q, "max_gust_next_24h": g,
                         "risk": model.predict(probe)["probability"].mean()})
    return pd.DataFrame(rows)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "port_calls.csv", parse_dates=["arrival_time", "berth_time"])
    train, test = df[df["arrival_time"] < TRAIN_END], df[df["arrival_time"] >= TRAIN_END]
    print(f"train: {len(train)} calls ({train[TARGET].mean():.1%} congested) | "
          f"test: {len(test)} calls ({test[TARGET].mean():.1%} congested)")

    # Each model's decision threshold is tuned on the last two weeks of the
    # training period (fit on data before VAL_START), never on the test set.
    fit_part, val_part = train[train["arrival_time"] < VAL_START], train[train["arrival_time"] >= VAL_START]
    y = test[TARGET].values
    results, test_probs = {}, {}
    for name, fn in BASELINES.items():
        threshold = best_threshold(val_part[TARGET].values, fn(fit_part, val_part))
        test_probs[f"score: {name}"] = fn(train, test)
        results[name] = {**metrics(y, test_probs[f"score: {name}"], threshold), "threshold": threshold}

    # Our model: a copy fitted before VAL_START predicts the validation weeks;
    # those held-out predictions fit the Platt calibrator and the conformal
    # cut-off, and tune the alert threshold. The final model uses all of Jan-Feb.
    y_val = val_part[TARGET].values
    val_raw = CongestionModel().fit(fit_part).member_probs(val_part).mean(axis=1)
    model = CongestionModel().fit(train).calibrate(val_raw, y_val)
    model.threshold = best_threshold(y_val, model.calibrate_probs(val_raw))
    pred = model.predict(test)
    test_probs[f"score: {OURS}"] = pred["probability"].values
    results[OURS] = {**metrics(y, pred["probability"].values, model.threshold), "threshold": model.threshold}

    table = pd.DataFrame(results).T.round(3)
    print("\n", table.to_string())

    calibration = {
        "brier_raw": brier_score_loss(y, pred["probability_raw"]), "brier_calibrated": brier_score_loss(y, pred["probability"]),
        "ece_raw": expected_calibration_error(y, pred["probability_raw"].values),
        "ece_calibrated": expected_calibration_error(y, pred["probability"].values),
    }
    sets = pred["prediction_set"]
    covered = np.where(sets == "either", True, np.where(sets == "delay", y == 1, y == 0))
    single = (sets != "either").values
    conformal = {
        "alpha": CONFORMAL_ALPHA, "q": model.conformal_q, "coverage": covered.mean(),
        "share_single": single.mean(), "single_accuracy": covered[single].mean() if single.any() else np.nan,
        "set_counts": sets.value_counts().to_dict(),
    }
    print("\ncalibration:", {k: round(v, 4) for k, v in calibration.items()})
    print("conformal:", conformal)

    # Does the confidence score mean something? Accuracy by confidence band.
    test = test.assign(**pred, **test_probs)
    test["correct"] = ((test["probability"] >= model.threshold).astype(int) == test[TARGET]).astype(int)
    by_conf = test.groupby("confidence_band").agg(n=("correct", "size"), accuracy=("correct", "mean")).round(3)
    print("\naccuracy by confidence band:\n", by_conf.to_string())

    by_port = test.groupby("port").apply(
        lambda g: pd.Series({"n": len(g), "congested_rate": g[TARGET].mean(),
                             "roc_auc": roc_auc_score(g[TARGET], g["probability"]) if g[TARGET].nunique() > 1 else np.nan}),
        include_groups=False).round(3)
    print("\nper-port performance:\n", by_port.to_string())

    # Global feature importance from SHAP.
    sv = model.shap_values(test)
    importance = sv.abs().mean().sort_values(ascending=False)
    print("\nmean |SHAP| (top 10):\n", importance.head(10).round(3).to_string())

    interaction = interaction_grid(model, test)
    interaction.to_csv(REPORTS_DIR / "interaction_queue_weather.csv", index=False)
    print("\nqueue x gust interaction (avg risk):\n",
          interaction.pivot(index="queue_per_berth_same_type", columns="max_gust_next_24h", values="risk").round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    importance.head(12)[::-1].plot.barh(ax=axes[0], color="#3b6ea5")
    axes[0].set_title("What drives congestion predictions (mean |SHAP|)")
    frac_pos, mean_pred = calibration_curve(y, test["probability"], n_bins=8, strategy="quantile")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    raw_pos, raw_pred = calibration_curve(y, test["probability_raw"], n_bins=8, strategy="quantile")
    axes[1].plot(raw_pred, raw_pos, "o--", color="#c0504d", alpha=0.5, label="before recalibration")
    axes[1].plot(mean_pred, frac_pos, "o-", color="#3b6ea5", label="after Platt recalibration")
    axes[1].set_xlabel("predicted probability"); axes[1].set_ylabel("observed congested share")
    axes[1].set_title("Calibration on March test set"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "importance_calibration.png", dpi=150)

    joblib.dump(model, MODELS_DIR / "congestion_model.joblib")
    test.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump({"models": table.to_dict(orient="index"),
                   "confidence_bands": by_conf.to_dict(orient="index"),
                   "per_port": by_port.to_dict(orient="index"),
                   "shap_importance": importance.round(4).to_dict(),
                   "calibration": calibration, "conformal": conformal,
                   "n_train": len(train), "n_test": len(test)}, f, indent=2, default=float)
    print(f"\nsaved model to {MODELS_DIR}, reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
