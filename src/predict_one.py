"""Predict one real March 2023 arrival from the command line.

Run:  python -m src.predict_one            (a random test arrival)
      python -m src.predict_one 5389       (a specific visit_id)
"""
import sys

import joblib
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.model import explain_row, recommend, summarise

model = joblib.load(MODELS_DIR / "congestion_model.joblib")
test = pd.read_csv(REPORTS_DIR / "test_predictions.csv", parse_dates=["arrival_time"])

if len(sys.argv) > 1:
    row = test[test["visit_id"] == int(sys.argv[1])]
    if row.empty:
        sys.exit(f"visit_id {sys.argv[1]} is not in the March test set")
else:
    row = test.sample(1)

r = row.iloc[0]
pred = model.predict(row).iloc[0]
drivers = explain_row(r, model.shap_values(row).iloc[0])

print(f"\nShip MMSI {r.MMSI} ({r.vessel_group}) arriving {r.port} on {r.arrival_time:%d %b %Y %H:%M}  [visit_id {r.visit_id}]")
print(f"  Predicted risk of 24 h+ wait : {pred.probability:.0%}  "
      f"({'FLAGGED' if pred.probability >= model.threshold else 'not flagged'}, threshold {model.threshold:.0%})")
print(f"  Confidence                   : {int(pred.confidence * 100)}% ({pred.confidence_band})")
print(f"  90% prediction set           : {pred.prediction_set}")
print(f"  Summary: {summarise(pred.probability, drivers, model.threshold)}")
print("  Why:")
for d in drivers:
    print(f"    {'+' if d['direction'] == 'raises' else '-'} {d['text']}  (impact {d['impact']:+.2f})")
print(f"  Suggested action: {recommend(pred.probability, pred.confidence_band, drivers, model.threshold)}")
print(f"  What actually happened: waited {r.anchor_wait_h:.1f} h -> "
      f"{'congested' if r.congested else 'not congested'}\n")
