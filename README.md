# Flow watch: network intrusion detection on NSL-KDD

Classifies network connections as **normal, DoS, probe, R2L or U2R**, raises alerts
through a FastAPI service, and explains every alert with SHAP in an analyst console.

## Run it

```bash
pip install -r requirements.txt
./scripts/download_data.sh          # NSL-KDD train/test into ./data
python src/train.py                 # compares 6 model variants, saves artifacts/
python src/explain.py               # global SHAP chart
uvicorn app.main:app                # console at http://127.0.0.1:8000, API docs at /docs
pytest -q tests
```

## What is in here

| Path | Purpose |
|---|---|
| `src/data.py` | Loading, attack-to-family mapping, feature engineering, frozen one-hot schema |
| `src/train.py` | RandomForest / XGBoost / LightGBM, with and without imbalance weighting |
| `src/explain.py` | SHAP per-alert reasons and global importance |
| `app/main.py` | `POST /api/score`, `GET /api/replay`, `GET /api/results`, `GET /api/health` |
| `app/static/index.html` | Analyst console: traffic ribbon, alert queue, flow inspector |

## Evaluation protocol

1. KDDTrain+ is deduplicated and split 80/20 (stratified) into train/validation.
2. All six variants are compared on **validation macro-F1**; the test set plays no part in selection.
3. The winner is refit on all training data and evaluated once on **KDDTest+**,
   which contains attack types that never appear in training.

Imbalance handling: plain training vs. square-root-balanced sample weights. Full
"balanced" weights would give the 52 U2R rows a weight in the hundreds.

## Results

Selected model: **XGBoost with sqrt-balanced weights**.

| | Validation (train distribution) | Test (KDDTest+) |
|---|---|---|
| Macro-F1, 5 classes | 0.977 | 0.576 |
| Attack-vs-normal precision | | 0.968 |
| Attack-vs-normal recall | | 0.662 |
| False-alarm rate | | 2.9% |
| Detection of attack types seen in training | | 77.3% |
| Detection of attack types **not** seen in training | | 39.4% |

Per class on the test set: DoS F1 0.90, probe 0.75, U2R 0.32, **R2L recall 0.06**.

Weighting helped consistently (XGBoost 0.563 to 0.576, LightGBM 0.459 to 0.573 macro-F1 on test).

### Read this before trusting the numbers

- The validation-to-test drop (0.98 to 0.58) is the main finding, not a bug. Validation
  comes from the same distribution as training, so it is optimistic. KDDTest+ shifts
  the attack mix and adds new attacks, which is closer to what a deployed detector faces.
- Many R2L attacks in the test set (snmpguess, snmpgetattack, httptunnel, sendmail)
  are types the model never saw. It scores them as normal with high confidence, so
  lowering the alert threshold barely helps (recall 0.66 at 0.5, 0.74 at 0.05).
  Supervised models learn known attacks. Catching novel ones needs anomaly detection
  or different features, which is the natural next step.
- NSL-KDD is a cleaned 1999 dataset. Its connection-level features are not what a
  modern sensor emits, so treat this as a method demonstration, not a deployable IDS.
- The console replays a sample enriched with attacks (about 1 in 5), so its
  alert counts are not a real-world base rate.

## Next steps

- Re-run the same pipeline on UNSW-NB15 or CICIDS2017 (deduplicate first, and report
  train-on-one, test-on-other results).
- Add an anomaly-detection stage (isolation forest or autoencoder on normal traffic) for novel attacks.
- Calibrate probabilities and tune the alert threshold against an alert-volume budget.
