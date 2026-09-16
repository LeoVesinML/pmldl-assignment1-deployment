# `models/`

This folder holds the artifacts produced by **stage 2 (model engineering)**:

| File | Description |
| --- | --- |
| `model.joblib` | Packaged end-to-end scikit-learn pipeline (feature engineering → preprocessing → classifier). Baked into the API image during stage 3. |
| `metrics.json` | Test-set metrics of the champion model plus the cross-validation leaderboard of every candidate. |
| `model_metadata.json` | Model name/version, training timestamp, git revision, feature schema and training-set feature ranges (consumed by the API and the web app). |

The artifacts are **not committed** (see `.gitignore`) because they are
regenerated on every pipeline run. To create them:

```bash
make up          # Airflow runs the full pipeline every 5 minutes
# or, without Airflow:
make data && make train
```
