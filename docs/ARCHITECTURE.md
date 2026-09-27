# Architecture & Code Organization

## Repository Structure

```
card-risk/
├── app.py                       # Streamlit web interface (UI in English)
├── train.py                     # Initial training script (manual / build-time)
├── retrain_pipeline.py          # Automated maintenance pipeline (retrain + promote)
├── test_app.py                  # App & model functional tests (11 cases)
├── test_maintenance_ci.py       # Maintenance & CI tests (3 cases / 10 tests)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # CI: test + docker-build + deploy (on push/PR to main)
│       └── maintenance.yml      # Maintenance: weekly retraining (cron Sundays 03:00 UTC)
│
├── data/
│   └── heart.csv                # UCI Heart Disease dataset (auto-downloaded if missing)
│
├── model/
│   ├── cardio_risk_model.joblib # PRODUCTION alias — the file app.py always reads
│   ├── model_metadata.json      # Metrics & hyperparameters of the current production model
│   ├── registry.json            # Model registry: current champion + full version history
│   └── registry/                # Versioned model artifacts (model_v1.joblib, v2, ...)
│
└── docs/
    ├── DEPLOYMENT.md            # Deployment guide & secrets setup
    ├── ARCHITECTURE.md          # This file
    ├── CI_CD.md                 # CI/CD pipeline details
    └── MAINTENANCE.md           # Maintenance pipeline & model registry
```

---

## Design Principles

### Single production alias
`app.py` always loads one fixed path: `model/cardio_risk_model.joblib`.  
The app and the `Dockerfile` never need to know *which* version is champion — that's managed exclusively by `retrain_pipeline.py` through the registry. This decouples the model lifecycle from the code lifecycle.

### train.py is the shared foundation
Both the initial training (`train.py __main__`) and the maintenance pipeline (`retrain_pipeline.py`) reuse the same functions:

```
train.py
  ├── download_dataset()
  ├── load_and_engineer_features()
  ├── build_preprocessor()
  ├── train_and_select_best_model()   ← GridSearchCV over 3 candidates
  └── evaluate_model()
         ↑
retrain_pipeline.py imports these directly
```

This ensures that both training runs (initial and periodic) use identical feature engineering and preprocessing — preventing train/serve skew.

### Pure promotion logic
`decide_promotion()` in `retrain_pipeline.py` is a **pure function** with no side effects. It can be tested in isolation without training a model, making the test suite fast and deterministic.

---

## Key Files

| File | Responsibility |
|---|---|
| [`app.py`](../app.py) | Streamlit UI + `build_feature_row()` for inference-time feature engineering |
| [`train.py`](../train.py) | Dataset download, feature engineering, GridSearchCV, evaluation, serialization |
| [`retrain_pipeline.py`](../retrain_pipeline.py) | Registry management, promotion logic, versioned model artifacts |
| [`test_app.py`](../test_app.py) | Input validation, model endpoint, Streamlit AppTest smoke test |
| [`test_maintenance_ci.py`](../test_maintenance_ci.py) | Promotion logic (cases 1 & 2), workflow YAML validation (case 3) |

---

## Feature Engineering

13 original clinical variables + 4 derived features = 17 model inputs:

| Feature | Formula | Type |
|---|---|---|
| `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | — | Numeric (original) |
| `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal` | — | Categorical (original) |
| `bp_chol_ratio` | `trestbps / chol` | Numeric (derived) |
| `max_hr_expected` | `220 - age` | Numeric (derived) |
| `hr_reserve_deficit` | `max_hr_expected - thalach` | Numeric (derived) |
| `age_group` | bins: `<40`, `40-49`, `50-59`, `60+` | Categorical (derived) |

Preprocessing pipeline: `StandardScaler` (numeric) + `OneHotEncoder(handle_unknown="ignore")` (categorical), wrapped in a `ColumnTransformer` inside a scikit-learn `Pipeline`. The serialized `.joblib` includes the full preprocessor — no manual preprocessing at inference time.

---

## Model Selection

Three candidates evaluated via `GridSearchCV` with 5-fold stratified CV, optimizing ROC-AUC:

| Model | Best CV ROC-AUC |
|---|---|
| Logistic Regression | **0.9077** ✅ champion |
| Random Forest | 0.9062 |
| Gradient Boosting | 0.8905 |

**Selection criterion:** best CV ROC-AUC + parsimony (simplest model among equivalent performers).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web UI | Streamlit |
| ML | scikit-learn, joblib |
| Container | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest, `streamlit.testing.v1.AppTest`, PyYAML |
