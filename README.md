# Cardiovascular Risk Predictor

> A machine learning web application that estimates cardiovascular disease risk from clinical data.  
> Built with Python · scikit-learn · Streamlit · Docker · GitHub Actions.

![CI](https://github.com/<owner>/card-risk/actions/workflows/ci.yml/badge.svg)
![Maintenance](https://github.com/<owner>/card-risk/actions/workflows/maintenance.yml/badge.svg)

---

## What it does

Enter a patient's 13 clinical measurements and get an estimated probability of cardiovascular disease, powered by a Logistic Regression pipeline trained on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

**Current model performance (hold-out test set):**

| Metric | Value |
|---|---|
| ROC-AUC | **0.8972** |
| Recall (sensitivity) | **0.9091** |
| Accuracy | 0.8525 |
| F1-score | 0.8696 |

> This tool is for **educational purposes only** and is not a substitute for professional medical advice.

---

## Quickstart

```bash
pip install -r requirements.txt
python train.py                      # train and save the model
streamlit run app.py                 # open http://localhost:8501
```

**Run tests:**
```bash
pytest test_app.py -v                # 11 tests — app & model
pytest test_maintenance_ci.py -v     # 10 tests — maintenance & CI (3 cases)
```

**Run the maintenance pipeline manually:**
```bash
python retrain_pipeline.py --dry-run   # evaluate only, nothing is written
python retrain_pipeline.py             # full run — promotes if model improves
```

---

## Automated Pipelines

### Continuous Integration
Every push and pull request to `main` automatically:
1. Runs the full test suite (21 tests)
2. Validates the Docker image builds cleanly
3. Deploys to the production server via SSH (on push to `main` only)

→ [CI/CD details](docs/CI_CD.md)

### Scheduled Maintenance
Every Sunday at 03:00 UTC, the maintenance pipeline:
1. Downloads the latest dataset
2. Retrains and evaluates 3 model candidates
3. Promotes the new model **only if** it beats the current champion by more than 0.001 ROC-AUC
4. Commits the new model artifact and redeploys — or discards it safely if there's no improvement

→ [Maintenance pipeline details](docs/MAINTENANCE.md)

---

## Tools & Platforms

| Category | Tool |
|---|---|
| Language | Python 3.11 |
| Web UI | Streamlit |
| ML | scikit-learn · joblib |
| Containerization | Docker · Docker Compose |
| CI/CD | GitHub Actions |
| Production server | Ubuntu (SSH deploy via `appleboy/ssh-action`) |
| Testing | pytest · `streamlit.testing.v1.AppTest` · PyYAML |
| Model registry | Custom JSON registry (`model/registry.json`) |

→ [Full stack & architecture details](docs/ARCHITECTURE.md)

---

## Documentation

| Document | Description |
|---|---|
| [Architecture & Code Structure](docs/ARCHITECTURE.md) | Repository layout, design principles, feature engineering |
| [Deployment Guide](docs/DEPLOYMENT.md) | Server setup, GitHub secrets, first deploy, emergency redeploy |
| [CI/CD Pipeline](docs/CI_CD.md) | Workflow diagrams, test suite breakdown, how to add a model |
| [Maintenance Pipeline](docs/MAINTENANCE.md) | Retraining logic, model registry, promotion rules, runbook |
| [Training Report](docs/TRAINING_REPORT.md) | Dataset, hyperparameter tuning, evaluation (Spanish) |
| [Technical Report — Unit II](docs/TECHNICAL_REPORT.md) | Full technical report for the TI team (Spanish) |

---

## Project Structure

```
├── app.py                    # Streamlit web app
├── train.py                  # Initial training script
├── retrain_pipeline.py       # Automated maintenance pipeline
├── test_app.py               # App & model tests (11)
├── test_maintenance_ci.py    # Maintenance & CI tests (10)
├── Dockerfile / docker-compose.yml
├── .github/workflows/
│   ├── ci.yml                # CI: test + build + deploy
│   └── maintenance.yml       # Weekly retraining
├── model/
│   ├── cardio_risk_model.joblib   # Production model alias
│   ├── registry.json              # Model version registry
│   └── registry/                  # Versioned model artifacts
└── docs/                     # Detailed documentation
```

→ [Full architecture details](docs/ARCHITECTURE.md)
