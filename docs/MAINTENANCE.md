# Maintenance Pipeline & Model Registry

## Overview

The maintenance pipeline (`retrain_pipeline.py` + `maintenance.yml`) keeps the production model up to date without human intervention. It implements a lightweight **model registry** pattern with a built-in safeguard: it will never silently replace a good model with a worse one.

---

## Schedule

```
Every Sunday at 03:00 UTC
cron: "0 3 * * 0"
```

Can also be triggered manually from **GitHub → Actions → Model Maintenance → Run workflow**, with an optional `dry_run` flag that evaluates the candidate without writing anything.

---

## Pipeline Steps

```
cron / workflow_dispatch
        │
        ▼
retrain_pipeline.py [--dry-run]
        │
        ├─ 1. download_dataset()
        │       Downloads heart.csv if a newer version is available
        │
        ├─ 2. load_and_engineer_features()
        │       Same feature engineering as initial training (no skew)
        │
        ├─ 3. train_and_select_best_model()
        │       GridSearchCV over 3 candidates, 5-fold stratified CV
        │       Optimizing: ROC-AUC
        │
        ├─ 4. evaluate_model()
        │       Measures accuracy, precision, recall, F1, ROC-AUC
        │       on a held-out test set
        │
        └─ 5. decide_promotion()
                candidate.roc_auc > champion.roc_auc + 0.001 ?
                        │
                ┌───────┴───────┐
              YES              NO
                │               │
                ▼               ▼
          PROMOTE            DISCARD
          ────────           ───────
          Copy .joblib       Log to history
          to production      Keep current
          Update registry    model in prod
          Update metadata    (safe rollback)
          Run regression
          tests (pytest)
          git commit [skip ci]
          git push
          SSH redeploy
```

---

## Promotion Logic

Implemented in `decide_promotion()` — a **pure function** (no side effects, fully unit-tested):

```python
def decide_promotion(champion, candidate_metrics, metric="roc_auc", tolerance=0.001):
    if champion is None:
        return True   # always promote the first model
    return (candidate_metrics[metric] - champion["metrics"][metric]) > tolerance
```

The `0.001` tolerance prevents promoting a model that improved only due to random variation in the train/test split or the optimizer.

**Exit codes:**
- `0` → model promoted (or dry-run completed without errors)
- `1` → candidate did not beat the champion; production unchanged

---

## Model Registry

All promotion decisions are recorded in `model/registry.json`:

```json
{
  "champion": {
    "version": "v1",
    "model_name": "logistic_regression",
    "metrics": { "roc_auc": 0.8972, "recall": 0.9091, ... },
    "best_params": { "model__C": 0.1, "model__solver": "lbfgs" },
    "trained_at": "2026-09-27T02:50:51Z",
    "promoted": true,
    "artifact_path": "model/registry/model_v1.joblib"
  },
  "history": [ ... ]
}
```

Each run — whether or not it promotes — appends an entry to `history`, providing a full audit trail.

---

## Audit Trail

Every model change is recorded in three places:

| Where | What |
|---|---|
| **Git history** | Commit: `chore(maintenance): retrain and promote new champion model [skip ci]` |
| **`model/registry.json`** | Structured record: version, metrics, params, timestamp, promoted flag |
| **GitHub Actions artifacts** | `registry.json` uploaded as `model-registry-{run_id}` after every run |

---

## Versioned Artifacts

```
model/
├── cardio_risk_model.joblib     ← production alias (always up to date)
├── registry.json                ← registry state
└── registry/
    ├── model_v1.joblib          ← v1 artifact (preserved forever)
    ├── model_v2.joblib          ← v2 artifact
    └── ...
```

`app.py` only reads `cardio_risk_model.joblib`. Rolling back means copying any `model_v{N}.joblib` over it and pushing.

---

## Runbook

### Model is producing wrong predictions

Two levels of rollback are available depending on the severity:

#### Level 1 — Automatic Safe Rollback (no action needed)

If the weekly retraining produces a worse model, `decide_promotion()` returns `False`, the pipeline exits with code `1`, and **production is never touched**. No intervention required.

#### Level 2 — Emergency Rollback via GitHub Actions

Use this when a bad model was already promoted and is actively causing wrong predictions in production:

1. Go to **GitHub → Actions → Emergency Rollback → Run workflow**.
2. Enter the `target_ref` (e.g. `v1.1.0`, `v1.1.1`, or a specific commit SHA). Use `git tag -l` to see available tags.
3. Enter a brief `reason` for the audit log (e.g. `"Corrupt ingestion from sensor batch #42"`).
4. Click **Run workflow** — the server reverts in **< 90 seconds**.

The workflow:
- Runs `git checkout <target_ref>` on the server
- Rebuilds the Docker container from the historical source (including the `.joblib` at that commit)
- Prunes stale images

> **After rollback:** Review `model/registry.json` and commit a corrected state to `main` to re-establish a clean history.

See [CI_CD.md → Emergency Rollback Workflow](./CI_CD.md#emergency-rollback-workflow-rollbackyml) for the full details, secrets, and flow diagram.

#### Level 3 — Manual File Rollback (alternative)

If you only need to swap the model file without reverting code:

```bash
cp model/registry/model_vN.joblib model/cardio_risk_model.joblib
# update "champion" in model/registry.json to point to vN
git add model/ && git commit -m "fix: rollback model to vN" && git push
```

CI picks up the push and redeploys automatically.

### Weekly retraining fails

1. Go to **GitHub → Actions → Model Maintenance** and inspect the failed run.
2. Common causes:
   - Dataset source is down (GitHub mirror unavailable)
   - `GridSearchCV` timeout
   - `pytest test_app.py` fails after promotion (real regression — correctly blocked before commit)
3. Re-run with `workflow_dispatch` + `dry_run: true` to diagnose without touching production.

### Running maintenance locally

```bash
# Evaluate only — nothing is written or promoted
python retrain_pipeline.py --dry-run

# Full run — may promote and update model/ artifacts
python retrain_pipeline.py
```
