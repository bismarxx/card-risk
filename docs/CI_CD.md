# CI/CD Pipeline

## Overview

Two independent automated workflows handle different concerns:

| Workflow | File | Trigger | Purpose |
|---|---|---|---|
| **CI** | `ci.yml` | push / PR to `main` | Validate code changes before merging |
| **Maintenance** | `maintenance.yml` | Weekly cron + manual | Keep the model fresh |

---

## CI Workflow (`ci.yml`)

### Triggers
- `push` to `main`
- `pull_request` targeting `main`
- `workflow_dispatch` (manual run from GitHub UI)

### Jobs

```
push / PR to main
      │
      ▼
┌───────────────────────────────────────────┐
│ Job: test                                 │
│  1. Checkout repository                   │
│  2. Setup Python 3.11 (with pip cache)    │
│  3. pip install -r requirements.txt       │
│  4. python train.py   ← trains the model  │
│  5. pytest test_app.py -v       (11 tests)│
│  6. pytest test_maintenance_ci.py (10)    │
└───────────────────────────────────────────┘
      │  needs: test
      ▼
┌───────────────────────────────────────────┐
│ Job: docker-build                         │
│  docker/build-push-action (push: false)   │
│  Validates Dockerfile builds cleanly      │
│  without publishing the image             │
└───────────────────────────────────────────┘
      │  needs: [test, docker-build]
      │  only on: push to main
      ▼
┌───────────────────────────────────────────┐
│ Job: deploy (appleboy/ssh-action)         │
│  SSH into production server               │
│  git fetch + git reset --hard origin/main │
│  docker compose down                      │
│  docker compose up -d --build             │
│  docker image prune -f                    │
└───────────────────────────────────────────┘
```

> The `deploy` job only runs on `push` to `main` (not on PRs), so branch tests never trigger a deployment.

### Why train inside CI?
`test_app.py` loads the serialized model for inference tests — it needs a `.joblib` file to exist. Running `train.py` inside CI ensures tests always run against a freshly trained model, avoiding stale artifact issues.

---

## Maintenance Workflow (`maintenance.yml`)

See [MAINTENANCE.md](./MAINTENANCE.md) for the full retraining pipeline.

The maintenance workflow redeploys using the **same SSH mechanism** as the CI deploy job — same secrets, same script. The only difference is *what* triggers the redeploy: a model promotion commit (with `[skip ci]`) instead of a code push.

---

## Why `[skip ci]` on model commits?

When the maintenance pipeline promotes a new model, it commits the new `.joblib` to `main`. Without `[skip ci]`, this push would trigger the CI workflow, which would:
1. Run `python train.py` → train a *new* model from scratch
2. Potentially overwrite the just-promoted champion

The `[skip ci]` tag in the commit message tells GitHub Actions to skip all workflows for that push, preventing the loop.

---

## Test Suite at a Glance

### `test_app.py` — 11 tests

| Class | Tests | What it covers |
|---|---|---|
| `TestInputValidation` | 6 | Feature schema, value ranges, categorical codes |
| `TestModelPrediction` | 4 | Model file exists, valid probability, binary label, deterministic output |
| `TestStreamlitApp` | 2 | App loads without exceptions, predict button returns a result |

### `test_maintenance_ci.py` — 10 tests

| Class | Tests | What it covers |
|---|---|---|
| `TestMaintenanceCase1PromotionWhenBetter` | 2 | Candidate with better ROC-AUC is promoted; first model always promotes |
| `TestMaintenanceCase2NoPromotionWhenNotBetter` | 3 | Worse / equal / within-tolerance candidates are NOT promoted |
| `TestCICase3WorkflowsAreValid` | 5 | Workflow YAML validity, triggers, and expected commands |

**Total: 21 tests — all passing.**

```bash
pytest test_app.py test_maintenance_ci.py -v
# 21 passed in ~20s
```

---

## Adding a New Model Candidate

1. Add an entry to `get_candidate_models()` in [`train.py`](../train.py).
2. That's it — `retrain_pipeline.py` imports the function directly; the promotion logic is model-agnostic.
3. Push to a branch, open a PR — CI will run the full test suite automatically.
