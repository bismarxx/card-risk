# Deployment Guide

## Prerequisites

- Docker & Docker Compose installed on the target server
- SSH access to the server
- GitHub repository with the 4 secrets configured (see below)

## Required GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|---|---|
| `SERVER_HOST` | IP address or domain of the production server |
| `SERVER_USER` | SSH username (e.g. `root`) |
| `SSH_PRIVATE_KEY` | Private RSA/ED25519 key (the server must have the matching public key in `authorized_keys`) |
| `SERVER_PORT` | SSH port — defaults to `22` if omitted |

> `GITHUB_TOKEN` is provided automatically by GitHub Actions — no manual setup needed.

---

## Initial Deployment (first time)

```bash
# On the server
git clone <your-repo-url> /root/projects/card-risk
cd /root/projects/card-risk
docker compose up -d --build
```

The `Dockerfile` runs `python train.py` at build time, so the first image ships with a fully trained model — no manual training step needed on the server.

After the first deploy, **all subsequent deploys are automatic** via GitHub Actions on every push to `main`.

---

## How Automatic Redeploy Works

Every push to `main` that passes CI triggers the `deploy` job in [`ci.yml`](../.github/workflows/ci.yml):

```
GitHub Actions runner  →  SSH into server  →  git pull + docker compose up
```

The exact script run on the server:

```bash
cd /root/projects/card-risk
git fetch origin main
git reset --hard origin/main
docker compose down
docker compose up -d --build
docker image prune -f
```

The same script also runs when the [maintenance pipeline](./MAINTENANCE.md) promotes a new model.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8501` | Port Streamlit listens on — injected by the platform automatically |

The `Dockerfile` and `docker-compose.yml` are already configured to read `PORT` from the environment. No code changes are needed when deploying to a different port.

---

## Healthcheck

The `Dockerfile` includes a healthcheck against Streamlit's internal health endpoint:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1
```

This lets Docker (and your orchestrator) know when the app is ready to serve traffic.

---

## Manual Emergency Redeploy

If you need to force a redeploy without pushing new code:

```bash
ssh <SERVER_USER>@<SERVER_HOST>
cd /root/projects/card-risk
git fetch origin main && git reset --hard origin/main
docker compose down && docker compose up -d --build
docker image prune -f
```

Or trigger the CI workflow manually from **GitHub → Actions → CI → Run workflow**.
