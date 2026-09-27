# Informe Técnico — Mantenimiento e Integración Continua
## Cardiovascular Risk Predictor · Unidad II

**Audiencia:** equipo de TI responsable de operar, mantener e integrar cambios sobre esta aplicación en producción.
**Alcance:** este documento cubre los procesos automatizados de mantenimiento e integración continua (Unidad II). El entrenamiento inicial, el dataset y el despliegue manual se documentan en `INFORME.md`.

---

## 1. Funcionamiento de la aplicación

La aplicación (`app.py`, Streamlit) expone un formulario clínico en inglés y devuelve una probabilidad de riesgo cardiovascular usando el modelo en producción.

**Flujo de una predicción:**

```
Usuario rellena 13 campos clínicos
        ↓
build_feature_row()  → construye 17 features (13 originales + 4 derivadas)
        ↓
model.predict_proba()  → probabilidad ∈ [0, 1]
        ↓
Resultado: "High risk" (≥ 0.5) o "Low risk" (< 0.5) + barra de progreso
```

**Características derivadas** (calculadas en `build_feature_row`, idénticas al entrenamiento):

| Feature | Fórmula |
|---|---|
| `bp_chol_ratio` | `trestbps / chol` |
| `max_hr_expected` | `220 - age` |
| `hr_reserve_deficit` | `max_hr_expected - thalach` |
| `age_group` | Bins: `<40`, `40-49`, `50-59`, `60+` |

El modelo serializado (`model/cardio_risk_model.joblib`) es un Pipeline de scikit-learn que incluye el preprocesamiento completo (`StandardScaler` + `OneHotEncoder`), garantizando consistencia exacta entre entrenamiento e inferencia.

---

## 2. Herramientas y plataformas necesarias

| Categoría | Herramienta | Propósito |
|---|---|---|
| Lenguaje | Python 3.11 | Runtime de la app, entrenamiento y tests |
| Interfaz web | Streamlit | UI interactiva |
| ML | scikit-learn, joblib | Entrenamiento, pipeline de preprocesamiento, serialización |
| Control de versiones | GitHub | Repositorio único fuente de verdad para código, modelo y workflows |
| CI/CD | GitHub Actions | Ejecuta pruebas, construye la imagen y corre el reentrenamiento programado |
| Contenedorización | Docker / Docker Compose | Empaqueta la app de forma reproducible y la levanta en producción |
| Servidor de producción | Ubuntu Cloud Server (vía SSH) | Sirve el contenedor; el deploy se hace con `docker compose up --build` |
| Testing | pytest, `streamlit.testing.v1.AppTest`, PyYAML | Pruebas funcionales, de mantenimiento y validación de configuración CI |
| Registro de modelos | `model/registry.json` + `model/registry/` | Trazabilidad de versiones, métricas y decisiones de promoción |

**Secrets a configurar en GitHub** (`Settings → Secrets and variables → Actions`):

| Secret | Requerido | Uso |
|---|---|---|
| `GITHUB_TOKEN` | Automático | Permite al workflow de mantenimiento hacer commit/push del modelo promovido |
| `SERVER_HOST` | Sí | IP o dominio del servidor de producción |
| `SERVER_USER` | Sí | Usuario SSH (ej. `root`) |
| `SSH_PRIVATE_KEY` | Sí | Llave privada RSA/ED25519 para autenticación SSH sin contraseña |
| `SERVER_PORT` | Opcional | Puerto SSH (se asume 22 si no se define) |

---

## 3. Organización del código fuente

```
card-risk/
├── app.py                       # Interfaz Streamlit (UI en inglés)
├── train.py                     # Entrenamiento inicial (manual / build-time)
├── retrain_pipeline.py          # Pipeline de MANTENIMIENTO: reentrena, evalúa y promueve
├── test_app.py                  # Pruebas funcionales de la app y el modelo (11 casos)
├── test_maintenance_ci.py       # Pruebas de mantenimiento y CI (3 casos exigidos)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml               # CI: test + docker-build + deploy (push/PR a main)
│       └── maintenance.yml      # Mantenimiento: reentrenamiento (cron domingos 03:00 UTC)
├── data/
│   └── heart.csv                # Dataset UCI Heart Disease
├── model/
│   ├── cardio_risk_model.joblib # Alias de PRODUCCIÓN (el que consume app.py)
│   ├── model_metadata.json      # Métricas e hiperparámetros del modelo en producción
│   ├── registry.json            # Registro: campeón actual + historial de versiones
│   └── registry/                # Artefactos versionados (model_v1.joblib, v2, ...)
├── INFORME.md                   # Dataset, entrenamiento y despliegue (Unidad I)
└── INFORME_TECNICO_MANTENIMIENTO_CI.md   # Este documento (Unidad II)
```

**Principio de diseño clave:** `app.py` siempre lee un único archivo fijo (`model/cardio_risk_model.joblib`). Ni la app ni el `Dockerfile` necesitan saber qué versión es la campeona — eso lo gestiona exclusivamente `retrain_pipeline.py` a través del registro. Esto desacopla el ciclo de vida del modelo del ciclo de vida del código de la aplicación.

---

## 4. Consideraciones de despliegue inicial

1. **Build de la imagen:** el `Dockerfile` ejecuta `RUN python train.py` en tiempo de build. La primera imagen siempre contiene un modelo entrenado, incluso partiendo de un repositorio sin artefactos binarios.
2. **Dataset:** `train.py` descarga `heart.csv` automáticamente desde un mirror público de GitHub. El CSV también está versionado en el repo como fallback si el servidor no tiene acceso a internet durante el build.
3. **Variables de entorno:** Streamlit lee `PORT` desde el entorno (`--server.port=${PORT:-8501}`), haciendo el contenedor portable entre plataformas sin cambios de código.
4. **Healthcheck:** el `Dockerfile` incluye un `HEALTHCHECK` contra `/_stcore/health` para que el orchestrador detecte si el proceso está sano antes de enrutar tráfico.
5. **Arranque en producción (primera vez):**
   ```bash
   git clone <repo> /root/projects/card-risk
   cd /root/projects/card-risk
   docker compose up -d --build
   ```
   Cada redeploy posterior (por CI o por mantenimiento) ejecuta `git pull + docker compose up` automáticamente vía SSH desde GitHub Actions.

---

## 5. Flujo de trabajo automatizado de INTEGRACIÓN CONTINUA (CI)

**Archivo:** `.github/workflows/ci.yml`

**Objetivo:** evitar que cambios de código rompan la aplicación, las pruebas o el build de producción antes de fusionarlos a `main`.

**Disparadores:**
- `push` a `main`
- `pull_request` contra `main`
- `workflow_dispatch` (manual)

**Jobs y pasos:**

```
push / PR a main
      │
      ▼
┌──────────────────────────────────────────────┐
│ Job: test                                    │
│  1. Checkout                                 │
│  2. Setup Python 3.11 + cache pip            │
│  3. pip install -r requirements.txt          │
│  4. python train.py  (genera modelo en CI)   │
│  5. pytest test_app.py -v      (11 tests)    │
│  6. pytest test_maintenance_ci.py -v (10)    │
└──────────────────────────────────────────────┘
      │ needs: test
      ▼
┌──────────────────────────────────────────────┐
│ Job: docker-build                            │
│  docker/build-push-action (push: false)      │
│  → valida que el Dockerfile siga siendo OK   │
└──────────────────────────────────────────────┘
      │ needs: [test, docker-build] + push a main
      ▼
┌──────────────────────────────────────────────┐
│ Job: deploy (appleboy/ssh-action)            │
│  cd /root/projects/card-risk                 │
│  git fetch origin main                       │
│  git reset --hard origin/main                │
│  docker compose down                         │
│  docker compose up -d --build                │
│  docker image prune -f                       │
└──────────────────────────────────────────────┘
```

**Separación de responsabilidades:** CI valida *cambios de código* en cada push/PR (rápido, ~2-5 min); Mantenimiento valida *frescura del modelo* de forma programada (semanal, ~20-40 min por el `GridSearchCV`). Esto evita que un cambio de texto en la UI dispare un reentrenamiento, y evita que el reentrenamiento bloquee la fusión de código no relacionado.

---

## 6. Flujo de trabajo automatizado de MANTENIMIENTO

**Archivo:** `.github/workflows/maintenance.yml`

**Objetivo:** mantener el modelo actualizado sin intervención manual, con una salvaguarda que impide degradar el modelo en producción.

**Disparadores:**
- `schedule`: cron semanal — domingos a las 03:00 UTC (`0 3 * * 0`).
- `workflow_dispatch`: ejecución manual bajo demanda (incluye opción `dry_run`).

**Pasos del flujo:**

```
cron domingo 03:00 UTC / workflow_dispatch
              │
              ▼
   python retrain_pipeline.py [--dry-run]
              │
    ┌─────────────────────────────────────┐
    │ 1. download_dataset()               │  actualiza CSV si hay nueva versión
    │ 2. load_and_engineer_features()     │  feature engineering idéntico al inicial
    │ 3. train_and_select_best_model()    │  GridSearchCV 3 modelos, CV 5-fold
    │ 4. evaluate_model()                 │  métricas en hold-out
    │ 5. decide_promotion()               │  candidato.roc_auc > campeón + 0.001?
    └─────────────────────────────────────┘
              │
        ┌─────┴──────┐
      SI mejora    NO mejora
        │            │
        ▼            ▼
  Promueve:      Descarta candidato:
  copia .joblib  registra en history
  registry.json  producción SIN cambios
  metadata.json  (rollback seguro)
        │
        ▼ (solo si no es dry_run)
  pytest test_app.py  ← validación de regresión
        │
        ▼ (si hubo cambios en model/)
  git commit [skip ci] + git push
        │
        ▼ (si hubo promoción)
  appleboy/ssh-action → servidor Ubuntu
    git fetch + git reset --hard origin/main
    docker compose up -d --build
    docker image prune -f
```

**`[skip ci]`** en el commit evita que el workflow de CI se dispare sobre el push del modelo, previniendo un loop infinito.

**Lógica de promoción (`decide_promotion`, función pura y testeada):**
- Sin campeón previo → siempre promueve el primer modelo.
- Con campeón → promueve solo si `candidato.roc_auc - campeón.roc_auc > 0.001`.
- La tolerancia de `0.001` evita "promociones" por varianza numérica del split o del optimizador.

**Exit codes del script:**
- `0` → modelo promovido (o dry-run sin errores).
- `1` → candidato no superó al campeón; producción sin cambios.

---

## 7. Pruebas de funcionamiento de mantenimiento e integración continua

Implementadas en `test_maintenance_ci.py` (**10 pruebas** en los **3 casos** exigidos por la rúbrica):

### Caso 1 — Promoción cuando el candidato mejora
**Clase:** `TestMaintenanceCase1PromotionWhenBetter`

| Prueba | Qué valida |
|---|---|
| `test_candidate_with_better_roc_auc_is_promoted` | Un candidato con ROC-AUC superior al campeón en más de la tolerancia SÍ se promueve |
| `test_no_champion_yet_always_promotes_first_model` | El primer modelo (sin campeón previo) siempre se promueve |

### Caso 2 — No promoción cuando el candidato no mejora (rollback seguro)
**Clase:** `TestMaintenanceCase2NoPromotionWhenNotBetter`

| Prueba | Qué valida |
|---|---|
| `test_candidate_with_worse_roc_auc_is_not_promoted` | ROC-AUC inferior NO promueve |
| `test_candidate_with_equal_roc_auc_is_not_promoted` | ROC-AUC igual NO promueve (empates no cuentan) |
| `test_candidate_within_tolerance_is_not_promoted` | Mejora menor a la tolerancia (`0.001/2`) NO promueve |

### Caso 3 — Validez de los workflows de integración continua
**Clase:** `TestCICase3WorkflowsAreValid`

| Prueba | Qué valida |
|---|---|
| `test_ci_workflow_exists_and_is_valid_yaml` | `ci.yml` existe y es YAML válido |
| `test_ci_workflow_triggers_on_push_and_pull_request` | `ci.yml` tiene disparadores `push` y `pull_request` |
| `test_ci_workflow_runs_the_test_suite` | `ci.yml` ejecuta ambos conjuntos de pytest |
| `test_maintenance_workflow_exists_and_is_scheduled` | `maintenance.yml` existe con `schedule` y `workflow_dispatch` |
| `test_maintenance_workflow_runs_retrain_pipeline` | `maintenance.yml` ejecuta `retrain_pipeline.py` |

**Resultado de la última ejecución: 21 passed (11 de `test_app.py` + 10 de `test_maintenance_ci.py`)**

Estas pruebas corren automáticamente dentro del job `test` de `ci.yml`, cerrando el ciclo: **el pipeline de CI se prueba a sí mismo en cada ejecución.**

```bash
pip install -r requirements.txt
python train.py                               # genera el modelo si no existe
pytest test_app.py test_maintenance_ci.py -v  # 21 passed
```

---

## 8. Runbook operativo para el equipo de TI

**Modelo degradado / predicciones inconsistentes:**
1. Revisar `model/registry.json` → campo `champion` (versión y métricas vigentes).
2. Revisar `history` para identificar si hubo una promoción reciente sospechosa.
3. Rollback manual: copiar `model/registry/model_v{N}.joblib` sobre `model/cardio_risk_model.joblib`, actualizar `champion` en `registry.json`, hacer commit/push — el CI redesplegará.

**Reentrenamiento semanal falla:**
1. Revisar logs del job `retrain` en `GitHub → Actions`.
2. Causas típicas: fuente del dataset caída, timeout de `GridSearchCV`, fallo de `pytest test_app.py` post-promoción (indica regresión real, correctamente bloqueada antes del commit).
3. Re-disparar con `workflow_dispatch` y `dry_run: true` para diagnosticar sin afectar producción.

**Agregar un nuevo modelo candidato (ej. XGBoost):**
1. Agregar la entrada en `get_candidate_models()` en `train.py`.
2. No se toca `retrain_pipeline.py` ni los workflows — la lógica de promoción es agnóstica al tipo de modelo.

**Redeploy manual de emergencia:**
```bash
ssh <SERVER_USER>@<SERVER_HOST>
cd /root/projects/card-risk
git fetch origin main && git reset --hard origin/main
docker compose down && docker compose up -d --build
docker image prune -f
```

---

## 9. Trazabilidad y auditoría del modelo

Todo cambio al modelo en producción queda registrado en tres lugares consistentes:

1. **Git history** — cada promoción es un commit: `chore(maintenance): retrain and promote new champion model [skip ci]`.
2. **`model/registry.json`** — historial estructurado con métricas, hiperparámetros, timestamp y estado de promoción de cada corrida.
3. **GitHub Actions artifacts** — cada ejecución del workflow de mantenimiento sube `registry.json` como artefacto descargable (`model-registry-{run_id}`), incluso si no hubo promoción.

**Estado actual del modelo en producción:**

| Campo | Valor |
|---|---|
| Versión | v1 |
| Algoritmo | Logistic Regression (`C=0.1`) |
| ROC-AUC (test) | 0.8972 |
| Recall (sensibilidad) | 0.9091 |
| Promovido en | 2026-09-27T02:50 UTC |

---

## 1. Funcionamiento de la aplicación (resumen)

La aplicación (`app.py`, Streamlit) expone un formulario clínico y devuelve una probabilidad de riesgo cardiovascular. **Toda la interfaz de usuario está en inglés** (etiquetas, botones, mensajes de resultado), tal como exige la rúbrica. El detalle funcional completo está en `INFORME.md`, sección 7.

## 2. Despliegue / puesta en producción (resumen)

La app se contenedoriza con `Dockerfile` (basado en `python:3.11-slim`) y se despliega en **Render** o **Hugging Face Spaces**, ambos vía el mismo `Dockerfile` (guía paso a paso ya entregada previamente y documentada en `INFORME.md`, sección 10). El contenedor lee el puerto desde la variable de entorno `PORT`, lo que lo hace portable entre plataformas sin cambios de código.

---

## 3. Herramientas y plataformas necesarias

| Categoría | Herramienta | Propósito |
|---|---|---|
| Control de versiones | GitHub | Repositorio único fuente de verdad para código, modelo versionado y workflows |
| CI/CD | GitHub Actions | Ejecuta pruebas, construye la imagen Docker y corre el reentrenamiento programado |
| Contenedorización | Docker | Empaqueta la app y sus dependencias de forma reproducible |
| Hosting / producción | Render o Hugging Face Spaces | Sirve el contenedor públicamente |
| ML / serialización | scikit-learn, joblib | Entrenamiento, pipeline de preprocesamiento e inferencia, serialización del modelo |
| Testing | pytest, `streamlit.testing.v1.AppTest`, PyYAML | Pruebas funcionales, de mantenimiento y de validación de configuración CI |
| Registro de modelos | `model/registry.json` (registry casero basado en JSON + archivos versionados) | Trazabilidad de versiones de modelo, métricas y decisiones de promoción |

No se requieren plataformas de pago: todo el stack (GitHub Actions incluido, dentro de límites gratuitos razonables para un proyecto académico) es gratuito o de nivel free-tier.

**Secretos a configurar en el repositorio (Settings → Secrets and variables → Actions):**

| Secreto | Requerido | Uso |
|---|---|---|
| `GITHUB_TOKEN` | Automático (lo provee GitHub) | Permite al workflow de mantenimiento hacer commit/push del modelo promovido |
| `RENDER_DEPLOY_HOOK_URL` | Opcional | Si se define, el workflow de mantenimiento dispara un redeploy automático en Render tras promover un modelo nuevo |

---

## 4. Organización del código fuente

```
cardio_risk/
├── app.py                      # Interfaz Streamlit (UI en inglés)
├── train.py                    # Entrenamiento inicial (manual / build-time)
├── retrain_pipeline.py         # Pipeline de MANTENIMIENTO: reentrena, evalúa y promueve
├── test_app.py                 # Pruebas funcionales de la app y el modelo (11 casos)
├── test_maintenance_ci.py      # Pruebas de mantenimiento y CI (3 casos exigidos)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .github/
│   └── workflows/
│       ├── ci.yml               # Integración continua (push / PR)
│       └── maintenance.yml      # Mantenimiento programado (cron semanal)
├── data/
│   └── heart.csv                # Dataset (se descarga automáticamente si falta)
├── model/
│   ├── cardio_risk_model.joblib # Alias de PRODUCCIÓN (el que consume app.py)
│   ├── model_metadata.json      # Métricas y config del modelo en producción
│   ├── registry.json            # Registro: campeón actual + historial de versiones
│   └── registry/                # Artefactos versionados (model_v1.joblib, v2, ...)
├── INFORME.md                   # Informe de entrenamiento y despliegue (Unidad I)
└── INFORME_TECNICO_MANTENIMIENTO_CI.md   # Este documento (Unidad II)
```

**Principio de diseño:** `app.py` siempre lee un único archivo fijo, `model/cardio_risk_model.joblib` (el "alias de producción"). Ni la app ni el `Dockerfile` necesitan saber qué versión específica es la campeona — eso lo gestiona exclusivamente `retrain_pipeline.py` a través del registro. Esto desacopla el ciclo de vida del modelo del ciclo de vida del código de la aplicación.

---

## 5. Consideraciones de despliegue inicial

- El `Dockerfile` ejecuta `RUN python train.py` en tiempo de build, por lo que la primera imagen siempre contiene un modelo entrenado, aunque el repositorio no incluya artefactos binarios versionados.
- Si se prefiere **no** reentrenar en cada build (para acelerar despliegues), se puede versionar `model/cardio_risk_model.joblib` y `model/registry.json` directamente en Git (son archivos pequeños, <10 KB) y quitar esa línea del `Dockerfile`; el pipeline de mantenimiento seguiría actualizándolos vía commits automáticos.
- El dataset (`data/heart.csv`) se descarga desde un mirror público de GitHub la primera vez; si la plataforma de despliegue restringe el acceso a internet durante el build, debe versionarse el CSV en el repo (ya se incluye en este proyecto por robustez).
- El healthcheck del contenedor (`/_stcore/health`) permite a Render/HF Spaces detectar si el proceso de Streamlit está sano antes de enrutar tráfico.

---

## 6. Flujo de trabajo automatizado de MANTENIMIENTO

**Objetivo:** mantener el modelo actualizado sin intervención manual, con una salvaguarda que impide degradar el modelo en producción.

**Disparadores** (`.github/workflows/maintenance.yml`):
- `schedule`: cron semanal (lunes 03:00 UTC).
- `workflow_dispatch`: ejecución manual bajo demanda (incluye opción `dry_run`).

**Pasos del flujo** (implementados en `retrain_pipeline.py`):

1. Descarga/actualiza el dataset (reutiliza `train.download_dataset`).
2. Reentrena y vuelve a optimizar hiperparámetros con `GridSearchCV` sobre los 3 modelos candidatos (idéntico proceso al entrenamiento inicial, ver `INFORME.md` sección 4).
3. Evalúa el candidato en un hold-out (`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`).
4. **Decisión de promoción** (`decide_promotion`, función pura y testeada): el candidato reemplaza al campeón actual **solo si** su ROC-AUC lo supera por más de `0.001` (tolerancia contra ruido numérico). Si no hay campeón previo, el primer modelo siempre se promueve.
5. Si se promueve:
   - Se guarda una copia versionada en `model/registry/model_v{N}.joblib`.
   - Se sobrescribe el alias de producción `model/cardio_risk_model.joblib`.
   - Se actualiza `model/registry.json` (nuevo campeón) y `model/model_metadata.json`.
   - El workflow corre `pytest test_app.py` contra el modelo ya promovido como validación de regresión.
   - Se hace `git commit` + `git push` automático de los artefactos (`[skip ci]` para no disparar un loop con el workflow de CI).
   - Si está configurado el secreto `RENDER_DEPLOY_HOOK_URL`, se dispara un redeploy automático.
6. Si **no** se promueve: el candidato queda registrado en el historial de auditoría (`registry.json → history`) pero el modelo en producción no cambia — comportamiento de **rollback seguro** ante una corrida de mala calidad (por ejemplo, un mal split o un dataset corrupto en la fuente).

Este flujo es completamente reproducible localmente:
```bash
python retrain_pipeline.py --dry-run   # solo evalúa, no escribe ni promueve
python retrain_pipeline.py             # corrida real
```

---

## 7. Flujo de trabajo automatizado de INTEGRACIÓN CONTINUA (CI)

**Objetivo:** evitar que cambios de código rompan la aplicación, las pruebas o el build de producción antes de fusionarlos a `main`.

**Disparadores** (`.github/workflows/ci.yml`):
- `push` a `main`.
- `pull_request` contra `main`.
- `workflow_dispatch` (manual).

**Jobs:**

| Job | Pasos | Propósito |
|---|---|---|
| `test` | Instala dependencias → entrena el modelo → corre `pytest test_app.py` → corre `pytest test_maintenance_ci.py` | Valida funcionamiento de la app/modelo y de la lógica de mantenimiento/CI en cada cambio |
| `docker-build` (depende de `test`) | `docker/build-push-action` con `push: false` | Verifica que la imagen de producción sigue construyendo correctamente, sin publicarla |

**Separación de responsabilidades:** CI valida *cambios de código* en cada push/PR (rápido, minutos); Mantenimiento valida *frescura del modelo* de forma programada (semanal, más costoso por el `GridSearchCV`). Mantener ambos flujos separados evita que un simple cambio de texto en la UI dispare un reentrenamiento innecesario, y evita que el reentrenamiento bloquee la fusión de código no relacionado.

---

## 8. Pruebas de funcionamiento de mantenimiento e integración continua

Implementadas en `test_maintenance_ci.py` (10 pruebas, agrupadas en los **3 casos** exigidos por la rúbrica):

| Caso | Clase de prueba | Qué valida |
|---|---|---|
| **1. Promoción cuando el candidato mejora** | `TestMaintenanceCase1PromotionWhenBetter` | La función pura `decide_promotion` promueve un candidato con mejor ROC-AUC, y siempre promueve el primer modelo (sin campeón previo) |
| **2. No promoción cuando el candidato no mejora** | `TestMaintenanceCase2NoPromotionWhenNotBetter` | Un candidato peor, igual, o con una mejora por debajo del margen de tolerancia, **no** reemplaza al campeón (protege el modelo en producción) |
| **3. Validez de los workflows de CI** | `TestCICase3WorkflowsAreValid` | Los YAML de `.github/workflows/` existen, son sintácticamente válidos, tienen los disparadores correctos (`push`/`pull_request` en CI; `schedule`/`workflow_dispatch` en mantenimiento) y ejecutan los comandos esperados (`pytest`, `retrain_pipeline.py`) |

Ejecución:
```bash
pip install -r requirements.txt
pytest test_maintenance_ci.py -v   # 10 passed
pytest test_app.py test_maintenance_ci.py -v   # 21 passed (suite completa)
```

Estas pruebas corren automáticamente dentro del propio job `test` de `ci.yml`, cerrando el ciclo: **el pipeline de CI se prueba a sí mismo** en cada ejecución.

---

## 9. Runbook operativo (para el equipo de TI)

**Escenario: el modelo en producción parece estar degradado / dando predicciones inconsistentes.**
1. Revisar `model/registry.json` → campo `champion` para ver versión y métricas vigentes.
2. Revisar el historial (`history`) para ver si hubo una promoción reciente sospechosa.
3. Rollback manual: copiar el artefacto de una versión anterior desde `model/registry/model_v{N}.joblib` sobre `model/cardio_risk_model.joblib`, actualizar `champion` en `registry.json`, hacer commit y push (o disparar `workflow_dispatch` de `maintenance.yml` con un dataset corregido).

**Escenario: el reentrenamiento semanal falla.**
1. Revisar los logs del job `retrain` en la pestaña *Actions* de GitHub.
2. Causas típicas: fuente del dataset caída (mirror de GitHub), tiempo de ejecución excedido por `GridSearchCV`, fallo de `pytest test_app.py` post-promoción (indicaría una regresión real, correctamente bloqueada antes del commit).
3. Se puede re-disparar manualmente con `workflow_dispatch` y `dry_run: true` para diagnosticar sin afectar producción.

**Escenario: se necesita agregar un nuevo modelo candidato (por ejemplo, XGBoost).**
1. Agregar la entrada correspondiente en `get_candidate_models()` (`train.py`), reutilizada automáticamente tanto por `train.py` como por `retrain_pipeline.py`.
2. No se requiere tocar `retrain_pipeline.py` ni los workflows — la lógica de selección/promoción es agnóstica al tipo de modelo.

---

## 10. Resumen de trazabilidad y auditoría

Todo cambio al modelo en producción queda registrado en tres lugares consistentes entre sí:
1. **Git history** — cada promoción es un commit (`chore(maintenance): retrain and promote new champion model`).
2. **`model/registry.json`** — historial estructurado con métricas, hiperparámetros y timestamp de cada corrida (promovida o no).
3. **GitHub Actions artifacts** — cada ejecución del workflow de mantenimiento sube `registry.json` como artefacto descargable, incluso si no hubo cambios de código.
