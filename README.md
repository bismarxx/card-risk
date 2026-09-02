# Cardiovascular Risk Predictor

Proyecto del curso *Aprendizaje de Máquina I* — Maestría en Ciencia de Datos, UNA Puno.
Aplicación con un componente de predicción (riesgo cardiovascular) desplegada en producción, con procesos de **mantenimiento** e **integración continua** automatizados.

## Mapeo con las indicaciones del curso

### Unidad I — Entrega de avance / parcial
| Requisito | Dónde está |
|---|---|
| Funcionamiento de la aplicación en inglés | `app.py` |
| Dataset, entrenamiento, features e hiperparámetros optimizados | `train.py`, `data/heart.csv`, `model/model_metadata.json` |
| Despliegue / puesta en producción | `Dockerfile`, `.dockerignore`, guía de despliegue (Render / Hugging Face Spaces) |
| Pruebas de funcionamiento (≥2) | `test_app.py` (11 pruebas) |
| Informe detallado de entrenamiento y despliegue | `INFORME.md` |

### Unidad II — Producto final
| Requisito | Dónde está |
|---|---|
| Funcionamiento de la aplicación en inglés | `app.py` (sin cambios) |
| Despliegue / puesta en producción | Igual que Unidad I |
| Flujos de mantenimiento automatizados | `.github/workflows/maintenance.yml` + `retrain_pipeline.py` |
| Flujos de integración continua automatizados | `.github/workflows/ci.yml` |
| Pruebas de mantenimiento e integración continua (3 casos) | `test_maintenance_ci.py` |
| Informe técnico para equipo de TI | `INFORME_TECNICO_MANTENIMIENTO_CI.md` |

## Estructura del repositorio

```
app.py                                  # App Streamlit (UI en inglés)
train.py                                # Entrenamiento inicial
retrain_pipeline.py                     # Pipeline de mantenimiento (reentrena + promueve)
test_app.py                             # Pruebas funcionales (11)
test_maintenance_ci.py                  # Pruebas de mantenimiento y CI (3 casos / 10 tests)
requirements.txt / Dockerfile / .dockerignore
.github/workflows/ci.yml                # CI: pytest + build de imagen
.github/workflows/maintenance.yml       # Mantenimiento: reentrenamiento semanal
data/heart.csv                          # Dataset (UCI Heart Disease)
model/                                  # Modelo en producción + registro de versiones
INFORME.md                              # Informe Unidad I
INFORME_TECNICO_MANTENIMIENTO_CI.md     # Informe técnico Unidad II
```

## Quickstart

```bash
pip install -r requirements.txt
python train.py                         # entrena y guarda el modelo inicial
streamlit run app.py                    # http://localhost:8501

# Pruebas
pytest test_app.py -v                   # funcionamiento de la app / modelo
pytest test_maintenance_ci.py -v        # mantenimiento + CI (3 casos)

# Pipeline de mantenimiento (manual / lo automatiza maintenance.yml)
python retrain_pipeline.py --dry-run    # solo evalúa
python retrain_pipeline.py              # reentrena y promueve si mejora
```

## Despliegue

Ver `INFORME.md` sección 10 para la guía paso a paso (Render y Hugging Face Spaces vía Docker).

## CI/CD en un vistazo

```
push/PR a main ──► ci.yml ──► pytest (app + mantenimiento/CI) ──► docker build (validación)

cron semanal ──► maintenance.yml ──► retrain_pipeline.py ──► ¿mejora el modelo?
                                                                ├─ sí → commit + push + redeploy
                                                                └─ no → se descarta, queda en el historial
```

Detalle completo de ambos flujos, herramientas requeridas, organización del código y runbook operativo en `INFORME_TECNICO_MANTENIMIENTO_CI.md`.
