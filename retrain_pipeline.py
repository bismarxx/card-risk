"""
retrain_pipeline.py
--------------------
Pipeline de MANTENIMIENTO automatizado del modelo de riesgo cardiovascular.

A diferencia de `train.py` (entrenamiento inicial, ejecución manual/build-time),
este script está pensado para ejecutarse de forma recurrente (cron / GitHub
Actions) y encapsula la lógica de un "model registry" simplificado:

    1. Descarga/actualiza el dataset.
    2. Reentrena y vuelve a optimizar hiperparámetros (reutiliza train.py).
    3. Evalúa el nuevo modelo candidato en un hold-out.
    4. Compara sus métricas contra el "campeón" (modelo actualmente en
       producción, registrado en model/registry.json).
    5. Si el candidato mejora al campeón -> lo promueve: lo versiona,
       actualiza el alias de producción (model/cardio_risk_model.joblib) y
       actualiza el registro.
    6. Si NO lo mejora -> lo descarta como campeón (queda solo en el
       historial de auditoría) y el modelo en producción no cambia
       (comportamiento de "rollback seguro" ante una corrida de mala calidad).

Diseñado para poder testearse sin reentrenar (la función `decide_promotion`
es pura y se prueba de forma aislada en test_maintenance_ci.py).

Uso:
    python retrain_pipeline.py                # corrida real completa
    python retrain_pipeline.py --dry-run       # solo evalúa, no promueve ni escribe
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib

import train as train_module  # reutiliza download_dataset, engineering, tuning, eval

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model"
REGISTRY_DIR = MODEL_DIR / "registry"
REGISTRY_PATH = MODEL_DIR / "registry.json"
PRODUCTION_MODEL_PATH = MODEL_DIR / "cardio_risk_model.joblib"  # alias que consume app.py
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Métrica usada para decidir promoción. Debe existir en el dict de métricas
# devuelto por train.evaluate_model().
PROMOTION_METRIC = "roc_auc"
# Margen mínimo de mejora exigido para evitar "promociones" por ruido numérico.
PROMOTION_TOLERANCE = 0.001


# --------------------------------------------------------------------------
# Registry helpers
# --------------------------------------------------------------------------

def load_registry() -> dict:
    """Carga el registro de modelos; si no existe, lo inicializa desde cero."""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"champion": None, "history": []}


def save_registry(registry: dict) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


def next_version(registry: dict) -> str:
    n = len(registry.get("history", [])) + 1
    return f"v{n}"


def decide_promotion(
    champion: Optional[dict],
    candidate_metrics: dict,
    metric: str = PROMOTION_METRIC,
    tolerance: float = PROMOTION_TOLERANCE,
) -> bool:
    """
    Función PURA (sin efectos secundarios) que decide si un modelo candidato
    debe reemplazar al campeón actual.

    Regla: se promueve solo si no hay campeón todavía, o si el candidato
    supera al campeón en `metric` por más de `tolerance`. Esto evita que un
    reentrenamiento con una corrida de mala suerte (varianza del split /
    del optimizador) degrade silenciosamente el modelo en producción.
    """
    if champion is None:
        return True
    champion_score = champion["metrics"][metric]
    candidate_score = candidate_metrics[metric]
    return (candidate_score - champion_score) > tolerance


# --------------------------------------------------------------------------
# Pipeline principal
# --------------------------------------------------------------------------

def run_pipeline(dry_run: bool = False) -> dict:
    print(f"[maintenance] Iniciando pipeline de mantenimiento (dry_run={dry_run})")
    registry = load_registry()
    champion = registry.get("champion")

    # 1-2. Datos + reentrenamiento (reutiliza toda la lógica de train.py)
    csv_path = train_module.download_dataset()
    df = train_module.load_and_engineer_features(csv_path)

    numeric_features = train_module.NUMERIC_FEATURES + [
        "bp_chol_ratio", "max_hr_expected", "hr_reserve_deficit",
    ]
    categorical_features = train_module.CATEGORICAL_FEATURES + ["age_group"]

    X = df[numeric_features + categorical_features]
    y = df[train_module.TARGET_COL].astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_module.TEST_SIZE,
        random_state=train_module.RANDOM_STATE, stratify=y,
    )

    preprocessor = train_module.build_preprocessor(numeric_features, categorical_features)
    best_name, best_search, leaderboard = train_module.train_and_select_best_model(
        X_train, y_train, preprocessor
    )
    candidate_model = best_search.best_estimator_

    # 3. Evaluación
    metrics = train_module.evaluate_model(candidate_model, X_test, y_test)
    print(f"[maintenance] Métricas del candidato ({best_name}): {metrics}")

    # 4. Decisión de promoción
    should_promote = decide_promotion(champion, metrics)
    version = next_version(registry)
    timestamp = datetime.now(timezone.utc).isoformat()

    entry = {
        "version": version,
        "model_name": best_name,
        "metrics": metrics,
        "best_params": best_search.best_params_,
        "trained_at": timestamp,
        "promoted": should_promote,
    }

    if dry_run:
        print(
            f"[maintenance] DRY-RUN: candidato {version} "
            f"{'sería promovido' if should_promote else 'NO sería promovido'}. "
            "No se escribieron artefactos."
        )
        return {"promoted": should_promote, "entry": entry}

    # 5-6. Persistencia + (posible) promoción
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    versioned_path = REGISTRY_DIR / f"model_{version}.joblib"
    joblib.dump(candidate_model, versioned_path)
    entry["artifact_path"] = str(versioned_path.relative_to(PROJECT_DIR))

    registry.setdefault("history", []).append(entry)

    if should_promote:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(versioned_path, PRODUCTION_MODEL_PATH)
        registry["champion"] = entry
        METADATA_PATH.write_text(json.dumps({
            "best_model_name": best_name,
            "best_params": best_search.best_params_,
            "cv_leaderboard": leaderboard,
            "test_metrics": metrics,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target": train_module.TARGET_COL,
            "version": version,
            "promoted_at": timestamp,
        }, indent=2, ensure_ascii=False))
        print(f"[maintenance] Candidato {version} PROMOVIDO a producción "
              f"({PROMOTION_METRIC}={metrics[PROMOTION_METRIC]}).")
    else:
        prev = champion["version"] if champion else "N/A"
        print(
            f"[maintenance] Candidato {version} NO superó al campeón {prev} "
            f"({PROMOTION_METRIC} candidato={metrics[PROMOTION_METRIC]} vs "
            f"campeón={champion['metrics'][PROMOTION_METRIC] if champion else None}). "
            "Se conserva el modelo en producción (rollback seguro)."
        )

    save_registry(registry)
    return {"promoted": should_promote, "entry": entry}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de mantenimiento del modelo")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Ejecuta el reentrenamiento y la evaluación sin escribir artefactos ni promover.",
    )
    args = parser.parse_args()
    result = run_pipeline(dry_run=args.dry_run)

    # Exit 0  → modelo promovido (o dry-run completado sin errores).
    # Exit 1  → candidato no superó al campeón; producción sin cambios.
    # Útil para pipelines externos que quieran reaccionar al resultado.
    raise SystemExit(0 if result["promoted"] else 1)


if __name__ == "__main__":
    main()
