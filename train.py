"""
train.py
--------
Entrena un modelo de clasificación para predecir el riesgo de enfermedad
cardiovascular a partir de datos clínicos.

Flujo:
    1. Descarga el dataset público (UCI Heart Disease - Cleveland, 303 pacientes).
    2. Limpieza y extracción/ingeniería de características.
    3. Búsqueda de hiperparámetros (GridSearchCV) para varios modelos candidatos,
       optimizando ROC-AUC mediante validación cruzada estratificada.
    4. Evaluación final sobre un conjunto de prueba independiente.
    5. Serialización del mejor pipeline (preprocesamiento + modelo) con joblib,
       junto con un archivo de metadatos (métricas, features, mejores hiperparámetros).

Uso:
    python train.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/"
    "master/heart.csv"
)
DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "heart.csv"

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "cardio_risk_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columnas numéricas y categóricas originales del dataset UCI Heart Disease.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]
TARGET_COL = "target"


# --------------------------------------------------------------------------
# 1. Descarga del dataset
# --------------------------------------------------------------------------

def download_dataset(url: str = DATA_URL, dest: Path = DATA_PATH) -> Path:
    """Descarga el dataset público si aún no existe localmente."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[download] Dataset ya existe en {dest}, se omite la descarga.")
        return dest

    print(f"[download] Descargando dataset desde {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[download] Dataset guardado en {dest}")
    except Exception as exc:  # pragma: no cover - depende de la red
        raise RuntimeError(
            f"No se pudo descargar el dataset desde {url}. "
            "Verifica la conexión a internet o coloca manualmente el archivo "
            f"'heart.csv' en {dest}."
        ) from exc
    return dest


# --------------------------------------------------------------------------
# 2. Extracción / ingeniería de características
# --------------------------------------------------------------------------

def load_and_engineer_features(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV crudo y agrega variables derivadas (feature engineering)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # El dataset original no trae nulos, pero validamos y limpiamos de todos modos.
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    # --- Ingeniería de características ---
    # Presión de pulso aproximada: relación entre presión sistólica y colesterol.
    df["bp_chol_ratio"] = df["trestbps"] / df["chol"]

    # Categoría de riesgo por edad (feature clínicamente relevante).
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 50, 60, 120],
        labels=["under_40", "40_49", "50_59", "60_plus"],
    ).astype(str)

    # Frecuencia cardíaca máxima esperada (fórmula 220 - edad) y su déficit.
    df["max_hr_expected"] = 220 - df["age"]
    df["hr_reserve_deficit"] = df["max_hr_expected"] - df["thalach"]

    return df


# --------------------------------------------------------------------------
# 3. Pipeline de preprocesamiento
# --------------------------------------------------------------------------

def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Construye el ColumnTransformer: escalado numérico + one-hot categórico."""
    from sklearn.preprocessing import OneHotEncoder

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# --------------------------------------------------------------------------
# 4. Optimización de hiperparámetros
# --------------------------------------------------------------------------

def get_candidate_models() -> dict:
    """Define los modelos candidatos y sus grillas de hiperparámetros."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "param_grid": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__solver": ["lbfgs"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        },
    }


def train_and_select_best_model(X_train, y_train, preprocessor):
    """
    Ejecuta GridSearchCV (optimizando ROC-AUC vía validación cruzada
    estratificada de 5 folds) para cada modelo candidato y retorna el mejor.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = get_candidate_models()

    best_score = -np.inf
    best_name = None
    best_search: GridSearchCV | None = None
    leaderboard = {}

    for name, spec in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", spec["estimator"])]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["param_grid"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        t0 = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        leaderboard[name] = {
            "best_cv_roc_auc": round(search.best_score_, 4),
            "best_params": search.best_params_,
            "seconds": round(elapsed, 1),
        }
        print(
            f"[tuning] {name}: mejor ROC-AUC (CV) = {search.best_score_:.4f} "
            f"| params = {search.best_params_} | {elapsed:.1f}s"
        )

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = name
            best_search = search

    return best_name, best_search, leaderboard


# --------------------------------------------------------------------------
# 5. Evaluación
# --------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    csv_path = download_dataset()
    df = load_and_engineer_features(csv_path)

    numeric_features = NUMERIC_FEATURES + [
        "bp_chol_ratio", "max_hr_expected", "hr_reserve_deficit",
    ]
    categorical_features = CATEGORICAL_FEATURES + ["age_group"]

    X = df[numeric_features + categorical_features]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    print("\n=== Optimización de hiperparámetros (GridSearchCV, scoring=roc_auc) ===")
    best_name, best_search, leaderboard = train_and_select_best_model(
        X_train, y_train, preprocessor
    )
    best_model = best_search.best_estimator_
    print(f"\n[selección] Mejor modelo: {best_name}")

    print("\n=== Evaluación en conjunto de prueba (hold-out) ===")
    metrics = evaluate_model(best_model, X_test, y_test)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[guardado] Modelo serializado en {MODEL_PATH}")

    metadata = {
        "best_model_name": best_name,
        "best_params": best_search.best_params_,
        "cv_leaderboard": leaderboard,
        "test_metrics": metrics,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET_COL,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "sklearn_version": __import__("sklearn").__version__,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[guardado] Metadatos en {METADATA_PATH}")


if __name__ == "__main__":
    main()
