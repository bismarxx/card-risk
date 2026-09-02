"""
test_maintenance_ci.py
-----------------------
Pruebas de funcionamiento del MANTENIMIENTO y la INTEGRACIÓN CONTINUA,
tal como lo exige la rúbrica del proyecto (Unidad II: "Pruebas de
funcionamiento del mantenimiento e integración continua, 3 casos
considerados").

Caso 1 — Mantenimiento: un modelo candidato con mejores métricas SÍ debe
         promoverse a campeón.
Caso 2 — Mantenimiento: un modelo candidato con métricas iguales o peores
         NO debe promoverse (se preserva el modelo en producción; rollback
         seguro ante una corrida de mala calidad).
Caso 3 — Integración continua: los workflows de GitHub Actions (CI y
         mantenimiento) existen, son YAML válidos y están correctamente
         configurados (triggers y pasos mínimos esperados).

Estas pruebas son independientes de test_app.py (que cubre el
funcionamiento de la aplicación / modelo de predicción) y se ejecutan tanto
localmente como dentro del propio workflow de CI (ver .github/workflows/ci.yml).

Run with:
    pytest test_maintenance_ci.py -v
"""

from pathlib import Path

import pytest
import yaml

from retrain_pipeline import decide_promotion, PROMOTION_METRIC, PROMOTION_TOLERANCE

PROJECT_DIR = Path(__file__).parent
WORKFLOWS_DIR = PROJECT_DIR / ".github" / "workflows"

CHAMPION = {
    "version": "v3",
    "metrics": {"roc_auc": 0.8900, "accuracy": 0.84, "f1_score": 0.85},
}


# --------------------------------------------------------------------------
# Caso 1 — Mantenimiento: promoción cuando el candidato mejora al campeón
# --------------------------------------------------------------------------

class TestMaintenanceCase1PromotionWhenBetter:
    def test_candidate_with_better_roc_auc_is_promoted(self):
        candidate_metrics = {"roc_auc": 0.9200, "accuracy": 0.88, "f1_score": 0.89}
        assert decide_promotion(CHAMPION, candidate_metrics) is True

    def test_no_champion_yet_always_promotes_first_model(self):
        """El primer modelo entrenado (sin campeón previo) siempre se promueve."""
        candidate_metrics = {"roc_auc": 0.5, "accuracy": 0.5, "f1_score": 0.5}
        assert decide_promotion(None, candidate_metrics) is True


# --------------------------------------------------------------------------
# Caso 2 — Mantenimiento: NO promoción cuando el candidato no mejora
# --------------------------------------------------------------------------

class TestMaintenanceCase2NoPromotionWhenNotBetter:
    def test_candidate_with_worse_roc_auc_is_not_promoted(self):
        candidate_metrics = {"roc_auc": 0.8500, "accuracy": 0.80, "f1_score": 0.81}
        assert decide_promotion(CHAMPION, candidate_metrics) is False

    def test_candidate_with_equal_roc_auc_is_not_promoted(self):
        """Empates no cuentan como mejora: evita 'promociones' por ruido numérico."""
        candidate_metrics = {"roc_auc": CHAMPION["metrics"]["roc_auc"], "accuracy": 0.84, "f1_score": 0.85}
        assert decide_promotion(CHAMPION, candidate_metrics) is False

    def test_candidate_within_tolerance_is_not_promoted(self):
        """Una mejora por debajo del margen de tolerancia tampoco promueve."""
        tiny_improvement = CHAMPION["metrics"][PROMOTION_METRIC] + (PROMOTION_TOLERANCE / 2)
        candidate_metrics = {"roc_auc": tiny_improvement, "accuracy": 0.84, "f1_score": 0.85}
        assert decide_promotion(CHAMPION, candidate_metrics) is False


# --------------------------------------------------------------------------
# Caso 3 — Integración continua: workflows válidos y correctamente configurados
# --------------------------------------------------------------------------

class TestCICase3WorkflowsAreValid:
    def test_ci_workflow_exists_and_is_valid_yaml(self):
        ci_path = WORKFLOWS_DIR / "ci.yml"
        assert ci_path.exists(), "Falta el workflow de CI (.github/workflows/ci.yml)"

        workflow = yaml.safe_load(ci_path.read_text())
        assert workflow is not None

    def test_ci_workflow_triggers_on_push_and_pull_request(self):
        workflow = yaml.safe_load((WORKFLOWS_DIR / "ci.yml").read_text())
        # YAML interpreta la clave 'on' como booleano True; se maneja ambos casos.
        triggers = workflow.get("on", workflow.get(True))
        assert "push" in triggers
        assert "pull_request" in triggers

    def test_ci_workflow_runs_the_test_suite(self):
        ci_path = WORKFLOWS_DIR / "ci.yml"
        content = ci_path.read_text()
        assert "pytest test_app.py" in content
        assert "pytest test_maintenance_ci.py" in content

    def test_maintenance_workflow_exists_and_is_scheduled(self):
        maintenance_path = WORKFLOWS_DIR / "maintenance.yml"
        assert maintenance_path.exists(), (
            "Falta el workflow de mantenimiento (.github/workflows/maintenance.yml)"
        )

        workflow = yaml.safe_load(maintenance_path.read_text())
        triggers = workflow.get("on", workflow.get(True))
        assert "schedule" in triggers, "El workflow de mantenimiento debe correr en un cron programado"
        assert "workflow_dispatch" in triggers, "Debe poder dispararse manualmente también"

    def test_maintenance_workflow_runs_retrain_pipeline(self):
        content = (WORKFLOWS_DIR / "maintenance.yml").read_text()
        assert "retrain_pipeline.py" in content
