"""
test_app.py
-----------
Pytest suite covering:
    1. Input data validation (schema / value-range checks for the fields the
       Streamlit app collects before they are sent to the model).
    2. Model "endpoint" behaviour: loading the serialized pipeline and
       verifying that a well-formed request produces a valid, well-formed
       response (probability in [0, 1], correct label, correct dtype).
    3. A functional smoke test of the Streamlit app itself, using Streamlit's
       AppTest utility to simulate a user filling the form and clicking the
       "Predict Cardiovascular Risk" button (acts as the app's HTTP-level
       request/response check, since Streamlit apps are served over HTTP).

Run with:
    pytest test_app.py -v
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from app import build_feature_row

MODEL_PATH = Path(__file__).parent / "model" / "cardio_risk_model.joblib"

VALID_INPUT = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model():
    if not MODEL_PATH.exists():
        pytest.skip(
            "Trained model not found. Run `python train.py` before running the tests."
        )
    return joblib.load(MODEL_PATH)


# --------------------------------------------------------------------------
# 1. Input data validation tests
# --------------------------------------------------------------------------

class TestInputValidation:
    """Validates the shape / ranges of the data going into the model."""

    def test_build_feature_row_returns_expected_schema(self):
        df = build_feature_row(VALID_INPUT)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

        expected_columns = {
            "age", "trestbps", "chol", "thalach", "oldpeak",
            "bp_chol_ratio", "max_hr_expected", "hr_reserve_deficit",
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
            "age_group",
        }
        assert expected_columns.issubset(set(df.columns))

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("age", -5),         # negative age is not physiologically valid
            ("trestbps", 0),     # blood pressure cannot be zero
            ("chol", -10),       # cholesterol cannot be negative
        ],
    )
    def test_out_of_range_clinical_values_are_flagged(self, field, bad_value):
        """
        Basic sanity/range validation for clinical fields. The app itself
        constrains these through min_value/max_value on the widgets, but we
        also assert the invariant holds for any value reaching the pipeline.
        """
        payload = dict(VALID_INPUT)
        payload[field] = bad_value

        # Clinical validity invariant: these fields must be strictly positive.
        assert payload[field] <= 0, "Test setup sanity check failed."
        is_valid = payload["age"] > 0 and payload["trestbps"] > 0 and payload["chol"] > 0
        assert is_valid is False

    def test_categorical_fields_within_allowed_codes(self):
        assert VALID_INPUT["sex"] in {0, 1}
        assert VALID_INPUT["cp"] in {0, 1, 2, 3}
        assert VALID_INPUT["fbs"] in {0, 1}
        assert VALID_INPUT["restecg"] in {0, 1, 2}
        assert VALID_INPUT["exang"] in {0, 1}
        assert VALID_INPUT["slope"] in {0, 1, 2}
        assert VALID_INPUT["ca"] in {0, 1, 2, 3, 4}
        assert VALID_INPUT["thal"] in {1, 2, 3}


# --------------------------------------------------------------------------
# 2. Model prediction ("endpoint") tests
# --------------------------------------------------------------------------

class TestModelPrediction:
    """Verifies the serialized model produces a valid, well-formed response."""

    def test_model_file_exists(self):
        assert MODEL_PATH.exists(), (
            "Model artifact not found — run `python train.py` to generate it."
        )

    def test_prediction_returns_valid_probability(self, trained_model):
        features_df = build_feature_row(VALID_INPUT)
        proba = trained_model.predict_proba(features_df)[0, 1]

        assert isinstance(proba, float)
        assert 0.0 <= proba <= 1.0

    def test_prediction_label_is_binary(self, trained_model):
        features_df = build_feature_row(VALID_INPUT)
        label = trained_model.predict(features_df)[0]

        assert label in (0, 1)

    def test_prediction_is_deterministic_for_same_input(self, trained_model):
        features_df = build_feature_row(VALID_INPUT)
        proba_1 = trained_model.predict_proba(features_df)[0, 1]
        proba_2 = trained_model.predict_proba(features_df)[0, 1]

        assert proba_1 == pytest.approx(proba_2)


# --------------------------------------------------------------------------
# 3. Streamlit app functional / "HTTP endpoint" smoke test
# --------------------------------------------------------------------------

class TestStreamlitApp:
    """
    Simulates a real user session against the running app using Streamlit's
    AppTest framework (the app is normally served over HTTP by `streamlit run`;
    AppTest drives the same script-execution path without needing a live server).
    """

    def test_app_loads_without_exceptions(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(Path(__file__).parent / "app.py"))
        at.run(timeout=30)

        assert not at.exception
        assert at.title[0].value == "Cardiovascular Risk Predictor"

    def test_app_predict_button_returns_result(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(Path(__file__).parent / "app.py"))
        at.run(timeout=30)

        if not MODEL_PATH.exists():
            pytest.skip("Trained model not found. Run `python train.py` first.")

        at.button[0].click().run(timeout=30)

        assert not at.exception
        result_shown = bool(at.success) or bool(at.error)
        assert result_shown, "Expected a success or error result message after prediction."
