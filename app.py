"""
app.py
------
Interactive Streamlit application for cardiovascular risk prediction.

All user-facing elements (labels, buttons, results, messages) are in English,
regardless of the internal code comments language used elsewhere in the project.

Run locally with:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "cardio_risk_model.joblib"

st.set_page_config(
    page_title="Cardiovascular Risk Predictor",
    page_icon="\u2764\ufe0f",
    layout="centered",
)


# --------------------------------------------------------------------------
# Model loading (cached so it is loaded only once per session)
# --------------------------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Replicates the exact feature engineering performed in train.py."""
    age = inputs["age"]
    trestbps = inputs["trestbps"]
    chol = inputs["chol"]
    thalach = inputs["thalach"]

    bp_chol_ratio = trestbps / chol
    max_hr_expected = 220 - age
    hr_reserve_deficit = max_hr_expected - thalach

    if age < 40:
        age_group = "under_40"
    elif age < 50:
        age_group = "40_49"
    elif age < 60:
        age_group = "50_59"
    else:
        age_group = "60_plus"

    row = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": inputs["oldpeak"],
        "bp_chol_ratio": bp_chol_ratio,
        "max_hr_expected": max_hr_expected,
        "hr_reserve_deficit": hr_reserve_deficit,
        "sex": inputs["sex"],
        "cp": inputs["cp"],
        "fbs": inputs["fbs"],
        "restecg": inputs["restecg"],
        "exang": inputs["exang"],
        "slope": inputs["slope"],
        "ca": inputs["ca"],
        "thal": inputs["thal"],
        "age_group": age_group,
    }
    return pd.DataFrame([row])


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("Cardiovascular Risk Predictor v1.1.1")
st.write(
    "Enter the patient's clinical data below to estimate the probability of "
    "cardiovascular disease. This tool is for educational purposes only and "
    "is **not** a substitute for professional medical advice."
)

model = load_model()

if model is None:
    st.error(
        "No trained model was found. Please run `python train.py` first to "
        "generate the model file before launching this app."
    )
    st.stop()

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    cp = st.selectbox(
        "Chest pain type",
        options=[
            ("Typical angina", 0),
            ("Atypical angina", 1),
            ("Non-anginal pain", 2),
            ("Asymptomatic", 3),
        ],
        format_func=lambda x: x[0],
    )
    trestbps = st.number_input(
        "Resting blood pressure (mm Hg)", min_value=80, max_value=220, value=130
    )
    chol = st.number_input(
        "Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=240
    )
    fbs = st.selectbox(
        "Fasting blood sugar > 120 mg/dl",
        options=[("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0],
    )
    restecg = st.selectbox(
        "Resting ECG results",
        options=[
            ("Normal", 0),
            ("ST-T wave abnormality", 1),
            ("Left ventricular hypertrophy", 2),
        ],
        format_func=lambda x: x[0],
    )

with col2:
    thalach = st.number_input(
        "Maximum heart rate achieved", min_value=60, max_value=220, value=150
    )
    exang = st.selectbox(
        "Exercise-induced angina",
        options=[("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0],
    )
    oldpeak = st.number_input(
        "ST depression induced by exercise (oldpeak)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f",
    )
    slope = st.selectbox(
        "Slope of the peak exercise ST segment",
        options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
        format_func=lambda x: x[0],
    )
    ca = st.selectbox(
        "Number of major vessels colored by fluoroscopy",
        options=[(str(i), i) for i in range(0, 5)],
        format_func=lambda x: x[0],
    )
    thal = st.selectbox(
        "Thalassemia test result",
        options=[("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3)],
        format_func=lambda x: x[0],
    )

predict_clicked = st.button("Predict Cardiovascular Risk", type="primary")

if predict_clicked:
    inputs = {
        "age": age,
        "sex": sex[1],
        "cp": cp[1],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs[1],
        "restecg": restecg[1],
        "thalach": thalach,
        "exang": exang[1],
        "oldpeak": oldpeak,
        "slope": slope[1],
        "ca": ca[1],
        "thal": thal[1],
    }

    features_df = build_feature_row(inputs)
    probability = float(model.predict_proba(features_df)[0, 1])
    prediction = int(probability >= 0.5)

    st.header("Result")

    if prediction == 1:
        st.error(f"High risk of cardiovascular disease — estimated probability: {probability:.1%}")
    else:
        st.success(f"Low risk of cardiovascular disease — estimated probability: {probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "This estimate is based on a statistical model trained on historical "
        "clinical data and should not be used as a medical diagnosis. "
        "Please consult a qualified healthcare professional."
    )

st.divider()
st.caption("Model: Logistic Regression pipeline trained on the UCI Heart Disease dataset.")
