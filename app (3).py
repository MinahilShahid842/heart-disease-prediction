"""Heart Disease Prediction System - Streamlit web app (Logistic Regression)."""
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHEST_PAIN = {1: "1 - Typical Angina", 2: "2 - Atypical Angina",
              3: "3 - Non-anginal Pain", 4: "4 - Asymptomatic"}


def _retrain():
    """Fallback: rebuild the encoders and model from the bundled data file.
    Used only if the saved .pkl files cannot be loaded (e.g. scikit-learn version mismatch)."""
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    raw = pd.read_csv(os.path.join(BASE_DIR, "processed.cleveland.data"),
                      header=None, names=cols, na_values="?")
    enc = {
        "gender": LabelEncoder().fit(["Male", "Female"]),
        "chestpain": LabelEncoder().fit(list(CHEST_PAIN.values())),
        "exang": LabelEncoder().fit(["Yes", "No"]),
        "disease": LabelEncoder().fit(["Yes", "No"]),
    }
    X = pd.DataFrame({
        "Gender": enc["gender"].transform(raw["sex"].map({1: "Male", 0: "Female"})),
        "ChestPain": enc["chestpain"].transform(raw["cp"].astype(int).map(CHEST_PAIN)),
        "MaxHeartRate": raw["thalach"].astype(int),
        "ExerciseAngina": enc["exang"].transform(raw["exang"].map({1: "Yes", 0: "No"})),
    })
    y = (raw["num"] > 0).astype(int)
    model = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)
    return model, enc


@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open(os.path.join(BASE_DIR, "logistic_regression_model.pkl"), "rb"))
        enc = pickle.load(open(os.path.join(BASE_DIR, "label_encoders.pkl"), "rb"))
        return model, enc
    except Exception:
        return _retrain()


def predict(model, enc, gender, chest_pain, max_hr, exang):
    features = pd.DataFrame({
        "Gender": enc["gender"].transform([gender]),
        "ChestPain": enc["chestpain"].transform([chest_pain]),
        "MaxHeartRate": [int(max_hr)],
        "ExerciseAngina": enc["exang"].transform([exang]),
    })
    return int(model.predict(features)[0]), float(model.predict_proba(features)[0][1])


CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 980px;}
.hero {background: linear-gradient(120deg, #0f766e 0%, #134e4a 100%);
       border-radius: 18px; padding: 28px 32px; color: #f0fdfa; margin-bottom: 22px;}
.hero h1 {margin: 0 0 6px 0; font-size: 2rem; color: #f0fdfa;}
.hero p {margin: 0; opacity: .85; font-size: 1rem;}
.card {border-radius: 16px; padding: 22px 24px; border: 1px solid rgba(128,128,128,.25);}
.result-high {background: #fef2f2; border-color: #fca5a5; color: #7f1d1d;}
.result-low  {background: #f0fdf4; border-color: #86efac; color: #14532d;}
.result-card h2 {margin: 0 0 4px 0; font-size: 1.5rem;}
.meter {height: 14px; border-radius: 999px; background: rgba(0,0,0,.12); overflow: hidden; margin: 14px 0 6px 0;}
.meter > div {height: 100%; border-radius: 999px;}
.small {font-size: .85rem; opacity: .8;}
.placeholder {text-align: center; padding: 48px 12px; opacity: .65;}
</style>
"""


def risk_card(label, prob):
    high = label == 1
    css = "result-high" if high else "result-low"
    color = "#dc2626" if high else "#16a34a"
    title = "Higher risk detected" if high else "Lower risk detected"
    note = ("The inputs resemble patients in the dataset who had heart disease."
            if high else
            "The inputs resemble patients in the dataset who did not have heart disease.")
    return f"""
    <div class="card result-card {css}">
      <h2>{title}</h2>
      <div class="small">{note}</div>
      <div class="meter"><div style="width:{prob*100:.0f}%; background:{color};"></div></div>
      <div><b>{prob:.0%}</b> estimated probability of heart disease</div>
    </div>"""


def main():
    st.set_page_config(page_title="CardioCheck", page_icon="🫀", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <h1>🫀 CardioCheck</h1>
      <p>A quick heart-health screening demo powered by Logistic Regression
         and the UCI Cleveland Heart Disease dataset.</p>
    </div>""", unsafe_allow_html=True)

    model, enc = load_artifacts()

    with st.sidebar:
        st.header("About")
        st.write("CardioCheck estimates the chance of heart disease from four patient details.")
        st.markdown("**Model:** Logistic Regression  \n"
                    "**Data:** UCI Heart Disease (Cleveland, 303 patients)  \n"
                    "**Inputs:** gender, chest pain type, max heart rate, exercise angina")
        st.caption("Student project - not a medical device.")

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.subheader("Patient details")
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        chest_pain = st.selectbox("Chest pain type", list(CHEST_PAIN.values()), index=3)
        max_hr = st.slider("Maximum heart rate achieved (bpm)", 60, 220, 150)
        exang = st.radio("Chest pain during exercise?", ["No", "Yes"], horizontal=True)
        go = st.button("Run screening", type="primary", use_container_width=True)

    with right:
        st.subheader("Result")
        if go:
            label, prob = predict(model, enc, gender, chest_pain, max_hr, exang)
            st.markdown(risk_card(label, prob), unsafe_allow_html=True)
        else:
            st.markdown('<div class="card placeholder">Fill in the details and press '
                        '<b>Run screening</b> to see the result.</div>', unsafe_allow_html=True)

    st.caption("This tool is for learning purposes only and must not be used for real medical decisions.")


main()
