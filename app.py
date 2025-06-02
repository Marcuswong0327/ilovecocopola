import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Resignation Prediction", layout="centered")
st.title("💼 Employee Resignation Prediction App")

st.header("🔍 Input Employee Information")

# User input features
ge = st.slider("Age", 18, 65, 30)
work_hours = st.slider("Work Hours per Week", 0, 80, 40)
projects_handled = st.slider("Projects Handled", 0, 50, 5)
training_hours = st.slider("Training Hours", 0, 100, 20)
satisfaction = st.slider("Employee Satisfaction Score", 0.0, 1.0, 0.5)


# Prediction button
if st.button("Predict Resignation"):
    features = np.array([[age, work_hours, projects_handled,
                          training_hours, satisfaction]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]


    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    st.success("✅ Likely to Stay" if prediction == 0 else "⚠️ Likely to Resign")
    if prob >= 0.7:
        st.error("🔴 High Risk: Likely to Resign")
    elif prob >= 0.4:
        st.warning("🟠 Medium Risk: Uncertain")
    else:
        st.success("✅ Low Risk: Likely to Stay")

    # st.metric("Resignation Probability", f"{prob:.2%}")
    st.metric(label="Resignation Probability", value=f"{prob:.2%}")
