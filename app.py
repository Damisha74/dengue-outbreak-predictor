# app.py — Sri Lanka Dengue Outbreak Predictor
import streamlit as st
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt

# Load saved files
model   = pickle.load(open('xgb_model.pkl', 'rb'))
scaler  = pickle.load(open('scaler.pkl', 'rb'))
le      = pickle.load(open('label_encoder.pkl', 'rb'))
with open('feature_cols.json') as f:
    feature_cols = json.load(f)

st.set_page_config(page_title="🦟 Dengue Outbreak Predictor", page_icon="🦟")
st.title("🦟 Sri Lanka Dengue Outbreak Predictor")
st.markdown("Predict whether a district will experience a **HIGH or LOW dengue outbreak** based on weather and location.")

st.sidebar.header("📍 Input Parameters")

province_options = list(le.classes_)
province    = st.sidebar.selectbox("Province", province_options)
month       = st.sidebar.slider("Month", 1, 12, 6)
temp        = st.sidebar.slider("Avg Temperature (°C)", 20.0, 35.0, 28.0)
precip      = st.sidebar.slider("Avg Precipitation (mm)", 0.0, 500.0, 100.0)
humidity    = st.sidebar.slider("Avg Humidity (%)", 50.0, 100.0, 75.0)
latitude    = st.sidebar.number_input("Latitude",  5.9, 9.9, 7.5)
longitude   = st.sidebar.number_input("Longitude", 79.6, 81.9, 80.6)
elevation   = st.sidebar.number_input("Elevation (m)", 0, 2500, 100)

if st.sidebar.button("🔍 Predict Outbreak Level"):
    province_enc = le.transform([province])[0]
    input_data   = np.array([[latitude, longitude, elevation, month,
                               temp, precip, humidity, province_enc]])
    input_scaled = scaler.transform(input_data)

    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 HIGH Outbreak Risk — {probability*100:.1f}% confidence")
        st.markdown("**Recommendation:** Authorities should increase fumigation and public awareness in this district.")
    else:
        st.success(f"✅ LOW Outbreak Risk — {(1-probability)*100:.1f}% confidence")
        st.markdown("**Recommendation:** Continue routine monitoring.")

    # SHAP explanation
    st.subheader("🔬 Why did the model predict this?")
    explainer   = shap.Explainer(model)
    shap_values = explainer(input_scaled)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    st.caption("🔴 Red = pushes toward HIGH outbreak | 🔵 Blue = pushes toward LOW outbreak")

st.markdown("---")
st.markdown("*Built for MSc in AI — Machine Learning Assignment | Data: Sri Lanka Dengue 2019–2021*")
