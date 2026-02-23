# app.py - Sri Lanka Dengue Outbreak Predictor
import streamlit as st
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt

#  Load saved files
model       = pickle.load(open('xgb_model.pkl', 'rb'))
scaler      = pickle.load(open('scaler.pkl', 'rb'))
le_district = pickle.load(open('le_district.pkl', 'rb'))
with open('feature_cols.json') as f:
    feature_cols = json.load(f)

#  Page config 
st.set_page_config(page_title="🦟 Dengue Outbreak Predictor", page_icon="🦟")
st.title("🦟 Sri Lanka Dengue Outbreak Predictor")
st.markdown("Select your district and enter current weather conditions to predict dengue outbreak risk.")

#  Sidebar inputs 
st.sidebar.header("📍 Location")

district = st.sidebar.selectbox(
    "District",
    sorted(list(le_district.classes_))
)

st.sidebar.header("📅 Time")
month_names = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
month_name  = st.sidebar.selectbox("Month", month_names)
month_num   = month_names.index(month_name) + 1

st.sidebar.header("🌤️ Weather Conditions")
temp      = st.sidebar.slider("Avg Temperature (°C)",  20.0, 35.0, 28.0, step=0.1)
precip    = st.sidebar.slider("Avg Precipitation (mm)", 0.0, 500.0, 100.0, step=1.0)
humidity  = st.sidebar.slider("Avg Humidity (%)",       50.0, 100.0, 75.0, step=0.5)
elevation = st.sidebar.number_input("Elevation (m)", min_value=0, max_value=2500, value=100, step=1)

# Predict button 
if st.sidebar.button("🔍 Predict Outbreak Level"):

    district_enc = le_district.transform([district])[0]

    # ['District_encoded', 'Elevation', 'Month',
    #  'Temp_avg', 'Precipitation_avg', 'Humidity_avg']
    input_data   = np.array([[district_enc, elevation,
                               month_num, temp, precip, humidity]])
    input_scaled = scaler.transform(input_data)

    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    #  Results 
    st.markdown("---")
    st.subheader(f"📍 {district} District — {month_name}")

    if prediction == 1:
        st.error(f"🚨 HIGH Outbreak Risk — {probability*100:.1f}% confidence")
        st.markdown("""
        **Recommended Actions:**
        - 🏥 Alert district health authorities immediately
        - 🚿 Increase fumigation campaigns
        - 📢 Launch public awareness programmes
        - 🏠 Conduct house-to-house inspections for stagnant water
        """)
    else:
        st.success(f"✅ LOW Outbreak Risk — {(1-probability)*100:.1f}% confidence")
        st.markdown("""
        **Recommended Actions:**
        - 📋 Continue routine surveillance
        - 📊 Monitor weekly case counts
        - 🌧️ Reassess if rainfall increases significantly
        """)

    #  SHAP explanation 
    st.markdown("---")
    st.subheader("🔬 Why did the model predict this?")
    st.markdown("The chart below shows which factors influenced this prediction the most.")

    explainer   = shap.Explainer(model)
    shap_values = explainer(input_scaled)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    st.caption("🔴 Red bars push toward HIGH outbreak risk | 🔵 Blue bars push toward LOW outbreak risk")

    # Input summary
    st.markdown("---")
    st.subheader("📋 Input Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("District",      district)
        st.metric("Month",         month_name)
        st.metric("Elevation",     f"{elevation} m")
    with col2:
        st.metric("Temperature",   f"{temp} °C")
        st.metric("Precipitation", f"{precip} mm")
        st.metric("Humidity",      f"{humidity} %")

#  Footer 
st.markdown("---")
st.markdown("*Dataset: Sri Lanka Dengue 2019–2021*")

