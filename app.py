import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load Model
# -------------------------------
model_path = os.path.join(os.path.dirname(__file__), "aqi_model.pkl")
model = joblib.load(model_path)

# -------------------------------
# Header Section
# -------------------------------
st.markdown("""
# 🌍 Air Quality Prediction App  
This tool predicts **NO₂ concentration** based on environmental sensor inputs.  
Adjust the sliders below to explore predictions interactively.
""")

# -------------------------------
# Input Section (Two Columns + Sliders)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    co = st.slider("CO (mg/m³)", 0.0, 50.0, 2.0)
    temp = st.slider("Temperature (°C)", -20.0, 50.0, 20.0)
    ah = st.slider("Absolute Humidity", 0.0, 5.0, 1.0)
    so2 = st.slider("SO₂ (µg/m³)", 0.0, 500.0, 20.0)

with col2:
    o3 = st.slider("O₃ Sensor (PT08.S5)", 0.0, 5000.0, 1000.0)
    rh = st.slider("Relative Humidity (%)", 0, 100, 50)
    no = st.slider("NO (µg/m³)", 0.0, 500.0, 15.0)
    benzene = st.slider("Benzene (C6H6 µg/m³)", 0.0, 50.0, 5.0)

# -------------------------------
# Pollutant Selector
# -------------------------------
option = st.selectbox(
    "Choose pollutant to predict",
    ("NO₂", "CO", "O₃")
)

# -------------------------------
# AQI Interpretation Function
# -------------------------------
def interpret_no2(value):
    if value <= 40:
        return "🟢 Good"
    elif value <= 80:
        return "🟡 Moderate"
    elif value <= 180:
        return "🟠 Unhealthy"
    else:
        return "🔴 Hazardous"

# -------------------------------
# Live Prediction
# -------------------------------
features = np.array([[co, o3, temp, rh, ah]])
prediction = model.predict(features)[0]

st.success(f"✅ Predicted {option}: **{prediction:.2f} µg/m³**")
st.info(f"AQI Category: {interpret_no2(prediction)}")

# 📊 Bar chart
fig, ax = plt.subplots()
ax.bar([f"Predicted {option}"], [prediction], color="skyblue")
ax.set_ylabel("µg/m³")
ax.set_title(f"Predicted {option} Concentration")
st.pyplot(fig)

# -------------------------------
# Prediction History
# -------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("Save Prediction"):
    st.session_state["history"].append(prediction)

if st.session_state["history"]:
    st.markdown("### 📈 Prediction History")
    st.line_chart(st.session_state["history"])

# -------------------------------
# Comparison Mode
# -------------------------------
st.markdown("### 🔄 Compare Two Scenarios")

colA, colB = st.columns(2)

with colA:
    co1 = st.slider("Scenario 1 CO", 0.0, 50.0, 2.0, key="co1")
    temp1 = st.slider("Scenario 1 Temp", -20.0, 50.0, 20.0, key="temp1")

with colB:
    co2 = st.slider("Scenario 2 CO", 0.0, 50.0, 5.0, key="co2")
    temp2 = st.slider("Scenario 2 Temp", -20.0, 50.0, 25.0, key="temp2")

features1 = np.array([[co1, o3, temp1, rh, ah]])
features2 = np.array([[co2, o3, temp2, rh, ah]])

pred1 = model.predict(features1)[0]
pred2 = model.predict(features2)[0]

st.write(f"Scenario 1 Prediction: {pred1:.2f} µg/m³")
st.write(f"Scenario 2 Prediction: {pred2:.2f} µg/m³")

# 📥 Download button
result_df = pd.DataFrame({
    "CO": [co], "O₃ Sensor": [o3], "Temperature": [temp],
    "RH": [rh], "AH": [ah], "SO₂": [so2], "NO": [no],
    "Benzene": [benzene], f"Predicted {option}": [prediction]
})
st.download_button(
    label="📥 Download Prediction as CSV",
    data=result_df.to_csv(index=False),
    file_name="prediction.csv",
    mime="text/csv"
)

# -------------------------------
# Sidebar Info
# -------------------------------
with st.sidebar:
    st.markdown("### 📁 Project Info")
    st.markdown("- Dataset: UCI Air Quality")
    st.markdown("- Model: Random Forest Regressor")
    st.markdown("- Target: NO₂ Concentration")
    st.markdown("- Author: Prakrati")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | © 2025 Prakrati")
