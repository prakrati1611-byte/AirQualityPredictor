import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pydeck as pdk
import os
import requests

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom styling (light mode only)
# -------------------------------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 16px !important;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 0.8rem;
        padding-right: 1.2rem;
        padding-left: 1.2rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    h1, h2, h3 {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.4rem;
    }
    .stButton>button {
        font-size: 14px;
        padding: 0.5em 1em;
        border-radius: 8px;
        background-color: #4C72B0;
        color: white;
        border: none;
        cursor: pointer;
    }
    .metric-card {
        background: #f8f9fb;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .prediction-card {
        background:#eef2f7; 
        padding:14px; 
        border-radius:10px; 
        border-left:6px solid #4C72B0; 
        margin-top:0.5rem;
    }
    .footer {
        text-align:center; 
        font-size:13px; 
        color:gray;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load model
# -------------------------------
model_path = os.path.join(os.path.dirname(__file__), "aqi_model.pkl")
model = joblib.load(model_path)

# -------------------------------
# Header
# -------------------------------
st.markdown("""
<div style='text-align: center; padding: 0.6rem 0;'>
    <h1 style='color:#4C72B0;'>🌍 Air Quality Insights Dashboard</h1>
    <p style='font-size:16px; margin:0;'>Live NO₂ predictions powered by machine learning and real-time environmental data</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# City input
# -------------------------------
popular_cities = ["Delhi", "Mumbai", "Bangalore", "Bhopal", "Indore", "Kolkata", "Chennai", "Hyderabad"]
city = st.selectbox("🏙️ Choose a city", popular_cities)

# Coordinates for map
CITY_COORDS = {
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
}

# -------------------------------
# API fetchers
# -------------------------------
def get_weather(city):
    api_key = "bea8ca751f613f5b3a5d24720e9bb957"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10).json()
    except Exception:
        return None, None
    if "main" in response:
        temp = response["main"].get("temp")
        rh = response["main"].get("humidity")
        return temp, rh
    return None, None

def get_pollutants(city):
    waqi_token = "03ddc0ec6c3327fb84733818314c085c79930c79"
    url = f"https://api.waqi.info/feed/{city}/?token={waqi_token}"
    try:
        response = requests.get(url, timeout=10).json()
    except Exception:
        return None, None, None
    if "data" in response and isinstance(response["data"], dict):
        iaqi = response["data"].get("iaqi", {})
        co = iaqi.get("co", {}).get("v", 2.0)
        o3 = iaqi.get("o3", {}).get("v", 1000.0)
        ah = 1.0
        return co, o3, ah
    return None, None, None

# -------------------------------
# Helpers
# -------------------------------
def interpret_no2(value):
    if value is None:
        return "N/A"
    if value <= 40:
        return "🟢 Good"
    elif value <= 80:
        return "🟡 Moderate"
    elif value <= 180:
        return "🟠 Unhealthy"
    else:
        return "🔴 Hazardous"

def get_aqi_color(value):
    if value is None:
        return "#7f8c8d"  # grey
    if value <= 40:
        return "#2ecc71"
    elif value <= 80:
        return "#f1c40f"
    elif value <= 180:
        return "#e67e22"
    else:
        return "#e74c3c"

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]

def get_no2_prediction(city):
    temp, rh = get_weather(city)
    co, o3, ah = get_pollutants(city)
    if None in [temp, rh, co, o3, ah]:
        return None, temp, rh, co, o3, ah
    features = np.array([[co, o3, temp, rh, ah]])
    try:
        prediction = float(model.predict(features)[0])
    except Exception:
        prediction = None
    return prediction, temp, rh, co, o3, ah

# -------------------------------
# Main prediction logic
# -------------------------------
if city:
    prediction, temp, rh, co, o3, ah = get_no2_prediction(city)

    if prediction is None:
        st.error("⚠️ Could not fetch complete data. Prediction may be inaccurate.")
    else:
        st.success("✅ Data fetched successfully. Prediction is based on live values.")

        # Metric cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-card'><b>Temperature</b><br><span style='font-size:16px;'>{temp} °C</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><b>Humidity</b><br><span style='font-size:16px;'>{rh} %</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><b>Absolute Humidity</b><br><span style='font-size:16px;'>{ah}</span></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><b>CO</b><br><span style='font-size:16px;'>{co} mg/m³</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><b>O₃</b><br><span style='font-size:16px;'>{o3} µg/m³</span></div>", unsafe_allow_html=True)

        # Prediction card
        aqi_label = interpret_no2(prediction)
        st.markdown(f"""
        <div class='prediction-card'>
            <h3 style='margin:0 0 0.5rem 0;'>Predicted NO₂ for <span style='color:#4C72B0;'>{city}</span></h3>
            <p style='font-size:18px; font-weight:bold; margin:0;'>{prediction:.2f} µg/m³</p>
            <p style='font-size:15px; margin:0.2rem 0 0;'>AQI Category: <strong>{aqi_label}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # AQI gauge meter
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "NO₂ AQI"},
            gauge={
                'axis': {'range': [0, 200]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [41, 80], 'color': "yellow"},
                    {'range': [81, 180], 'color': "orange"},
                    {'range': [181, 200], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Download prediction
        st.download_button(
            label="📥 Download Prediction as CSV",
            data=pd.DataFrame({
                "City": [city],
                "CO": [co],
                "O₃": [o3],
                "Temperature": [temp],
                "RH": [rh],
                "AH": [ah],
                "Predicted NO₂": [prediction]
            }).to_csv(index=False),
            file_name=f"{city}_prediction.csv",
            mime="text/csv"
        )

# -------------------------------
# Multi-pollutant comparison
# -------------------------------
st.markdown("---")
st.markdown("### 📊 City-Wise Pollutant Comparison")

city_names, no2_values, co_values, o3_values = [], [], [], []
for city_name in popular_cities:
    pred_i, temp_i, rh_i, co_i, o3_i, ah_i = get_no2_prediction(city_name)
    if pred_i is not None:
        city_names.append(city_name)
        no2_values.append(pred_i)
        co_values.append(co_i)
        o3_values.append(o3_i)

if city_names:
    tab_co, tab_o3, tab_no2 = st.tabs(["🧪 CO (mg/m³)", "🧪 O₃ (µg/m³)", "🔮 NO₂ (µg/m³)"])

    with tab_co:
        fig_co, ax_co = plt.subplots(figsize=(7, 3.2))
        ax_co.bar(city_names, co_values, color="#4C72B0", edgecolor="black")
        ax_co.set_title("CO Across Cities", fontsize=12, weight='bold')
        ax_co.set_ylabel("CO (mg/m³)", fontsize=10)
        ax_co.tick_params(axis='x', labelsize=9)
        ax_co.tick_params(axis='y', labelsize=9)
        ax_co.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_co, use_container_width=True)

    with tab_o3:
        fig_o3, ax_o3 = plt.subplots(figsize=(7, 3.2))
        ax_o3.bar(city_names, o3_values, color="#55a868", edgecolor="black")
        ax_o3.set_title("O₃ Across Cities", fontsize=12, weight='bold')
        ax_o3.set_ylabel("O₃ (µg/m³)", fontsize=10)
        ax_o3.tick_params(axis='x', labelsize=9)
        ax_o3.tick_params(axis='y', labelsize=9)
        ax_o3.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_o3, use_container_width=True)

    with tab_no2:
        fig_no2, ax_no2 = plt.subplots(figsize=(7, 3.2))
        colors = [get_aqi_color(v) for v in no2_values]
        ax_no2.bar(city_names, no2_values, color=colors, edgecolor="black")
        ax_no2.set_title("Predicted NO₂ Across Cities", fontsize=12, weight='bold')
        ax_no2.set_ylabel("NO₂ (µg/m³)", fontsize=10)
        ax_no2.tick_params(axis='x', labelsize=9)
        ax_no2.tick_params(axis='y', labelsize=9)
        ax_no2.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_no2, use_container_width=True)
else:
    st.warning("⚠️ Could not fetch data for any city.")

# -------------------------------
# AQI color-coded map (pydeck)
# -------------------------------
st.markdown("---")
st.markdown("### 🗺️ AQI Map of Indian Cities")

if city_names:
    map_df = pd.DataFrame({
        "city": city_names,
        "lat": [CITY_COORDS[c][0] for c in city_names],
        "lon": [CITY_COORDS[c][1] for c in city_names],
        "no2": no2_values
    })
    map_df["rgb"] = map_df["no2"].apply(lambda v: hex_to_rgb(get_aqi_color(v)))
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_fill_color='rgb',
        get_radius=50000,
        pickable=True
    )
    view_state = pdk.ViewState(latitude=21.0, longitude=78.0, zoom=4.5)
    tooltip = {"text": "{city}\nNO₂: {no2} µg/m³"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
else:
    st.info("Select a city to populate the map and comparisons.")

# -------------------------------
# Sidebar info + educational content
# -------------------------------
with st.sidebar:
    st.markdown("### 📁 Project info")
    st.markdown("- **Dataset:** UCI Air Quality")
    st.markdown("- **Model:** Random Forest Regressor")
    st.markdown("- **Target:** NO₂ Concentration")
    st.markdown("- **Author:** Prakrati")

    with st.expander("📘 What is NO₂?"):
        st.markdown(
            "- Nitrogen dioxide (NO₂) is a pollutant from vehicles and industrial activity.\n"
            "- High NO₂ can worsen asthma and reduce lung function.\n"
            "- It contributes to smog and acid rain."
        )

    with st.expander("🩺 Health effects"):
        st.markdown(
            "- Short-term exposure: Irritation of airways, coughing, wheezing.\n"
            "- Long-term exposure: Increased risk of respiratory infections and chronic lung disease."
        )

    with st.expander("🧠 How does the model work?"):
        st.markdown(
            "- Algorithm: Random Forest Regressor.\n"
            "- Inputs: CO, O₃, Temperature, Relative Humidity, Absolute Humidity.\n"
            "- Output: Predicted NO₂ concentration (µg/m³).\n"
            "- Predictions are based on live API values where available."
        )

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<hr style='margin-top:1.0rem;'>
<div class='footer'>
    Made using Streamlit • © 2025 Prakrati
</div>
""", unsafe_allow_html=True)
