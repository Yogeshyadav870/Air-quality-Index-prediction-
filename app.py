import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load saved objects
# -----------------------------
category_model = joblib.load("aqi_category_model.pkl")
label_encoder = joblib.load("aqi_category_labelencoder.pkl")
features = joblib.load("aqi_category_features.pkl")

# Optional numeric AQI model (if file exists)
try:
    regression_model = joblib.load("aqi_regression_model.pkl")
    regression_features = joblib.load("aqi_regression_features.pkl")
    regression_available = True
except:
    regression_available = False

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AQI Prediction App", layout="centered")
st.title("🌍 Air Quality Index (AQI) Prediction")

st.write(
    "Enter **key pollutants**. Other pollutants are assumed at average levels "
    "to provide fast and user-friendly predictions."
)

# -----------------------------
# User Inputs (ONLY 3)
# -----------------------------
pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=50.0)
pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=80.0)
no2  = st.number_input("NO2 (µg/m³)",  min_value=0.0, value=40.0)

# -----------------------------
# Auto-filled pollutant values
# -----------------------------
default_values = {
    "PM2.5": pm25,
    "PM10": pm10,
    "NO2": no2,
    "OZONE": 30.0,
    "SO2": 10.0,
    "CO": 0.5,
    "NH3": 5.0
}

# Build input dataframe (correct order)
input_data = {f: default_values.get(f, 0) for f in features}
input_df = pd.DataFrame([input_data])

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict AQI"):
    # -------- Category Prediction --------
    pred = category_model.predict(input_df)
    category = label_encoder.inverse_transform(pred)[0]

    # AQI color mapping
    color_map = {
        "Good": "#2ecc71",
        "Satisfactory": "#7bed9f",
        "Moderate": "#f1c40f",
        "Poor": "#e67e22",
        "Very Poor": "#e74c3c",
        "Severe": "#8e44ad"
    }

    color = color_map.get(category, "#ffffff")

    # Display colored AQI box
    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:20px;
            border-radius:12px;
            text-align:center;
            color:black;
            font-size:24px;
            font-weight:bold;">
            AQI Category: {category}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------- Health Advisory --------
    advisory_map = {
        "Good": "Air quality is good. Ideal for outdoor activities.",
        "Satisfactory": "Air quality is acceptable. Minor discomfort to sensitive people.",
        "Moderate": "May cause breathing discomfort to people with lung or heart disease.",
        "Poor": "Breathing discomfort to most people on prolonged exposure.",
        "Very Poor": "Respiratory illness on prolonged exposure. Avoid outdoor activity.",
        "Severe": "Serious health impacts. Stay indoors and avoid exertion."
    }

    st.info(f"🩺 Health Advisory: {advisory_map.get(category, '')}")

    # -------- Numeric AQI Toggle --------
    if regression_available:
        show_numeric = st.checkbox("Show predicted numeric AQI value")

        if show_numeric:
            reg_input = input_df[regression_features]
            numeric_aqi = regression_model.predict(reg_input)[0]
            st.metric("Predicted AQI Value", round(numeric_aqi, 2))
