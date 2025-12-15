import streamlit as st
import pandas as pd
import joblib

# Load saved model objects
model = joblib.load("aqi_category_model.pkl")
label_encoder = joblib.load("aqi_category_labelencoder.pkl")
features = joblib.load("aqi_category_features.pkl")

st.set_page_config(page_title="AQI Category Prediction", layout="centered")
st.title("AQI Category Prediction App")

st.write(
    "Enter key pollutant values (PM2.5, PM10, NO2). "
    "Other pollutants are assumed to be at average levels."
)

# ------------------------
# User Inputs (Only 3)
# ------------------------
pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=50.0)
pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=80.0)
no2  = st.number_input("NO2 (µg/m³)",  min_value=0.0, value=40.0)

# ------------------------
# Auto-filled values
# ------------------------
default_values = {
    "PM2.5": pm25,
    "PM10": pm10,
    "NO2": no2,
    "OZONE": 30.0,
    "SO2": 10.0,
    "CO": 0.5,
    "NH3": 5.0
}

# Build input dataframe in correct order
input_data = {feature: default_values.get(feature, 0) for feature in features}
input_df = pd.DataFrame([input_data])

# ------------------------
# Prediction
# ------------------------
if st.button("Predict AQI Category"):
    pred = model.predict(input_df)
    category = label_encoder.inverse_transform(pred)

    st.success(f"Predicted AQI Category: **{category[0]}**")
