import streamlit as st
import pandas as pd
import joblib

# Load saved model objects
model = joblib.load("aqi_category_model.pkl")
label_encoder = joblib.load("aqi_category_labelencoder.pkl")
features = joblib.load("aqi_category_features.pkl")

st.title("AQI Category Prediction App")

st.write("Pollutant values enter karke AQI category predict karein")

input_data = {}

for col in features:
    input_data[col] = st.number_input(
        col,
        min_value=0.0,
        value=0.0,
        step=0.1
    )

input_df = pd.DataFrame([input_data])

if st.button("Predict AQI"):
    pred = model.predict(input_df)
    category = label_encoder.inverse_transform(pred)
    st.success(f"Predicted AQI Category: {category[0]}")
