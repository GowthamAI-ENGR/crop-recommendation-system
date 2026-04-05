import streamlit as st
import pickle
import numpy as np

# Load pickle files
model = pickle.load(open("model.pkl", "rb"))
minmax = pickle.load(open("minmaxscaler.pkl", "rb"))
standard = pickle.load(open("standscaler.pkl", "rb"))

# Crop label mapping
crop_dict = {
    0: "apple",
    1: "banana",
    2: "blackgram",
    3: "chickpea",
    4: "coconut",
    5: "coffee",
    6: "cotton",
    7: "grapes",
    8: "jute",
    9: "kidneybeans",
    10: "lentil",
    11: "maize",
    12: "mango",
    13: "mothbeans",
    14: "mungbean",
    15: "muskmelon",
    16: "orange",
    17: "papaya",
    18: "pigeonpeas",
    19: "pomegranate",
    20: "rice",
    21: "watermelon"
}

# Page settings
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("🌱 Crop Recommendation System")
st.write("Enter soil and weather details to get the recommended crop")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Apply scaling
    data_minmax = minmax.transform(input_data)
    final_data = standard.transform(data_minmax)

    # Prediction
    prediction = model.predict(final_data)[0]

    # Convert number to crop name
    crop_name = crop_dict.get(prediction, "Unknown Crop")

    st.success(f"🌾 Recommended Crop: {crop_name}")