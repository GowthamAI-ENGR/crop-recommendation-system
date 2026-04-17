import streamlit as st
import numpy as np
import pickle
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model and scalers
model = pickle.load(open(os.path.join(script_dir, 'model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(script_dir, 'standscaler.pkl'), 'rb'))
mx = pickle.load(open(os.path.join(script_dir, 'minmaxscaler.pkl'), 'rb'))

# Crop dictionary
crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

st.title("Crop Recommendation System 🌱")

st.markdown("Enter the soil and environmental parameters to get the best crop recommendation.")

# Input fields
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen", min_value=0.0, step=0.1)
with col2:
    P = st.number_input("Phosphorus", min_value=0.0, step=0.1)
with col3:
    K = st.number_input("Potassium", min_value=0.0, step=0.1)

col4, col5, col6 = st.columns(3)
with col4:
    temp = st.number_input("Temperature (°C)", step=0.01)
with col5:
    humidity = st.number_input("Humidity (%)", step=0.01)
with col6:
    ph = st.number_input("pH", step=0.01)

rainfall = st.number_input("Rainfall (mm)", step=0.01)

if st.button("Get Recommendation"):
    # Prepare features
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Transform
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    # Get crop
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        st.success(f"{crop} is the best crop to be cultivated right there!")
    else:
        st.error("Sorry, we could not determine the best crop to be cultivated with the provided data.")