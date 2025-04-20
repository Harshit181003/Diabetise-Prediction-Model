import streamlit as st
import joblib
import pandas as pd
import numpy as np

loaded_model = joblib.load("diabetes_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

st.title("Diabetes Predicter")
st.markdown("### Enter the details below to check diabetes risk:")

glucose = st.number_input("Glucose Level", min_value=0, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1)

def predict_diabetes():
    new_data = pd.DataFrame([[glucose, blood_pressure, bmi, dpf, age]], 
                             columns=["Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"])
    
    new_data_scaled = loaded_scaler.transform(new_data)
    prediction = loaded_model.predict(new_data_scaled)
    
    return "Diabetic 🩺" if prediction[0] == 1 else "Not Diabetic ✅"

if st.button("Predict"):
    if glucose and blood_pressure and bmi and dpf and age:  
        result = predict_diabetes()
        st.success(f"**Prediction: {result}**")
    else:
        st.warning("⚠️ Please enter all values before predicting.")
