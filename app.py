import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('heart_model.pkl')

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# App title
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's medical details below and click **Predict** to check the risk level.")

# Input fields
age = st.number_input("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-4)", [0, 1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced  Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
slope=st.number_input("Slope of the Peak Exercise ST Segment (0-3)", 0, 3, 1)

# Convert input to model format
sex_val = 1 if sex == "Male" else 0
features = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg, thalach, 
                      exang, oldpeak,slope]])

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        st.error("⚠️ The patient is **likely to have heart disease**.")
    else:
        st.success("✅ The patient is **unlikely to have heart disease**.")

# Footer
st.markdown("---")
st.caption("Developed by Hahaguys | Machine Learning Project | Streamlit ❤️")




