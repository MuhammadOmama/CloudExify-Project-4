import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('disease_model.pkl')
    features = joblib.load('disease_features.pkl')
    return model, features

model, features = load_artifacts()

st.title("🩺 Medical Disease Risk Classifier")
st.markdown("Enter the patient's diagnostic measurements to predict disease risk.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120, step=1)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80, step=1)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=33, step=1)

    submit_btn = st.form_submit_button("Predict Disease Risk")

if submit_btn:
    patient_data = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }])[features]

    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0, 1]

    if prediction == 1:
        st.error(f"### ⚠️ DISEASE LIKELY\n**Confidence / Probability:** {probability:.1%}")
    else:
        st.success(f"### ✅ NO DISEASE DETECTED\n**Confidence / Probability:** {(1 - probability):.1%}")