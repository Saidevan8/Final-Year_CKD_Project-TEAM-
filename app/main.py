import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from googletrans import Translator

# Page configuration for mobile responsiveness
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.title("Chronic Kidney Disease Prediction")
# Load the models
@st.cache_resource
def load_models():
    model = joblib.load("models/m4/rf_model.joblib")
    return model

model= load_models()

# Model performance metrics
def calculate_model_metrics():
    # Example validation data - replace with your actual validation data
    validation_predictions = [0, 1, 0, 1, 0]
    validation_true = [0, 1, 0, 0, 1]
    
    metrics = {
        'Accuracy': accuracy_score(validation_true, validation_predictions),
        'Precision': precision_score(validation_true, validation_predictions),
        'Recall': recall_score(validation_true, validation_predictions),
        'F1-Score': f1_score(validation_true, validation_predictions)
    }
    return metrics


# Recommendation system
def get_recommendations(prediction, input_data):
    recommendations = {
        'lifestyle': [],
        'medical': [],
        'diet': []
    }
    
    if prediction == 1:
        recommendations['lifestyle'] = [
            "Maintain regular physical activity (30 minutes daily)",
            "Ensure adequate sleep (7-8 hours)",
            "Reduce stress through meditation or relaxation techniques"
        ]
        recommendations['medical'] = [
            "Schedule immediate consultation with a nephrologist",
            "Monitor blood pressure daily",
            "Regular kidney function tests"
        ]
        recommendations['diet'] = [
            "Reduce salt intake",
            "Control protein consumption",
            "Stay hydrated with appropriate fluid intake"
        ]
    else:
        recommendations['lifestyle'] = [
            "Continue regular exercise routine",
            "Maintain healthy sleep habits",
            "Regular health check-ups"
        ]
        recommendations['medical'] = [
            "Annual kidney function screening",
            "Regular blood pressure monitoring",
            "Stay up to date with vaccinations"
        ]
        recommendations['diet'] = [
            "Maintain a balanced diet",
            "Moderate salt intake",
            "Regular water consumption"
        ]
    
    return recommendations

    # All your original input fields remain the same
age = st.number_input("Age", min_value=0, max_value= 120, value=30, step=1)
gender = st.selectbox("Gender", options=["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", options=["Caucasian", "African American", "Asian","Other"])  # Customize groups
socioeconomic_status = st.selectbox("Socioeconomic Status", options=["Low", "Medium", "High"])
education_level = st.selectbox("Education Level", options=["High School", "Bachelors", "Higher","None"])
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
smoking = st.selectbox("Smoking Status", options=["Yes", "No"])
alcohol_consumption = st.slider("Alcohol Consumption in Units",min_value=0,max_value=20,value=10,step=1)
physical_activity = st.slider("Physical Activity in Hours", min_value=0,max_value=10,value=5,step=1)
diet_quality = st.slider("Diet Quality Score", min_value=0,max_value=10,value=5,step=1)
sleep_quality = st.slider("Sleep Quality", min_value=4,max_value=10,value=6,step=1)
family_history_kidney_disease = st.selectbox("Family History of Kidney Disease", options=["Yes", "No"])
family_history_hypertension = st.selectbox("Family History of Hypertension", options=["Yes", "No"])
family_history_diabetes = st.selectbox("Family History of Diabetes", options=["Yes", "No"])
previous_acute_kidney_injury = st.selectbox("Previous Acute Kidney Injury", options=["Yes", "No"])
urinary_tract_infections = st.selectbox("History of Urinary Tract Infections", options=["Yes", "No"])
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=90, max_value=180, value=120, step=1)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=60, max_value=120, value=80, step=1)
fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=70, max_value=200, value=90, step=1)
hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=10.0, value=5.5, step=0.1)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
bun_levels = st.number_input("BUN Levels (mg/dL)", min_value=5, max_value=50, value=20, step=1)
gfr = st.number_input("Glomerular Filtration Rate (GFR)", min_value=15, max_value=120, value=90, step=1)
protein_in_urine = st.selectbox("Protein in Urine", options=["Negative", "0+", "1+", "2+", "3+", "4+"] )
acr = st.number_input("Albumin-to-Creatinine Ratio (ACR)", min_value=0.0, max_value=300.0, value=30.0, step=0.1)
serum_sodium = st.number_input("Serum Electrolytes - Sodium (mEq/L)", min_value=135, max_value=150, value=140, step=1)
serum_potassium = st.number_input("Serum Electrolytes - Potassium (mEq/L)", min_value=3.5, max_value=7.0, value=4.0, step=0.1)
serum_calcium = st.number_input("Serum Electrolytes - Calcium (mg/dL)", min_value=4.0, max_value=12.0, value=9.0, step=0.1)
serum_phosphorus = st.number_input("Serum Electrolytes - Phosphorus (mg/dL)", min_value=1.0, max_value=10.0, value=3.5, step=0.1)
hemoglobin_levels = st.number_input("Hemoglobin Levels (g/dL)", min_value=8.5, max_value=20.5, value=14.0, step=0.1)
cholesterol_total = st.number_input("Cholesterol Total (mg/dL)", min_value=100, max_value=300, value=200, step=1)
cholesterol_ldl = st.number_input("Cholesterol LDL (mg/dL)", min_value=50, max_value=200, value=100, step=1 )
cholesterol_hdl = st.number_input("Cholesterol HDL (mg/dL)", min_value=20, max_value=100, value=50, step=1)
cholesterol_triglycerides = st.number_input("Cholesterol Triglycerides (mg/dL)", min_value=50, max_value=500, value=150, step=1)
ace_inhibitors = st.selectbox("Taking ACE Inhibitors", options=["Yes", "No"])
diuretics = st.selectbox("Taking Diuretics", options=["Yes", "No"])
nsaids_use = st.slider("Using NSAIDs", min_value=0,max_value=10,value=5,step=1)
statins = st.selectbox("Taking Statins", options=["Yes", "No"])
antidiabetic_medications = st.selectbox("Taking Antidiabetic Medications", options=["Yes", "No"])
edema = st.selectbox("Presence of Edema", options=["Yes", "No"])
fatigue_levels = st.slider("Fatigue Levels", min_value=0,max_value=10,value=5,step=1)
nausea_vomiting = st.number_input("Nausea or Vomiting",min_value=0,max_value=7,value=3,step=1)
muscle_cramps = st.number_input("Muscle Cramps", min_value=0,max_value=7,value=3,step=1)
itching = st.number_input("Itching", min_value=0,max_value=10,value=3,step=1)
quality_of_life_score = st.number_input("Quality of Life Score", min_value=0, max_value=100, value=50, step=1)
heavy_metals_exposure = st.selectbox("Exposure to Heavy Metals", options=["Yes", "No"])
occupational_exposure_chemicals = st.selectbox("Occupational Exposure to Chemicals", options=["Yes", "No"])
water_quality = st.selectbox("Water Quality", options=["Poor", "Good"])
medical_checkups_frequency = st.selectbox("Frequency of Medical Checkups", options=["Rarely", "Annually", "Frequently"])
medication_adherence = st.selectbox("Medication Adherence", options=["Poor", "Average", "Good"])
health_literacy = st.selectbox("Health Literacy", options=["Low", "Medium", "High"])

# Original convert_to_numerical function
def convert_to_numerical():
    gender_map = {"Male": 0, "Female": 1}
    ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other":3}  # Customize as needed
    socioeconomic_status_map = {"Low": 0, "Medium": 1, "High": 2}
    education_level_map = {"None": 0, "High School": 1, "Bachelors": 2, "Higher":3}
    smoking_map = {"Yes": 1, "No": 0}
    family_history_map = {"Yes": 1, "No": 0}
    protein_in_urine_map = {"Negative": 0, "0+": 0.3, "1+":1, "2+": 1, "3+": 3, "4+": 4}
    water_quality_map = {"Poor": 0, "Good": 1}
    medication_adherence_map = {"Poor": 0, "Average": 5, "Good": 10}
    health_literacy_map = {"Low": 0, "Medium": 1, "High": 2}
    medical_checkups_frequency_map = {"Rarely": 0, "Annually": 1, "Frequently": 2}
        # Return the mapped values in the same order as your original code
    return [
            age, 
            gender_map[gender], 
            ethnicity_map[ethnicity], 
            socioeconomic_status_map[socioeconomic_status], 
            education_level_map[education_level], 
            bmi, 
            smoking_map[smoking], 
            alcohol_consumption, 
            physical_activity, 
            diet_quality, 
            sleep_quality, 
            family_history_map[family_history_kidney_disease], 
            family_history_map[family_history_hypertension], 
            family_history_map[family_history_diabetes], 
            family_history_map[previous_acute_kidney_injury], 
            family_history_map[urinary_tract_infections], 
            systolic_bp, 
            diastolic_bp, 
            fasting_blood_sugar, 
            hba1c, 
            serum_creatinine, 
            bun_levels, 
            gfr, 
            protein_in_urine_map[protein_in_urine], 
            acr, 
            serum_sodium, 
            serum_potassium, 
            serum_calcium, 
            serum_phosphorus, 
            hemoglobin_levels, 
            cholesterol_total, 
            cholesterol_ldl, 
            cholesterol_hdl, 
            cholesterol_triglycerides, 
            family_history_map[ace_inhibitors], 
            family_history_map[diuretics], 
            nsaids_use, 
            family_history_map[statins], 
            family_history_map[antidiabetic_medications], 
            family_history_map[edema], 
            fatigue_levels, 
            nausea_vomiting, 
            muscle_cramps, 
            itching, 
            quality_of_life_score, 
            family_history_map[heavy_metals_exposure], 
            family_history_map[occupational_exposure_chemicals], 
            water_quality_map[water_quality], 
            medical_checkups_frequency_map[medical_checkups_frequency], 
            medication_adherence_map[medication_adherence], 
            health_literacy_map[health_literacy]
        ]

# Prediction button and results
if st.button('predict_button'):
    input_features = convert_to_numerical()
    prediction = model.predict([input_features])
    # Display prediction
    if prediction == 1:
        st.error('high_risk')
    else:
        st.success('low_risk')

    st.subheader("Personalized Recommendations")
    recommendations = get_recommendations(prediction[0], input_features)
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Lifestyle Recommendations")
        for rec in recommendations['lifestyle']:
            st.write(f"• {rec}")
    with col2:
        st.write("Medical Recommendations")
        for rec in recommendations['medical']:
            st.write(f"• {rec}")
    with col3:
        st.write("Dietary Recommendations")
        for rec in recommendations['diet']:
            st.write(f"• {rec}")

# Mobile responsiveness CSS
st.markdown("""
    <style>
        .stApp {
            max-width: 100%;
            padding: 1rem;
        }
        @media (max-width: 768px) {
            .stApp {
                padding: 0.5rem;
            }
            .st-bw {
                flex-direction: column;
            }
        }
    </style>
    """, unsafe_allow_html=True)