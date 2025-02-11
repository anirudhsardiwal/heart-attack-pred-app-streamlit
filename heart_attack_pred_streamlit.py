import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(
    "/Users/Sardiwal_Anirudh/Documents/App Deployment/Heart-Attack-Prediction-App/US_Heart_Patients.csv"
)

with st.form("main_form", enter_to_submit=False):
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 35, step=1)
    education = st.selectbox("Education", ["School", "Bachelors", "Masters", "PhD"])
    currentSmoker = st.radio("Smoker", ["Yes", "No"])
    cigsPerDay = st.number_input("Cigs Per Day", 0)
    BP_meds = st.radio("BP Meds", ["Yes", "No"])
    stroke = st.radio("Stroke", ["Yes", "No"])
    hypertension = st.radio("Hypertension", ["Yes", "No"])
    diabetes = st.radio("Diabetes", ["Yes", "No"])
    cholesterol = st.number_input("Total Cholesterol", 0, value=180)
    syst_bp = st.number_input("Systolic BP", 0, value=120)
    dyst_bp = st.number_input("Diastolic BP", 0, value=80)
    BMI = st.number_input("BMI", 0, value=22)
    heartRate = st.number_input("Heart Rate", 0, value=72)
    glucose = st.number_input("Glucose Level", 0, value=100)

    submitted = st.form_submit_button("Submit")

input_data = {
    "Gender": gender,
    "age": age,
    "education": education,
    "currentSmoker": currentSmoker,
    "cigsPerDay": cigsPerDay,
    "BP Meds": BP_meds,
    "prevalentStroke": stroke,
    "prevalentHyp": hypertension,
    "diabetes": diabetes,
    "tot cholesterol": cholesterol,
    "Systolic BP": syst_bp,
    "Diastolic BP": dyst_bp,
    "BMI": BMI,
    "heartRate": heartRate,
    "glucose": glucose,
}

input_df = pd.DataFrame(input_data, index=[0])

input_df.replace({"Male": 1, "Female": 0}, inplace=True)
input_df.replace({"Yes": 1, "No": 0}, inplace=True)
input_df.replace({"School": 1, "Bachelors": 2, "Masters": 3, "PhD": 4}, inplace=True)
input_df = input_df.astype("int")
input_df


model = RandomForestClassifier()
model.fit()

# predict_proba = loaded_model.predict_proba(input_row)
