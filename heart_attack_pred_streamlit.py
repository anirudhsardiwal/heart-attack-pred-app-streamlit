import numpy as np
import pandas as pd
import pickle
import streamlit as st

df = pd.read_csv(
    "/Users/Sardiwal_Anirudh/Documents/App Deployment/Heart-Attack-Prediction-App/US_Heart_Patients.csv"
)
df

with st.form("main_form", enter_to_submit=False):
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, step=1)
    education = st.select_slider

df.columns

"Gender"
"age"
"education"
"currentSmoker"
"cigsPerDay"
"BP Meds"
"prevalentStroke"
"prevalentHyp"
"diabetes"
"tot cholesterol"
"Systolic BP"
"Diastolic BP"
"BMI"
"heartRate"
"glucose"
"Heart-Att"
