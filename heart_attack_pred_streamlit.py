import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("US_Heart_Patients.csv")

df.replace({"Male": 1, "Female": 0}, inplace=True)

X = df.drop("Heart-Att", axis=1)
y = df["Heart-Att"]

model = RandomForestClassifier()
model.fit(X, y)

col1, col2, col3 = st.columns([1, 3, 1], gap="small")

with col2:
    st.title("Heart Attack Prediction")
    st.write("#### Make a single prediction:")
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
    input_df.replace(
        {"School": 1, "Bachelors": 2, "Masters": 3, "PhD": 4}, inplace=True
    )
    input_df = input_df.astype("int")

    result = model.predict(input_df)

    predict_proba = model.predict_proba(input_df)

    if submitted:
        if result == 1:
            st.success(
                f"Heart Attack __likely__ with {predict_proba[0][1]:.0%} probability"
            )
        elif result == 0:
            st.success(
                f"Heart Attack __unlikely__ with {predict_proba[0][0]:.0%} probability**"
            )

    ############################# File Upload #################################

    st.html("<h2 style='text-align:center;'> ------- OR -------</h2>")
    st.write("#### Predict in bulk:")

    uploaded_file = st.file_uploader("Upload a file", "csv")

    if uploaded_file is not None and uploaded_file != "":
        data = pd.read_csv(uploaded_file)
        data["Gender"].replace({"Male": 1, "Female": 0}, inplace=True)
        data.replace({"Yes": 1, "No": 0}, inplace=True)
        data.replace(
            {"School": 1, "Bachelors": 2, "Masters": 3, "PhD": 4}, inplace=True
        )
        data = data.astype("int")
        labels = model.predict(data)
        outcome_mapping = {0: "Healthy", 1: "Not Healthy"}
        data["Predictions"] = [outcome_mapping[label] for label in labels]

        original_file_name = uploaded_file.name.split(".")[0]
        result_filename = f"{original_file_name}_predictions.csv"

        csv = data.to_csv(index=False)

        st.download_button("Download predictions", csv, result_filename, "text/csv")

        st.write(data)
