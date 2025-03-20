import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Streamlit App Title
st.title("SmartPerf - AI-Powered Employee Insights 🚀")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])
if uploaded_file is not None:
    try:
        dataset = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.warning("Please upload a dataset to proceed.")

# Data Preprocessing
if uploaded_file is not None:
    dataset.drop(columns=['EmpNumber'], inplace=True, errors='ignore')  # Prevents error if column is missing

    # Use LabelEncoder for categorical data
    le_department = LabelEncoder()
    le_jobrole = LabelEncoder()

    dataset["EmpDepartment"] = le_department.fit_transform(dataset["EmpDepartment"])
    dataset["EmpJobRole"] = le_jobrole.fit_transform(dataset["EmpJobRole"])

    # Splitting Data
    X = dataset.drop(columns=['PerformanceRating'])
    y = dataset['PerformanceRating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Save Model
    joblib.dump(model, 'emp_perf_model.pkl')

# Prediction Section
st.subheader("Make a Prediction")
user_input = []
if uploaded_file is not None:
    for col in X.columns:
        value = st.number_input(f"Enter {col}", value=0.0)
        user_input.append(value)

    if st.button("Analyze Employee Insights"):
        model = joblib.load('emp_perf_model.pkl')
        prediction = model.predict([user_input])
        st.write(f"Predicted Performance Rating: {prediction[0]}")
