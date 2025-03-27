import numpy as np
import streamlit as st
import joblib  # Use joblib instead of pickle

# Title and description
st.title('Loan Approval Prediction')
st.header('Please Enter the Details Below:', divider=True)

# File paths (ensure correct model paths on Streamlit Cloud)
model_path1 = 'path_to_your_models/Logistic.pkl'  # Update this to your actual model path on Streamlit Cloud
model_path2 = 'path_to_your_models/random.pkl'   # Update this to your actual model path on Streamlit Cloud

# Load Logistic Model
try:
    with open(model_path1, 'rb') as model_file:
        model = joblib.load(model_file)
    st.success("Logistic Regression Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading Logistic model: {e}")

# Load Random Forest Model
try:
    with open(model_path2, 'rb') as model_file2:
        model2 = joblib.load(model_file2)
    st.success("Random Forest Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading Random Forest model: {e}")

# Input fields
Gender = st.number_input("Enter Gender (1 for Male, 0 for Female)", step=1)
Married = st.number_input("Enter Married Status (1 for Yes, 0 for No)", step=1)
Dependents = st.number_input("Enter Dependents", step=1)
Education = st.number_input("Enter Education (1 for Graduate, 0 otherwise)", step=1)
Self_Employed = st.number_input("Enter Self_Employed Status (1 for Yes, 0 for No)", step=1)
ApplicantIncome = st.number_input("Enter Applicant Income", step=1)
CoapplicantIncome = st.number_input("Enter Coapplicant Income", step=1)
LoanAmount = st.number_input("Enter Loan Amount", step=1)
Loan_Amount_Term = st.number_input("Enter Loan Amount Term (e.g. 360)", step=1)
Credit_History = st.number_input("Enter Credit History (1 for Good, 0 for Poor)", step=1)
Property_Area = st.number_input("Enter Property Area (0 for Rural, 1 for Semiurban, 2 for Urban)", step=1)

# Prediction Button for Logistic Model
if st.button('Predict with Logistic Model'):
    input_data = np.array([Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area])
    input_data = input_data.reshape(1, -1)

    try:
        prediction = model.predict(input_data)
        if prediction == 'Y':
            st.success('Loan Approved! Thank you for using the app.')
        else:
            st.error('Loan Denied! Thank you for using the app.')
    except Exception as e:
        st.error(f"Error making prediction with Logistic model: {e}")

# Prediction Button for Random Forest Model
elif st.button('Predict with Random Forest Model'):
    input_data = np.array([Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area])
    input_data = input_data.reshape(1, -1)

    try:
        prediction = model2.predict(input_data)
        if prediction == 'Y':
            st.success('Loan Approved! Thank you for using the app.')
        else:
            st.error('Loan Denied! Thank you for using the app.')
    except Exception as e:
        st.error(f"Error making prediction with Random Forest model: {e}")

# Footer information
st.header('', divider=True)
st.header('Created by Akash Gawade | Terms and Conditions Apply')
