pip install joblib
import requests
import joblib
import numpy as np
import streamlit as st

# Function to download model from GitHub
def download_model(url, model_filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(model_filename, 'wb') as f:
            f.write(response.content)
        return model_filename
    else:
        st.error(f"Failed to download model from {url}")
        return None

# Streamlit Title and Header
st.title('The Loan Approval Process')
st.header('(Use valid Input )', divider=True)

# Model URLs (Update these with your actual GitHub links)
logistic_model_url = "https://raw.githubusercontent.com/akash2728-02/Loan_Approval_Prediction/APP1/Logistic.pkl"
random_model_url = "https://raw.githubusercontent.com/akash2728-02/Loan_Approval_Prediction/APP1/random.pkl"

# Download models from GitHub
logistic_model_filename = download_model(logistic_model_url, "Logistic.pkl")
random_model_filename = download_model(random_model_url, "random.pkl")

# Load models using joblib
if logistic_model_filename and random_model_filename:
    try:
        with open(logistic_model_filename, 'rb') as model_file:
            model = joblib.load(model_file)  # Use joblib to load the model
        with open(random_model_filename, 'rb') as model_file2:
            model2 = joblib.load(model_file2)  # Use joblib to load the model
        st.write("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
else:
    st.error("One or both model files failed to download.")

# Input fields for the user to provide loan data
Gender = st.number_input("Enter Gender (1 for Male, 0 for Female)", step=1)
Married = st.number_input("Enter Married Status (1 for Yes, 0 for No)", step=1)
Dependents = st.number_input("Enter Number of Dependents", step=1)
Education = st.number_input('Enter Education (1 for Graduate, 0 for Not Graduate)', step=1)
Self_Employed = st.number_input("Enter Self_Employed Status (1 for Yes, 0 for No)", step=1)
ApplicantIncome = st.number_input('Enter Applicant Income', step=1) 
CoapplicantIncome = st.number_input('Enter Coapplicant Income', step=1)
LoanAmount = st.number_input('Enter Loan Amount', step=1)
Loan_Amount_Term = st.number_input('Enter Loan Amount Term (360)', step=1)
Credit_History = st.number_input("Enter Credit History (1 for Yes, 0 for No)", step=1)
Property_Area = st.number_input('Enter Property Area (0 for Rural, 1 for Semiurban, 2 for Urban)', step=1)

# Button for prediction with Logistic Regression model
if st.button('Predict with Logistic Regression'):
    input_data = np.array([Gender, Married, Dependents, Education, Self_Employed, 
                           ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, 
                           Credit_History, Property_Area])
    input_data = input_data.reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(input_data)
    
    if prediction == 'Y':
        st.success('Loan Approved! Thank you for using the app.')
    else:
        st.error('Loan Rejected! Thank you for using the app.')

# Button for prediction with Random Forest model
elif st.button('Predict with Random Forest'):
    input_data = np.array([Gender, Married, Dependents, Education, Self_Employed, 
                           ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, 
                           Credit_History, Property_Area])
    input_data = input_data.reshape(1, -1)  # Reshape for prediction
    prediction = model2.predict(input_data)
    
    if prediction == 'Y':
        st.success('Loan Approved! Thank you for using the app.')
    else:
        st.error('Loan Rejected! Thank you for using the app.')

# Footer with creator information
st.header('', divider=True)
st.header('Created by Akash Gawade, T & C Apply')

