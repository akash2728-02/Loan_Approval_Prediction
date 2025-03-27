st.write("Starting model download...")

def download_model(url, model_filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(model_filename, 'wb') as f:
            f.write(response.content)
        st.write(f"Model downloaded successfully: {model_filename}")
        return model_filename
    else:
        st.write(f"Failed to download model from {url} with status code {response.status_code}")
        return None

logistic_model_url = "https://raw.githubusercontent.com/akash2728-02/Loan_Approval_Prediction/APP1/Logistic.pkl"
random_model_url = "https://raw.githubusercontent.com/akash2728-02/Loan_Approval_Prediction/APP1/random.pkl"

logistic_model_filename = download_model(logistic_model_url, "Logistic.pkl")
random_model_filename = download_model(random_model_url, "random.pkl")

if logistic_model_filename:
    st.write("Loading Logistic model...")
    with open(logistic_model_filename, 'rb') as model_file:
        model = pk.load(model_file)
else:
    st.write("Logistic model download failed!")

if random_model_filename:
    st.write("Loading Random Forest model...")
    with open(random_model_filename, 'rb') as model_file2:
        model2 = pk.load(model_file2)
else:
    st.write("Random Forest model download failed!")



import numpy as np
import streamlit as st
import pickle as pk
import requests


# Title and Header
st.title('The Loan Approval Process')
st.header('(Use valid Input)', divider=True)




st.write("Starting model download...")

def download_model(url, model_filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(model_filename, 'wb') as f:
            f.write(response.content)
        st.write(f"Model downloaded successfully: {model_filename}")
        return model_filename
    else:
        st.write(f"Failed to download model from {url} with status code {response.status_code}")
        return None

logistic_model_url = "https://raw.githubusercontent.com/akash2728-02/Loan_Approval_Prediction/APP1/Logistic.pkl"
random_model_url = "https://raw.githubusercontent.com/akash2728-02/Loan_Approval_Prediction/APP1/random.pkl"

logistic_model_filename = download_model(logistic_model_url, "Logistic.pkl")
random_model_filename = download_model(random_model_url, "random.pkl")

if logistic_model_filename:
    st.write("Loading Logistic model...")
    with open(logistic_model_filename, 'rb') as model_file:
        model = pk.load(model_file)
else:
    st.write("Logistic model download failed!")

if random_model_filename:
    st.write("Loading Random Forest model...")
    with open(random_model_filename, 'rb') as model_file2:
        model2 = pk.load(model_file2)
else:
    st.write("Random Forest model download failed!")







# Title and Header
st.title('The Loan Approval Process')
st.header('(Use valid Input)', divider=True)

# Function to download models from GitHub
def download_model(url, model_filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(model_filename, 'wb') as f:
            f.write(response.content)
        return model_filename
    else:
        st.error(f"Failed to download model from {url}")
        return None

# URLs of the model files on GitHub (raw links)
logistic_model_url = "https://raw.githubusercontent.com/akash2728-02/Loan_Approval_Prediction/APP1/Logistic.pkl"
random_model_url = 'https://raw.githubusercontent.com/akash2728-02/Loan_Approval_Prediction/blob/APP1/Logistic.pkl'

# Download the models"https://raw.githubusercontent.com/akash2728-02/Loan_Approval_Prediction/APP1/random.pkl"
logistic_model_filename = download_model(logistic_model_url, "Logistic.pkl")
random_model_filename = download_model(random_model_url, "random.pkl")

# Load the models
if logistic_model_filename:
    with open(logistic_model_filename, 'rb') as model_file:
        model = pk.load(model_file)

if random_model_filename:
    with open(random_model_filename, 'rb') as model_file2:
        model2 = pk.load(model_file2)

# Input fields
Gender = st.number_input("Enter Gender (1 for Male or 0)", step=1)
Married = st.number_input("Enter Married Status (1 for Yes or 0)", step=1)
Dependents = st.number_input("Enter Dependents", step=1)
Education = st.number_input('Enter Education (1 for Graduate and 0 otherwise)', step=1)
Self_Employed = st.number_input("Enter the Self_Employed status (1 for Yes)", step=1)
ApplicantIncome = st.number_input('Enter Applicant Income', step=1)
CoapplicantIncome = st.number_input('Enter Coapplicant Income', step=1)
LoanAmount = st.number_input('Enter Loan Amount', step=1)
Loan_Amount_Term = st.number_input('Enter Loan Amount Term (e.g. 360)', step=1)
Credit_History = st.number_input("Enter Credit History (1 for Yes or 0)", step=1)
Property_Area = st.number_input('Enter Property Area (0 for Rural, 1 for Semiurban, 2 for Urban)', step=1)

# Prediction logic with Logistic model
if st.button('Predict with Logistic Regression'):
    input_data = np.array([Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
                           CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area])
    input_data = input_data.reshape(1, -1)
    prediction = model.predict(input_data)

    if prediction == 'Y':
        st.success('Approve Loan, Thank you for using the app!')
    else:
        st.error('Reject the Loan, Thank you for using the app!')

# Prediction logic with Random Forest model
elif st.button('Predict with Random Forest'):
    input_data = np.array([Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
                           CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area])
    input_data = input_data.reshape(1, -1)
    prediction = model2.predict(input_data)

    if prediction == 'Y':
        st.success('Approve Loan, Thank you for using the app!')
    else:
        st.error('Reject the Loan, Thank you for using the app!')

# Footer
st.header('', divider=True)
st.header('Created by Akash Gawade, T & C apply')
