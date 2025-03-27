






# -*- coding: utf-8 -*-



import numpy as np 
import streamlit as st
import pickle as pk


st.title('The Loan Aproval Process')
st.header('(Use valid Input )',divider=True)


model_path = r'C:\Users\akash\All Internships\AIS Intern\ML Project\demo\Logistic.pkl'

with open (model_path,'rb') as model_file:
    model=pk.load(model_file)

model_path2 = r"C:\Users\akash\All Internships\AIS Intern\ML Project\demo\random.pkl"

with open (model_path2,'rb') as model_file2:
    model2 = pk.load(model_file2)


# Input fields

Gender = st.number_input("Enter Gender(1 for Male OW 0)",step=1)
Married = st.number_input("Enter Married Status( 1 for Yes OW 0)",step=1)
Dependents = st.number_input("Enter Dependents",step=1)
Education = st.number_input('Enter Education (1 for Graduate and 0 otherwise)',step=1)
Self_Employed = st.number_input("enter the Self_Employed statue ( 1 for Yes) ",step=1)
ApplicantIncome = st.number_input('Enter ApplicantIncome',step=1) 
CoapplicantIncome = st.number_input('Enter CoapplicantIncome',step=1 )
LoanAmount = st.number_input('enter LoanAmount',step=1)
Loan_Amount_Term = st.number_input('Enter Loan_Amount_Term (360)',step=1)
Credit_History  = st.number_input("enter Credit_History (1 for Yes ow 0)",step=1)
Property_Area = st.number_input('Enter Property_Area (0 for Rural,1 for semiurban ,2 for Urban)',step=1)



if st.button('Predict'):
    input_data = np.array([Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area])
    input_data= input_data.reshape(1,-1)
    prediction = model.predict(input_data)
    
    if prediction == 'Y':
        st.success('Aprove Loan , Thank You For Using The App')
    else:
        st.error('Reject The Loan, Thank You fror using The App')
        
elif st.button('Predict With RAndom Forest'):
        
        
        input_data = np.array([Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area])
        input_data= input_data.reshape(1,-1)
        prediction = model2.predict(input_data)
        
        if prediction == 'Y':
            st.success('Aprove Loan,Thank You For Using The App')
        else:
            st.error('Reject The Loan,Thank You For Using The App')
    

st.header('',divider=True )
st.header('Created by Akash Gawade ,T & C apply')



