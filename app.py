# app.py
import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('svm_model.pkl')

# Set up the Streamlit app
st.title('Diabetes Classification')

# Create input fields for user input
pregnancies = st.number_input('Pregnancies', format="%.2f")
glucose = st.number_input('Glucose', format="%.2f")
blood_pressure = st.number_input('Blood Pressure', format="%.2f")
skin_thickness = st.number_input('Skin Thickness', format="%.2f")
insulin = st.number_input('Insulin', format="%.2f")
bmi = st.number_input('BMI', format="%.2f")
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', format="%.2f")
age = st.number_input('Age', format="%.2f")

# Create a button to make predictions
if st.button('Predict'):
    # Prepare the input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    if prediction[0] == 1:
        st.write('Prediction: The person has diabetes.')
    else:
        st.write('Prediction: The person does not have diabetes.')
