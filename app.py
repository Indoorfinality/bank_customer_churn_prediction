import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Bank Churn Prediction", layout="wide", page_icon="🏦")

# Load saved objects
stacking_model = joblib.load('stacking.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
feature_names = joblib.load('feature_names.pkl')

# Custom CSS aligned with your portfolio
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #6b21a8, #c084fc);
        color: #ffffff;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(107, 33, 168, 0.3);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(107, 33, 168, 0.5);
    }
    
    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .input-section:hover {
        transform: translateY(-5px);
    }
    
    .gradient-text {
        background: linear-gradient(45deg, #6b21a8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .stNumberInput input, .stSelectbox select {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    .stNumberInput label, .stSelectbox label {
        color: #e2e8f0 !important;
    }
    
    .floating-icon {
        position: fixed;
        bottom: 20px;
        right: 20px;
        font-size: 24px;
        color: #6b21a8;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }
    
    </style>
""", unsafe_allow_html=True)

# Add floating animation icon
st.markdown("""<div class="floating-icon">🏦</div>""", unsafe_allow_html=True)

# Header with gradient text
st.markdown("""
    <h1 style='text-align: center; margin-bottom: 2rem;'>
        <span class="gradient-text">Bank Customer Churn Prediction</span>
    </h1>
""", unsafe_allow_html=True)

# Create horizontal layout with columns
col1, col2, col3 = st.columns(3)

# Column 1 - Personal Info
with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="gradient-text">👤 Personal Information</h3>', unsafe_allow_html=True)
    credit_score = st.number_input("Credit Score", 0, 1000, 600)
    age = st.number_input("Age", 18, 100, 40)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
    st.markdown('</div>', unsafe_allow_html=True)

# Column 2 - Banking Info
with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="gradient-text">💳 Banking Information</h3>', unsafe_allow_html=True)
    tenure = st.number_input("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, step=1000.0, value=50000.0, format="%.2f")
    num_products = st.number_input("Number of Products", 1, 4, 1)
    estimated_salary = st.number_input("Estimated Salary", 0.0, step=1000.0, value=50000.0, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

# Column 3 - Status Info
with col3:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="gradient-text">📊 Status Information</h3>', unsafe_allow_html=True)
    has_cr_card = st.selectbox("Has Credit Card", [1, 0], 
                              format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.selectbox("Is Active Member", [1, 0], 
                                  format_func=lambda x: 'Yes' if x == 1 else 'No')
    st.markdown('</div>', unsafe_allow_html=True)

# Feature engineering
balance_to_salary = balance / estimated_salary if estimated_salary != 0 else 0
tenure_to_age = tenure / age if age != 0 else 0
balance_age_interaction = balance * age
products_age_interaction = num_products * age

# Encode categorical variables
gender_male = 1 if gender == 'Male' else 0
geography_spain = 1 if geography == 'Spain' else 0
geography_germany = 1 if geography == 'Germany' else 0

# Create input DataFrame
user_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Balance_to_Salary': [balance_to_salary],
    'Tenure_to_Age': [tenure_to_age],
    'Balance_Age_Interaction': [balance_age_interaction],
    'Products_Age_Interaction': [products_age_interaction],
    'Geography_Spain': [geography_spain],
    'Geography_Germany': [geography_germany],
    'Gender_Male': [gender_male]
})

# Reorder to match training features
user_data = user_data[feature_names]

# Predict button and results
st.markdown("---")
predict_button = st.button("🚀 Predict Churn Probability", use_container_width=True)

if predict_button:
    # Scale and apply PCA
    user_data_scaled = scaler.transform(user_data)
    user_data_pca = pca.transform(user_data_scaled)
    
    # Predict
    prediction = stacking_model.predict(user_data_pca)
    probability = stacking_model.predict_proba(user_data_pca)[:, 1][0]
    
    # Result animation and styling
    st.markdown(f"""
    <div class="prediction-box">
        <h3 style="margin-bottom: 1.5rem;">📈 Prediction Results</h3>
        <div style="font-size: 1.5rem; margin-bottom: 1rem;">
            Churn Prediction: <span style="color: {'#dc2626' if prediction[0] == 1 else '#10b981'}">{'🚨 Yes' if prediction[0] == 1 else '✅ No'}</span>
        </div>
        <div style="font-size: 2rem; font-weight: bold;">
            Churn Probability: <span style="color: #6b21a8">{probability:.2%}</span>
        </div>
        <div style="margin-top: 2rem;">
            {'⚠️ High Risk Customer - Immediate Action Recommended' if prediction[0] == 1 else '🎉 Low Risk Customer - Keep Up the Good Work!'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add confetti animation for positive result
    if prediction[0] == 0:
        st.balloons()