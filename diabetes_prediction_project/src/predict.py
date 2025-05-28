"""
This script provides a simple prediction interface for the diabetes prediction model.
It loads the trained model and allows users to make predictions on new data.
"""

import joblib
import pandas as pd
import numpy as np

def load_model_and_scaler():
    """
    Load the trained model and scaler
    
    Returns:
        tuple: (model, scaler) - The trained model and scaler
    """
    try:
        model = joblib.load('diabetes_prediction_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("Error: Model files not found. Please ensure the model has been trained and saved.")
        return None, None

def preprocess_input(data, scaler):
    """
    Preprocess input data for prediction
    
    Args:
        data (dict): Dictionary containing patient information
        scaler (StandardScaler): Trained scaler for feature normalization
    
    Returns:
        numpy.ndarray: Preprocessed data ready for prediction
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([data])
    
    # Handle categorical variables
    if 'gender' in input_df.columns:
        # Create dummy variables for gender
        input_df = pd.get_dummies(input_df, columns=['gender'], drop_first=True)
    
    if 'smoking_history' in input_df.columns:
        # Create dummy variables for smoking_history
        input_df = pd.get_dummies(input_df, columns=['smoking_history'], drop_first=True)
    
    # Ensure all columns from training are present
    expected_columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 
                        'blood_glucose_level', 'gender_Male', 'gender_Other', 
                        'smoking_history_current', 'smoking_history_ever', 
                        'smoking_history_former', 'smoking_history_never', 
                        'smoking_history_not current']
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    return input_scaled

def predict_diabetes(data):
    """
    Predict diabetes based on input data
    
    Args:
        data (dict): Dictionary containing patient information
        
    Returns:
        tuple: (prediction, probability) - Binary prediction and probability of diabetes
    """
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        return None, None
    
    # Preprocess the input
    input_scaled = preprocess_input(data, scaler)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability

def main():
    """
    Main function to demonstrate the prediction functionality
    """
    print("Diabetes Prediction System")
    print("=========================")
    
    # Sample input
    sample_data = {
        'gender': 'Male',
        'age': 45.0,
        'hypertension': 0,
        'heart_disease': 0,
        'smoking_history': 'never',
        'bmi': 28.5,
        'HbA1c_level': 6.8,
        'blood_glucose_level': 140
    }
    
    prediction, probability = predict_diabetes(sample_data)
    
    if prediction is not None:
        print(f"\nSample Input: {sample_data}")
        print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
        print(f"Probability of diabetes: {probability:.4f}")
        print(f"Confidence: {probability * 100:.2f}% likelihood of diabetes")
    
if __name__ == "__main__":
    main()
