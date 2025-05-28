"""
This script demonstrates how to use the diabetes prediction model.
It provides a simple command-line interface for making predictions.
"""

import joblib
import pandas as pd
import argparse

def load_model_and_scaler():
    """Load the trained model and scaler"""
    model = joblib.load('diabetes_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def predict_diabetes(gender, age, hypertension, heart_disease, 
                    smoking_history, bmi, hba1c_level, blood_glucose_level):
    """
    Predict diabetes based on input parameters
    
    Args:
        gender (str): Gender of the patient (Male, Female, Other)
        age (float): Age of the patient in years
        hypertension (int): Whether the patient has hypertension (0: No, 1: Yes)
        heart_disease (int): Whether the patient has heart disease (0: No, 1: Yes)
        smoking_history (str): Smoking history of the patient
        bmi (float): Body Mass Index
        hba1c_level (float): HbA1c level
        blood_glucose_level (float): Blood glucose level
        
    Returns:
        tuple: (prediction, probability) - Binary prediction and probability of diabetes
    """
    try:
        model, scaler = load_model_and_scaler()
    except FileNotFoundError:
        print("Error: Model files not found. Please run the notebook first to train and save the model.")
        return None, None
    
    # Create a DataFrame with the input data
    data = {
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level]
    }
    
    input_df = pd.DataFrame(data)
    
    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['gender', 'smoking_history'], drop_first=True)
    
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
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability

def main():
    """Main function to run the prediction from command line"""
    parser = argparse.ArgumentParser(description='Predict diabetes based on health metrics')
    
    parser.add_argument('--gender', type=str, required=True, choices=['Male', 'Female', 'Other'],
                        help='Gender of the patient')
    parser.add_argument('--age', type=float, required=True,
                        help='Age of the patient in years')
    parser.add_argument('--hypertension', type=int, required=True, choices=[0, 1],
                        help='Whether the patient has hypertension (0: No, 1: Yes)')
    parser.add_argument('--heart_disease', type=int, required=True, choices=[0, 1],
                        help='Whether the patient has heart disease (0: No, 1: Yes)')
    parser.add_argument('--smoking_history', type=str, required=True,
                        choices=['current', 'ever', 'former', 'never', 'not current', 'No Info'],
                        help='Smoking history of the patient')
    parser.add_argument('--bmi', type=float, required=True,
                        help='Body Mass Index')
    parser.add_argument('--hba1c_level', type=float, required=True,
                        help='HbA1c level')
    parser.add_argument('--blood_glucose_level', type=float, required=True,
                        help='Blood glucose level')
    
    args = parser.parse_args()
    
    prediction, probability = predict_diabetes(
        args.gender, args.age, args.hypertension, args.heart_disease,
        args.smoking_history, args.bmi, args.hba1c_level, args.blood_glucose_level
    )
    
    if prediction is not None:
        print(f"\nPrediction: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
        print(f"Probability of diabetes: {probability:.4f}")
        print(f"Confidence: {probability * 100:.2f}% likelihood of diabetes")
    
if __name__ == "__main__":
    main()
