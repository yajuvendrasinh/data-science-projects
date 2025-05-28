# Diabetes Prediction Model

## Project Overview

This project aims to build a machine learning model to predict diabetes based on various health metrics. Early detection of diabetes is crucial for effective management and prevention of complications. The model uses a dataset with 100,000 records containing various health parameters to predict whether a person has diabetes.

## Dataset Information

The dataset used in this project contains 100,000 records with the following features:

- **gender**: Gender of the patient (Male, Female, Other)
- **age**: Age of the patient in years
- **hypertension**: Whether the patient has hypertension (0: No, 1: Yes)
- **heart_disease**: Whether the patient has heart disease (0: No, 1: Yes)
- **smoking_history**: Smoking history of the patient (current, former, never, etc.)
- **bmi**: Body Mass Index, a measure of body fat based on height and weight
- **HbA1c_level**: Hemoglobin A1c level, a measure of average blood sugar over the past 2-3 months
- **blood_glucose_level**: Current blood glucose level
- **diabetes**: Target variable indicating whether the patient has diabetes (0: No, 1: Yes)

## Project Structure

```
diabetes_prediction_project/
│
├── data/
│   └── diabetes_prediction_dataset.csv  # The dataset used for analysis
│
├── notebooks/
│   └── diabetes_prediction_model.ipynb  # Jupyter notebook with detailed analysis
│
├── src/
│   ├── diabetes_prediction_model.pkl    # Trained model (generated after running notebook)
│   ├── scaler.pkl                       # Feature scaler (generated after running notebook)
│   └── predict.py                       # Script for making predictions with the trained model
│
├── images/                              # Visualizations generated during analysis
│
└── README.md                            # This file
```

## Methods Used

### Data Preprocessing

1. **Data Cleaning**: Checked for missing values and handled them appropriately
2. **Feature Engineering**: Converted categorical variables to numerical using one-hot encoding
3. **Feature Scaling**: Standardized numerical features to have zero mean and unit variance

### Exploratory Data Analysis (EDA)

1. **Univariate Analysis**: Examined the distribution of individual features
2. **Bivariate Analysis**: Explored relationships between features and the target variable
3. **Correlation Analysis**: Identified correlations between different features

### Machine Learning Models

Three different models were implemented and compared:

1. **Logistic Regression**: A simple linear model for binary classification
2. **Decision Tree**: A non-linear model that makes decisions based on feature thresholds
3. **Random Forest**: An ensemble model that combines multiple decision trees

### Model Evaluation

Models were evaluated using the following metrics:

1. **Accuracy**: Proportion of correct predictions
2. **Precision**: Proportion of true positive predictions among all positive predictions
3. **Recall**: Proportion of true positive predictions among all actual positives
4. **F1 Score**: Harmonic mean of precision and recall
5. **AUC-ROC**: Area under the Receiver Operating Characteristic curve

## Key Findings

1. **Model Performance**: The Random Forest model performed the best among the three models, with the highest accuracy, precision, recall, F1 score, and AUC.

2. **Key Predictors**: The most important features for predicting diabetes are:
   - Blood glucose level
   - HbA1c level
   - Age
   - BMI

3. **Medical Relevance**: This aligns with medical knowledge, as high blood glucose levels and HbA1c levels are directly related to diabetes diagnosis.

4. **Risk Factors**: Age and BMI are significant risk factors for diabetes, which is also consistent with medical literature.

5. **Gender and Lifestyle Factors**: Gender and smoking history have some influence on diabetes risk, but they are less important than the physiological measurements.

## How to Use This Project

### Prerequisites

- Python 3.6 or higher
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

### Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/diabetes-prediction-project.git
cd diabetes-prediction-project
```

2. Install the required packages:
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Running the Analysis

1. Open and run the Jupyter notebook:
```
jupyter notebook notebooks/diabetes_prediction_model.ipynb
```

2. The notebook will:
   - Load and explore the dataset
   - Perform data preprocessing
   - Create visualizations for exploratory data analysis
   - Build and evaluate machine learning models
   - Save the best model for future use

### Making Predictions

You can use the trained model to make predictions on new data:

```python
from src.predict import predict_diabetes

# Example data for a patient
patient_data = {
    'gender': 'Male',
    'age': 45.0,
    'hypertension': 0,
    'heart_disease': 0,
    'smoking_history': 'never',
    'bmi': 28.5,
    'HbA1c_level': 6.8,
    'blood_glucose_level': 140
}

# Get prediction
prediction, probability = predict_diabetes(patient_data)
print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
print(f"Probability of diabetes: {probability:.4f}")
```

## Conclusions and Recommendations

### Conclusions

In this project, we built and compared three machine learning models to predict diabetes based on various health metrics. The Random Forest model performed the best, achieving high accuracy and AUC scores. The most important predictors were blood glucose level, HbA1c level, age, and BMI.

### Recommendations

1. **Regular Monitoring**: Individuals should regularly monitor their blood glucose and HbA1c levels, especially if they are in high-risk categories (older age, high BMI).

2. **Lifestyle Changes**: Maintaining a healthy BMI through diet and exercise can significantly reduce the risk of developing diabetes.

3. **Early Intervention**: Early detection and intervention can help manage diabetes effectively and prevent complications.

4. **Model Deployment**: This model could be integrated into healthcare systems to help identify high-risk individuals who may benefit from preventive interventions.

5. **Further Research**: Future work could explore more complex models or incorporate additional features such as dietary habits, physical activity levels, and family history for even more accurate predictions.

## Future Work

1. **Feature Engineering**: Explore more complex feature interactions and transformations
2. **Advanced Models**: Implement more sophisticated models like gradient boosting or neural networks
3. **Hyperparameter Tuning**: Optimize model parameters for better performance
4. **External Validation**: Validate the model on external datasets to ensure generalizability
5. **Web Application**: Develop a web interface for easy use by healthcare professionals


