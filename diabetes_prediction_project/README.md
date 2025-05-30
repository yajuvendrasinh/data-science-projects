# Diabetes Prediction Model

## Project Overview

This project aims to build a machine learning model to predict diabetes based on various health metrics. Early detection of diabetes is crucial for effective management and prevention of complications. The model uses a dataset with 100,000 records containing various health parameters to predict whether a person has diabetes.

## Dataset Information

The dataset used in this project contains 100,000 records with the following features:

- **gender**: Gender of the patient (Male, Female)
- **age**: Age of the patient in years
- **hypertension**: Whether the patient has hypertension (0: No, 1: Yes)
- **heart_disease**: Whether the patient has heart disease (0: No, 1: Yes)
- **smoking_history**: Smoking history of the patient (current, former, never, etc.)
- **bmi**: Body Mass Index, a measure of body fat based on height and weight
- **HbA1c_level**: Hemoglobin A1c level, a measure of average blood sugar over the past 2-3 months
- **blood_glucose_level**: Current blood glucose level
- **diabetes**: Target variable indicating whether the patient has diabetes (0: No, 1: Yes)

*Note: The original dataset contained 'Other' for gender and 'No Info' for smoking history. These were handled during preprocessing (see notebook for details).* 

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
│   ├── diabetes_prediction_model.pkl    # Trained (tuned) model (generated after running notebook)
│   ├── scaler.pkl                       # Feature scaler (generated after running notebook)
│   ├── predict.py                       # Script for making predictions (programmatic)
│   └── predict_cli.py                   # Script for making predictions (command-line)
│
├── images/                              # Visualizations generated during analysis
│
└── README.md                            # This file
```

## Methods Used

### Data Preprocessing

1.  **Data Cleaning**: Checked for missing values and handled inconsistent categorical entries ('Other' gender, 'No Info' smoking history).
2.  **Feature Engineering**: Converted categorical variables ('gender', 'smoking_history') to numerical using one-hot encoding.
3.  **Feature Scaling**: Standardized numerical features using `StandardScaler` to have zero mean and unit variance.
4.  **Train-Test Split**: Split the data into training (80%) and testing (20%) sets, using stratification to maintain the proportion of the target variable.

### Exploratory Data Analysis (EDA)

1.  **Univariate Analysis**: Examined the distribution of individual features (age, BMI, HbA1c, blood glucose) and the target variable.
2.  **Bivariate Analysis**: Explored relationships between features and the target variable using visualizations (histograms, count plots).
3.  **Correlation Analysis**: Calculated and visualized the correlation matrix for numerical features.

### Machine Learning Models

Three different baseline models were implemented and compared:

1.  **Logistic Regression**: A simple linear model for binary classification.
2.  **Decision Tree**: A non-linear model that makes decisions based on feature thresholds.
3.  **Random Forest (Baseline)**: An ensemble model that combines multiple decision trees, used as a baseline before tuning.

### Hyperparameter Tuning

1.  **Method**: Used `RandomizedSearchCV` to efficiently search for the best hyperparameters for the Random Forest model.
2.  **Parameters Tuned**: Explored combinations of `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `bootstrap`.
3.  **Result**: Identified the optimal hyperparameter set that improved model performance compared to the baseline Random Forest.
4.  **Accuracy Comparison** (See notebook sections 7.3 and 7.5 for exact values):
    *   Baseline Random Forest Accuracy (`accuracy_rf`): ~0.9705
    *   After Hyperparameter Tuning (RandomizedSearchCV) (`accuracy_rf_tuned`): ~0.9710

### Model Evaluation

Models (baseline and tuned) were evaluated using the following metrics:

1.  **Accuracy**: Proportion of correct predictions.
2.  **Precision**: Proportion of true positive predictions among all positive predictions.
3.  **Recall**: Proportion of true positive predictions among all actual positives (sensitivity).
4.  **F1 Score**: Harmonic mean of precision and recall.
5.  **AUC-ROC**: Area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes.
6.  **Confusion Matrix**: Visualized the counts of true positives, true negatives, false positives, and false negatives.
7.  **Classification Report**: Provided a detailed breakdown of precision, recall, and F1-score for each class.

## Key Findings

1.  **Best Model**: The **Tuned Random Forest** model demonstrated the best overall performance after hyperparameter optimization using `RandomizedSearchCV`.
2.  **Impact of Tuning**: Hyperparameter tuning led to improvements in the Random Forest model's performance, particularly in balancing precision and recall for the minority class (diabetes=1), although the overall accuracy improvement might be marginal.
3.  **Key Predictors**: The most important features for predicting diabetes, identified by the tuned Random Forest model, remain consistent:
    *   Blood glucose level
    *   HbA1c level
    *   Age
    *   BMI
4.  **Medical Relevance**: The findings align with established medical knowledge, confirming the significance of blood sugar metrics, age, and BMI as risk factors for diabetes.

## How to Use This Project

### Prerequisites

- Python 3.6 or higher
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/yourusername/diabetes-prediction-project.git
    cd diabetes-prediction-project
    ```

2.  Install the required packages:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    ```

### Running the Analysis

1.  Open and run the Jupyter notebook:
    ```bash
    jupyter notebook notebooks/diabetes_prediction_model.ipynb
    ```

2.  The notebook will:
    *   Load and explore the dataset.
    *   Perform data preprocessing and EDA.
    *   Build and evaluate baseline machine learning models.
    *   Perform hyperparameter tuning for the Random Forest model.
    *   Evaluate the tuned model and compare results.
    *   Save the best (tuned) model and the scaler to the `src/` directory.

### Making Predictions

You can use the trained (tuned) model to make predictions on new data using the provided scripts:

**1. Programmatic Prediction:**

```python
# Ensure you are in the project's root directory
# or adjust paths accordingly
import sys
sys.path.append("src") # Add src directory to path if needed
from predict import predict_diabetes

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
# Note: Ensure the model files (diabetes_prediction_model.pkl, scaler.pkl) 
# are in the same directory as predict.py or provide the correct path.
prediction, probability = predict_diabetes(patient_data)

if prediction is not None:
    print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
    print(f"Probability of diabetes: {probability:.4f}")
```

**2. Command-Line Prediction:**

Navigate to the `src` directory in your terminal and run:

```bash
cd src
python predict_cli.py --gender Male --age 45 --hypertension 0 --heart_disease 0 --smoking_history never --bmi 28.5 --hba1c_level 6.8 --blood_glucose_level 140
```

Replace the example values with the actual patient data.

## Conclusions and Recommendations

### Conclusions

This project successfully developed and optimized a machine learning model for diabetes prediction. The baseline Random Forest model showed strong performance, which was further enhanced through hyperparameter tuning using `RandomizedSearchCV`. The tuned model provides a reliable tool for identifying potential diabetes cases based on key health indicators like blood glucose level, HbA1c level, age, and BMI.

### Recommendations

1.  **Use Tuned Model**: For practical applications, the tuned Random Forest model (`src/diabetes_prediction_model.pkl`) should be used due to its optimized performance.
2.  **Regular Monitoring**: Emphasize the importance of regular monitoring of blood glucose and HbA1c levels, particularly for individuals with risk factors like older age or high BMI.
3.  **Lifestyle Interventions**: Promote healthy lifestyle choices, such as maintaining a healthy BMI through diet and exercise, as effective preventive measures.
4.  **Clinical Integration**: Consider integrating the model into clinical decision support systems to aid healthcare professionals in identifying high-risk individuals for early intervention.

## Future Work

1.  **Advanced Tuning**: Explore more advanced hyperparameter tuning techniques like Bayesian Optimization or `GridSearchCV` (though potentially more time-consuming).
2.  **Alternative Models**: Experiment with other algorithms like Gradient Boosting (XGBoost, LightGBM) or Neural Networks.
3.  **Feature Engineering**: Investigate creating new features from existing ones to potentially improve model accuracy.
4.  **Explainability**: Implement model explainability techniques (e.g., SHAP, LIME) to better understand the model's predictions.
5.  **Deployment**: Package the model into a web application or API for easier access and use.
