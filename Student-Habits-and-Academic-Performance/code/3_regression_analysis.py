import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.ticker as mtick
from pathlib import Path

# Set the style for visualizations
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')

# Create directories for output
Path('../visualizations').mkdir(exist_ok=True)
Path('../output').mkdir(exist_ok=True)

# Load the preprocessed dataset
print("Loading preprocessed dataset...")
df = pd.read_csv('../output/encoded_data.csv')

# Define target variable and features for regression
print("\nPreparing data for regression analysis...")
target = 'exam_score'
# Exclude student_id and the target variable from features
features = [col for col in df.columns if col != target and col != 'student_id']

# Split the data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Create and train the linear regression model
print("\nTraining linear regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Cross-validation
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R² score: {cv_scores.mean():.4f}")

# Save model results
results = {
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2,
    'CV_R2_mean': cv_scores.mean(),
    'CV_R2_std': cv_scores.std()
}
pd.DataFrame([results]).to_csv('../output/regression_results.csv', index=False)

# Get feature importance
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

# Save feature importance
coefficients.to_csv('../output/regression_coefficients.csv', index=False)

# Visualize top 15 feature importance
plt.figure(figsize=(12, 10))
top_features = coefficients.head(15)
colors = ['#1f77b4' if c > 0 else '#d62728' for c in top_features['Coefficient']]
sns.barplot(x='Abs_Coefficient', y='Feature', data=top_features, palette=colors)
plt.title('Top 15 Features Importance in Predicting Exam Score', fontsize=16)
plt.xlabel('Absolute Coefficient Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig('../visualizations/regression_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualize actual vs predicted values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Exam Score', fontsize=14)
plt.ylabel('Predicted Exam Score', fontsize=14)
plt.title('Actual vs Predicted Exam Scores', fontsize=16)
plt.tight_layout()
plt.savefig('../visualizations/regression_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualize residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 8))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Residuals', fontsize=16)
plt.tight_layout()
plt.savefig('../visualizations/regression_residuals_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Residuals vs Predicted values
plt.figure(figsize=(10, 8))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Exam Score', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Residuals vs Predicted Values', fontsize=16)
plt.tight_layout()
plt.savefig('../visualizations/regression_residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

# Analyze key relationships
key_features = ['study_hours_per_day', 'sleep_hours', 'wellness_score', 'academic_engagement']
for feature in key_features:
    plt.figure(figsize=(10, 8))
    sns.regplot(x=feature, y=target, data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title(f'Relationship between {feature} and Exam Score', fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Exam Score', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../visualizations/regression_{feature}_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nRegression analysis completed and results saved.")
