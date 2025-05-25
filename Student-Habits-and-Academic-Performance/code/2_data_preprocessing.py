import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

# Set the style for visualizations
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')

# Create directories for output
Path('../output').mkdir(exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('../enhanced_student_habits_performance_dataset/enhanced_student_habits_performance_dataset.csv')

# Display basic information
print(f"Dataset shape before preprocessing: {df.shape}")

# Create a copy of the dataframe for preprocessing
df_processed = df.copy()

# Check for outliers in numerical columns
print("\nChecking for outliers in numerical columns...")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('student_id')  # Remove student_id as it's not a feature

# Create boxplots for numerical features to visualize outliers
plt.figure(figsize=(15, 10))
df_processed[numerical_cols].boxplot(figsize=(15, 10))
plt.title('Boxplots of Numerical Features')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../visualizations/boxplots_numerical_features.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature Engineering

# 1. Create a study-to-entertainment ratio
print("\nCreating new features...")
df_processed['study_to_entertainment_ratio'] = df_processed['study_hours_per_day'] / (df_processed['social_media_hours'] + df_processed['netflix_hours'] + 0.1)

# 2. Create a wellness score (combination of sleep, diet, exercise)
# Map diet quality to numerical values
diet_mapping = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
df_processed['diet_quality_numeric'] = df_processed['diet_quality'].map(diet_mapping)

# Create wellness score
df_processed['wellness_score'] = (
    (df_processed['sleep_hours'] / 8) * 0.4 +  # Normalize to recommended 8 hours
    (df_processed['diet_quality_numeric'] / 4) * 0.3 +  # Diet quality (normalized)
    (df_processed['exercise_frequency'] / 7) * 0.3  # Exercise frequency (normalized)
) * 10  # Scale to 0-10

# 3. Create a screen time category
bins = [0, 5, 10, 15, 25]
labels = ['Low', 'Medium', 'High', 'Very High']
df_processed['screen_time_category'] = pd.cut(df_processed['screen_time'], bins=bins, labels=labels)

# 4. Create a balanced lifestyle indicator
df_processed['balanced_lifestyle'] = np.where(
    (df_processed['sleep_hours'] >= 7) & 
    (df_processed['exercise_frequency'] >= 3) & 
    (df_processed['stress_level'] <= 6) &
    (df_processed['study_hours_per_day'] >= 3),
    'Yes', 'No'
)

# 5. Create academic engagement score
df_processed['academic_engagement'] = (
    (df_processed['attendance_percentage'] / 100) * 0.4 +
    (df_processed['study_hours_per_day'] / 12) * 0.3 +  # Normalize to max 12 hours
    (df_processed['motivation_level'] / 10) * 0.3  # Normalize to max 10
) * 10  # Scale to 0-10

# Encode categorical variables
print("\nEncoding categorical variables...")
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical variables for modeling
df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

# Save the processed dataframe
print("\nSaving processed dataframe...")
df_processed.to_csv('../output/processed_data.csv', index=False)
df_encoded.to_csv('../output/encoded_data.csv', index=False)

print(f"Dataset shape after preprocessing and encoding: {df_encoded.shape}")
print("New features created:")
for feature in ['study_to_entertainment_ratio', 'wellness_score', 'screen_time_category', 
                'balanced_lifestyle', 'academic_engagement']:
    print(f"- {feature}")

print("\nData preprocessing completed and files saved.")
