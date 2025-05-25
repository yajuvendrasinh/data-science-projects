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
Path('../visualizations').mkdir(exist_ok=True)
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

# 3. Create a screen time category - but keep it numeric for modeling
# Instead of creating a categorical feature, create a numeric representation
df_processed['screen_time_level'] = pd.cut(
    df_processed['screen_time'], 
    bins=[0, 5, 10, 15, 25], 
    labels=[1, 2, 3, 4]  # Numeric labels instead of strings
).astype(int)  # Convert to integer explicitly

# 4. Create a balanced lifestyle indicator as numeric
df_processed['balanced_lifestyle_numeric'] = np.where(
    (df_processed['sleep_hours'] >= 7) & 
    (df_processed['exercise_frequency'] >= 3) & 
    (df_processed['stress_level'] <= 6) &
    (df_processed['study_hours_per_day'] >= 3),
    1, 0  # 1 for Yes, 0 for No
)

# 5. Create academic engagement score
df_processed['academic_engagement'] = (
    (df_processed['attendance_percentage'] / 100) * 0.4 +
    (df_processed['study_hours_per_day'] / 12) * 0.3 +  # Normalize to max 12 hours
    (df_processed['motivation_level'] / 10) * 0.3  # Normalize to max 10
) * 10  # Scale to 0-10

# Save the processed dataframe with new features (before encoding)
print("\nSaving processed dataframe with new features...")
df_processed.to_csv('../output/processed_data_with_features.csv', index=False)

# Prepare for modeling - create a separate dataframe for modeling
print("\nPreparing data for modeling...")
df_model = df_processed.copy()

# Handle categorical variables
categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

# One-hot encode all categorical variables
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=False)

# Double-check no object columns remain
remaining_object_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
print(f"Remaining object columns after encoding: {remaining_object_cols}")

# Verify all columns are numeric
print("\nVerifying all columns are numeric...")
for col in df_encoded.columns:
    if not np.issubdtype(df_encoded[col].dtype, np.number):
        print(f"Column {col} is not numeric: {df_encoded[col].dtype}")
        # Convert any non-numeric columns to numeric if possible
        try:
            df_encoded[col] = pd.to_numeric(df_encoded[col])
            print(f"  - Converted {col} to numeric")
        except:
            print(f"  - Could not convert {col} to numeric, dropping column")
            df_encoded = df_encoded.drop(columns=[col])

# Save the fully encoded dataframe for modeling
print("\nSaving fully encoded dataframe for modeling...")
df_encoded.to_csv('../output/encoded_data_for_modeling.csv', index=False)

print(f"Dataset shape after preprocessing and encoding: {df_encoded.shape}")
print("New features created:")
for feature in ['study_to_entertainment_ratio', 'wellness_score', 'screen_time_level', 
                'balanced_lifestyle_numeric', 'academic_engagement']:
    print(f"- {feature}")

print("\nData preprocessing completed and files saved.")
