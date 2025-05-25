import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary statistics:")
summary_stats = df.describe().T
print(summary_stats)

# Save summary statistics to CSV
summary_stats.to_csv('../output/summary_statistics.csv')

# Explore categorical variables
print("\nCategorical variables distribution:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col} distribution:")
    print(df[col].value_counts())
    
    # Create and save bar plots for categorical variables
    plt.figure(figsize=(10, 6))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'../visualizations/categorical_{col}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Explore numerical variables
print("\nNumerical variables distribution:")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if col != 'student_id':  # Skip student_id
        print(f"\n{col} statistics:")
        print(df[col].describe())
        
        # Create and save histograms for numerical variables
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'../visualizations/numerical_{col}.png', dpi=300, bbox_inches='tight')
        plt.close()

print("\nData exploration completed and visualizations saved.")
