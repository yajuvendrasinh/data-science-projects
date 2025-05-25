import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from pathlib import Path

# Set the style for visualizations
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')

# Create directories for output
Path('../visualizations').mkdir(exist_ok=True)
Path('../output').mkdir(exist_ok=True)

# Load the preprocessed dataset
print("Loading preprocessed dataset...")
df = pd.read_csv('../output/encoded_data_for_modeling.csv')

# Define target variable for classification
print("\nPreparing data for classification analysis...")
# We'll predict whether a student will score above or below the median exam score
median_score = df['exam_score'].median()
df['high_performer'] = (df['exam_score'] >= median_score).astype(int)
print(f"Median exam score: {median_score}")
print(f"High performers (1): {df['high_performer'].sum()}")
print(f"Low performers (0): {len(df) - df['high_performer'].sum()}")

# Define features for classification
# Exclude student_id, exam_score, and the target variable from features
features = [col for col in df.columns if col not in ['student_id', 'exam_score', 'high_performer']]

# Split the data into training and testing sets
X = df[features]
y = df['high_performer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Create a pipeline with preprocessing and model
print("\nCreating classification pipeline...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Use a smaller subset for faster training if dataset is very large
print("\nTraining RandomForest classifier with optimized parameters...")
# Use reasonable default parameters instead of extensive grid search
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy:.4f}")

# Classification report
class_report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save classification results
results = {
    'accuracy': accuracy,
    'precision_0': class_report['0']['precision'],
    'recall_0': class_report['0']['recall'],
    'f1_score_0': class_report['0']['f1-score'],
    'precision_1': class_report['1']['precision'],
    'recall_1': class_report['1']['recall'],
    'f1_score_1': class_report['1']['f1-score']
}
pd.DataFrame([results]).to_csv('../output/classification_results.csv', index=False)

# Get feature importance
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'Feature': [features[i] for i in indices],
        'Importance': importances[indices]
    })
    
    # Save feature importance
    feature_importance.to_csv('../output/classification_feature_importance.csv', index=False)
    
    # Visualize top 15 feature importance
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 Features for Predicting High Performance', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig('../visualizations/classification_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.savefig('../visualizations/classification_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('../visualizations/classification_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Analyze key factors for high performers vs. low performers
print("\nAnalyzing key factors for high performers vs. low performers...")
df_with_target = df.copy()
high_performers = df_with_target[df_with_target['high_performer'] == 1]
low_performers = df_with_target[df_with_target['high_performer'] == 0]

# Compare key metrics
key_metrics = ['study_hours_per_day', 'sleep_hours', 'social_media_hours', 'netflix_hours', 
               'attendance_percentage', 'wellness_score', 'academic_engagement', 
               'stress_level', 'time_management_score']

comparison = pd.DataFrame({
    'Metric': key_metrics,
    'High Performers': [high_performers[metric].mean() for metric in key_metrics],
    'Low Performers': [low_performers[metric].mean() for metric in key_metrics],
    'Difference': [high_performers[metric].mean() - low_performers[metric].mean() for metric in key_metrics]
})

# Save comparison
comparison.to_csv('../output/high_vs_low_performers_comparison.csv', index=False)

# Visualize key differences
plt.figure(figsize=(14, 10))
comparison['Difference %'] = (comparison['Difference'] / comparison['Low Performers']) * 100
comparison = comparison.sort_values('Difference %', ascending=False)

sns.barplot(x='Difference %', y='Metric', data=comparison)
plt.title('Percentage Difference in Key Metrics: High vs. Low Performers', fontsize=16)
plt.xlabel('Percentage Difference (%)', fontsize=14)
plt.ylabel('Metric', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('../visualizations/high_vs_low_performers_difference.png', dpi=300, bbox_inches='tight')
plt.close()

# Create comparison boxplots for key metrics
for metric in key_metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='high_performer', y=metric, data=df_with_target)
    plt.title(f'{metric} by Performance Group', fontsize=16)
    plt.xlabel('High Performer (1) vs. Low Performer (0)', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../visualizations/comparison_boxplot_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nClassification analysis completed and results saved.")
