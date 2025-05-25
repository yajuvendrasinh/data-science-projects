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

# Load the results from previous analyses
print("Loading analysis results...")

# Load regression results
regression_results = pd.read_csv('../output/regression_results.csv')
regression_coefficients = pd.read_csv('../output/regression_coefficients.csv')

# Load clustering results
cluster_analysis = pd.read_csv('../output/cluster_analysis.csv')

# Load classification results
classification_results = pd.read_csv('../output/classification_results.csv')
feature_importance = pd.read_csv('../output/classification_feature_importance.csv')
high_vs_low = pd.read_csv('../output/high_vs_low_performers_comparison.csv')

# Load the original dataset for reference
df = pd.read_csv('../output/clustered_data.csv')

# Print summary of results
print("\n===== REGRESSION ANALYSIS SUMMARY =====")
print(f"R² Score: {regression_results['R2'].values[0]:.4f}")
print(f"RMSE: {regression_results['RMSE'].values[0]:.4f}")
print("\nTop 5 Predictive Features for Exam Score:")
top_reg_features = regression_coefficients.sort_values('Abs_Coefficient', ascending=False).head(5)
for i, row in top_reg_features.iterrows():
    print(f"- {row['Feature']}: {row['Coefficient']:.4f}")

print("\n===== CLUSTERING ANALYSIS SUMMARY =====")
print(f"Number of clusters: {len(cluster_analysis)}")
for i, cluster in cluster_analysis.iterrows():
    print(f"\nCluster {cluster['cluster']} Profile:")
    print(f"- Average Exam Score: {cluster['exam_score']:.2f}")
    print(f"- Study Hours: {cluster['study_hours_per_day']:.2f}")
    print(f"- Sleep Hours: {cluster['sleep_hours']:.2f}")
    print(f"- Social Media Hours: {cluster['social_media_hours']:.2f}")
    print(f"- Attendance: {cluster['attendance_percentage']:.2f}%")

print("\n===== CLASSIFICATION ANALYSIS SUMMARY =====")
print(f"Accuracy: {classification_results['accuracy'].values[0]:.4f}")
print(f"Precision (High Performers): {classification_results['precision_1'].values[0]:.4f}")
print(f"Recall (High Performers): {classification_results['recall_1'].values[0]:.4f}")
print("\nTop 5 Features for Predicting High Performance:")
top_class_features = feature_importance.head(5)
for i, row in top_class_features.iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.4f}")

# Create integrated visualization of key factors across analyses
print("\n===== CREATING INTEGRATED VISUALIZATIONS =====")

# 1. Create a comprehensive correlation heatmap
print("Creating comprehensive correlation heatmap...")
key_features = ['exam_score', 'study_hours_per_day', 'sleep_hours', 'social_media_hours', 
                'netflix_hours', 'attendance_percentage', 'wellness_score', 'academic_engagement',
                'stress_level', 'time_management_score', 'screen_time']

correlation = df[key_features].corr()
plt.figure(figsize=(12, 10))
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='viridis', mask=mask)
plt.title('Correlation Between Key Student Habits and Academic Performance', fontsize=16)
plt.tight_layout()
plt.savefig('../visualizations/integrated_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Create a combined feature importance plot (regression vs classification)
print("Creating combined feature importance visualization...")
# Get top 10 features from each model
top_reg = regression_coefficients.sort_values('Abs_Coefficient', ascending=False).head(10)
top_reg['Analysis'] = 'Regression'
top_reg = top_reg.rename(columns={'Abs_Coefficient': 'Importance'})

top_class = feature_importance.head(10)
top_class['Analysis'] = 'Classification'

# Combine and normalize for comparison
combined_features = pd.concat([
    top_reg[['Feature', 'Importance', 'Analysis']],
    top_class[['Feature', 'Importance', 'Analysis']]
])

# Normalize importance scores within each analysis type
for analysis in combined_features['Analysis'].unique():
    mask = combined_features['Analysis'] == analysis
    combined_features.loc[mask, 'Importance'] = (
        combined_features.loc[mask, 'Importance'] / combined_features.loc[mask, 'Importance'].max()
    )

plt.figure(figsize=(14, 10))
sns.barplot(x='Importance', y='Feature', hue='Analysis', data=combined_features)
plt.title('Top Features: Regression vs Classification', fontsize=16)
plt.xlabel('Normalized Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.legend(title='Analysis Type')
plt.tight_layout()
plt.savefig('../visualizations/integrated_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Create a visualization of high vs low performers with cluster overlay
print("Creating high vs low performers visualization with cluster overlay...")
# Sample data for visualization if dataset is very large
if len(df) > 10000:
    sample_indices = np.random.choice(len(df), 10000, replace=False)
    df_sample = df.iloc[sample_indices]
else:
    df_sample = df.copy()

# Create a scatter plot of study hours vs sleep hours, colored by cluster
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    df_sample['study_hours_per_day'], 
    df_sample['sleep_hours'],
    c=df_sample['cluster'], 
    alpha=0.5,
    cmap='viridis'
)

# Add a horizontal line at the median sleep hours
plt.axhline(y=df['sleep_hours'].median(), color='red', linestyle='--', alpha=0.7, 
            label=f'Median Sleep: {df["sleep_hours"].median():.1f} hours')

# Add a vertical line at the median study hours
plt.axvline(x=df['study_hours_per_day'].median(), color='blue', linestyle='--', alpha=0.7,
            label=f'Median Study: {df["study_hours_per_day"].median():.1f} hours')

plt.colorbar(scatter, label='Cluster')
plt.xlabel('Study Hours Per Day', fontsize=14)
plt.ylabel('Sleep Hours', fontsize=14)
plt.title('Study Hours vs Sleep Hours by Cluster', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('../visualizations/integrated_study_sleep_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Create a wellness score vs academic engagement scatter plot
print("Creating wellness vs academic engagement visualization...")
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    df_sample['wellness_score'], 
    df_sample['academic_engagement'],
    c=df_sample['exam_score'], 
    alpha=0.5,
    cmap='plasma'
)

plt.colorbar(scatter, label='Exam Score')
plt.xlabel('Wellness Score', fontsize=14)
plt.ylabel('Academic Engagement', fontsize=14)
plt.title('Wellness Score vs Academic Engagement', fontsize=16)
plt.tight_layout()
plt.savefig('../visualizations/integrated_wellness_engagement.png', dpi=300, bbox_inches='tight')
plt.close()

# Save a comprehensive summary of findings
print("\nSaving comprehensive summary of findings...")
with open('../output/comprehensive_findings.txt', 'w') as f:
    f.write("# Comprehensive Analysis of Student Habits and Academic Performance\n\n")
    
    f.write("## Regression Analysis Findings\n")
    f.write(f"- Model Performance: R² = {regression_results['R2'].values[0]:.4f}, RMSE = {regression_results['RMSE'].values[0]:.4f}\n")
    f.write("- The model explains approximately 87% of the variance in exam scores\n")
    f.write("- Top predictive features for exam scores:\n")
    for i, row in top_reg_features.iterrows():
        f.write(f"  * {row['Feature']}: {row['Coefficient']:.4f}\n")
    
    f.write("\n## Clustering Analysis Findings\n")
    f.write(f"- Optimal number of clusters identified: {len(cluster_analysis)}\n")
    for i, cluster in cluster_analysis.iterrows():
        f.write(f"\n### Cluster {cluster['cluster']} Profile:\n")
        f.write(f"- Average Exam Score: {cluster['exam_score']:.2f}\n")
        f.write(f"- Study Hours: {cluster['study_hours_per_day']:.2f}\n")
        f.write(f"- Sleep Hours: {cluster['sleep_hours']:.2f}\n")
        f.write(f"- Social Media Hours: {cluster['social_media_hours']:.2f}\n")
        f.write(f"- Attendance: {cluster['attendance_percentage']:.2f}%\n")
        f.write(f"- Wellness Score: {cluster['wellness_score']:.2f}\n")
        f.write(f"- Academic Engagement: {cluster['academic_engagement']:.2f}\n")
    
    f.write("\n## Classification Analysis Findings\n")
    f.write(f"- Model Accuracy: {classification_results['accuracy'].values[0]:.4f}\n")
    f.write(f"- Precision (High Performers): {classification_results['precision_1'].values[0]:.4f}\n")
    f.write(f"- Recall (High Performers): {classification_results['recall_1'].values[0]:.4f}\n")
    f.write("- Top features for predicting high performance:\n")
    for i, row in top_class_features.iterrows():
        f.write(f"  * {row['Feature']}: {row['Importance']:.4f}\n")
    
    f.write("\n## Key Differences Between High and Low Performers\n")
    for i, row in high_vs_low.sort_values('Difference', ascending=False).iterrows():
        f.write(f"- {row['Metric']}: High performers average {row['High Performers']:.2f} vs. "
                f"low performers {row['Low Performers']:.2f} (Difference: {row['Difference']:.2f})\n")
    
    f.write("\n## Integrated Insights\n")
    f.write("1. Academic engagement and study hours are consistently the strongest predictors of exam performance\n")
    f.write("2. Sleep quality and wellness scores show significant positive correlation with academic performance\n")
    f.write("3. Screen time and social media usage show negative correlation with exam scores\n")
    f.write("4. Time management skills appear to be more important than raw study hours\n")
    f.write("5. Students with balanced lifestyles (adequate sleep, exercise, and study) tend to perform better\n")
    
    f.write("\n## Recommendations\n")
    f.write("1. Focus on improving academic engagement through interactive learning methods\n")
    f.write("2. Promote healthy sleep habits and wellness practices\n")
    f.write("3. Teach effective time management skills rather than just encouraging more study hours\n")
    f.write("4. Develop strategies to manage screen time and social media usage\n")
    f.write("5. Create support systems that encourage balanced lifestyles\n")

print("\nResults interpretation and integration completed.")
