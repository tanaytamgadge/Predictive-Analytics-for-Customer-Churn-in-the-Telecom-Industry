# Import necessary libraries for deeper analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# Load the dataset for detailed exploration
df = pd.read_csv('customer_data.csv')

# 1. Summary Statistics: Detailed Statistical Overview
print("Summary Statistics:\n")
summary_stats = df.describe(include='all').transpose()

# Include more advanced statistics like mode, skewness, and kurtosis
summary_stats['mode'] = df.mode().iloc[0]
summary_stats['skewness'] = df.skew()
summary_stats['kurtosis'] = df.kurtosis()

print(summary_stats)

# 2. Data Distribution for Numerical Features: Visualizing Distributions in More Detail
print("\nDistribution of Numerical Features:")

# Select numerical features to analyze
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Visualizing distributions using histograms and KDEs for deeper insight into spread and skewness
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[feature], kde=True, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature} - Skew: {df[feature].skew():.2f}, Kurt: {df[feature].kurt():.2f}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 3. Data Distribution for Categorical Features: Distribution and Count Analysis
print("\nDistribution of Categorical Features:")

categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Visualizing distributions of categorical variables with count plots and percentages
plt.figure(figsize=(20, 15))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(4, 4, i)
    sns.countplot(x=df[feature], palette='Set2')
    plt.title(f'Distribution of {feature} - Count: {df[feature].value_counts().to_dict()}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Correlation Matrix for Numerical Features: Analyzing Correlations in Depth
print("\nCorrelation Matrix for Numerical Features:")

# Calculate correlation matrix
correlation_matrix = df.corr()

# Displaying correlation values in a detailed heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix - Detailed View')
plt.show()

# 5. Pairwise Relationships Between Numerical Features: Advanced Pairwise Analysis
print("\nPairwise Relationships Between Numerical Features:")

# Using pairplot to explore relationships between multiple numerical features
sns.pairplot(df[numerical_features], hue='Churn', diag_kind='kde', plot_kws={'alpha': 0.7})
plt.suptitle("Pairwise Relationships Between Numerical Features", size=16)
plt.show()

# 6. Churn Analysis: In-Depth Churn Analysis and Distribution
print("\nChurn Analysis and Distribution:")

# Count the distribution of churn and display the numbers
churn_counts = df['Churn'].value_counts()
churn_percentage = churn_counts / churn_counts.sum() * 100
print(f"Churn Counts: {churn_counts}\nChurn Percentages: {churn_percentage}")

# Visualizing churn distribution
sns.countplot(x='Churn', data=df, palette='Set1')
plt.title('Churn Distribution - Total Churn Count: {0}'.format(churn_counts.sum()))
plt.show()

# 7. Churn Analysis by Categorical Variables: Breaking Down Churn by Each Category
print("\nChurn Analysis by Categorical Variables:")

# Breaking down churn across all categorical variables
for feature in categorical_features:
    if feature != 'Churn':
        print(f"\nChurn by {feature}:")
        churn_by_feature = df.groupby(feature)['Churn'].value_counts().unstack()
        print(churn_by_feature)

        # Visualizing churn by the current categorical feature
        sns.countplot(x=feature, hue='Churn', data=df, palette='Set1')
        plt.title(f'Churn by {feature} - Total Counts: {churn_by_feature.sum().to_dict()}')
        plt.xticks(rotation=45)
        plt.show()

# 8. Key Insights and Relationships: Advanced Analysis on Key Features
print("\nKey Insights and Feature Relationships:")

# Analyzing the relationship between 'MonthlyCharges' and 'Churn'
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
plt.title('Monthly Charges vs Churn - Skew: {0:.2f}'.format(df['MonthlyCharges'].skew()))
plt.show()

# Analyzing the relationship between 'Tenure' and 'Churn'
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=df, palette='Set2')
plt.title('Tenure vs Churn - Skew: {0:.2f}'.format(df['tenure'].skew()))
plt.show()

# 9. Advanced Feature Engineering: Feature Interactions and Grouped Insights
print("\nAdvanced Feature Engineering and Grouped Insights:")

# Group by 'Contract' and analyze churn rate and average monthly charges
contract_churn = df.groupby('Contract').agg({'Churn': lambda x: (x == 'Yes').mean(), 'MonthlyCharges': 'mean'})
print(f"Churn Rate and Monthly Charges by Contract Type:\n{contract_churn}")

# Visualizing churn rate and monthly charges by contract type
plt.figure(figsize=(10, 6))
sns.barplot(x=contract_churn.index, y=contract_churn['Churn'], palette='Set3')
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate')
plt.show()

# Visualizing average Monthly Charges by Contract Type
plt.figure(figsize=(10, 6))
sns.barplot(x=contract_churn.index, y=contract_churn['MonthlyCharges'], palette='Set3')
plt.title('Average Monthly Charges by Contract Type')
plt.ylabel('Average Monthly Charges')
plt.show()

# 10. Profile Report: For Full Exploratory Data Analysis (Optional)
# A full report that details all aspects of the dataset, including missing values, distributions, etc.
profile = ProfileReport(df, title="Exploratory Data Analysis Report", explorative=True)
profile_path = "eda_report.html"
profile.to_file(profile_path)

# Print message indicating the completion of EDA
print("\nEDA analysis completed successfully! Full report generated at 'eda_report.html'.")
