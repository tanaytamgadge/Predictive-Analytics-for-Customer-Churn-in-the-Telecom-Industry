# Model Selection for Customer Churn Prediction

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the customer churn dataset
data = pd.read_csv('customer_data.csv')

# Display the first few rows of the dataset
data.head()

# Encoding categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                       'PaymentMethod', 'Churn']

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Splitting data into features and target
X = data.drop(['customerID', 'Churn'], axis=1)  # Features
y = data['Churn']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Training and evaluating each model
results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 
                           'Recall': recall, 'F1-Score': f1}

# Display results
results_df = pd.DataFrame(results).T
results_df.sort_values(by='Accuracy', ascending=False)

# Visualizing performance metrics
results_df.plot(kind='bar', figsize=(10,6), title="Model Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.show()
