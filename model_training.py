# Model Training for Customer Churn Prediction

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the customer churn dataset
# The dataset contains customer information, including demographic features, service details, and churn status.
data = pd.read_csv('customer_data.csv')

# Encoding categorical variables
# Label encoding is used for categorical variables, converting text values to numeric ones
# This is necessary for machine learning algorithms that require numerical input.
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                       'PaymentMethod', 'Churn']

for col in categorical_columns:
    # Applying label encoding for each categorical feature
    data[col] = label_encoder.fit_transform(data[col])

# Splitting data into features (X) and target (y)
# The features (X) include all columns except for 'customerID' and 'Churn' (the target).
X = data.drop(['customerID', 'Churn'], axis=1)  # Features (predictors)
y = data['Churn']  # Target variable (Churn status: 1 for churn, 0 for not churn)

# Split the data into training and testing sets
# We will use 80% of the data for training and 20% for testing to evaluate the model's performance.
# This helps ensure that the model is not overfitting and generalizes well on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
# Random Forest is a versatile machine learning algorithm, combining multiple decision trees to improve prediction accuracy.
# In this case, we'll use RandomForestClassifier from sklearn.
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting the churn status for the test set
# After training the model, we will predict the churn status on the test data (X_test).
y_pred = model.predict(X_test)

# Evaluating the model performance
# The performance of the model is evaluated using the following metrics:
# - Accuracy: Proportion of correctly predicted instances
# - Precision: Proportion of positive predictions that are actually positive
# - Recall: Proportion of actual positives that are correctly identified
# - F1-Score: The harmonic mean of precision and recall, providing a balance between the two.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print out the model performance metrics
print(f"Model Performance on Test Data:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Save the trained model to a file using joblib
# After training the model, it is important to save the model to avoid retraining every time.
# The trained model can later be loaded for predictions without retraining.
joblib.dump(model, 'churn_prediction_model.pkl')

# Save the label encoder as well to decode predictions later
# The label encoder is used to convert categorical values into numerical values during preprocessing.
# We save the encoder so it can be used later to decode the predictions back to their original labels.
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and label encoder saved successfully!")
