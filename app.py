import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import the preprocessing function from data_preprocessing.py
from model.data_preprocessing import preprocess_data

# Import the train_model function from model_training.py
from model.model_training import train_model

# Load the dataset
data_file_path = "data\sports_data.csv"  # Replace with the actual path to your data file
X_train, X_test, y_train, y_test = preprocess_data(data_file_path)

# Train the model
model = train_model(X_train, y_train)

# Create the Streamlit app
st.title("Sports Game Outcome Prediction")
st.sidebar.title("Settings")

# Add UI elements for user inputs
feature_values = {}

# Iterate through features and create input fields
for feature in X_train.columns:
    feature_values[feature] = st.sidebar.number_input(f"Enter {feature} value:")

# Create a DataFrame using the entered values
user_input_df = pd.DataFrame([feature_values])

# Make predictions using the trained model
prediction = model.predict(user_input_df)

# Display prediction result
st.write("## Prediction:")
st.write(f"The predicted outcome is: {prediction[0]}")

# Show accuracy score on training data
train_accuracy = accuracy_score(y_train, model.predict(X_train))
st.sidebar.write(f"Training Accuracy: {train_accuracy:.2f}")

# Show accuracy score on test data
test_accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.write(f"Test Accuracy: {test_accuracy:.2f}")

# Data Visualization
st.write("## Data Visualization")

# Plot correlation matrix heatmap
corr_matrix = X_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
st.pyplot()

# Plot feature importance
importance = model.feature_importances_
feature_names = X_train.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=feature_names)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance")
st.pyplot()


