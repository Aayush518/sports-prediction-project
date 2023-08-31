import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from model.data_preprocessing import preprocess_data
from model.model_training import train_model, evaluate_model

st.title("Sports Game Outcome Prediction")
st.sidebar.title("Settings")

# Initialize X_train as an empty DataFrame
X_train = pd.DataFrame()

uploaded_file = st.sidebar.file_uploader("Upload your own CSV:", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    st.write("New model trained with uploaded data!")

# Check if X_train is not empty before proceeding
if not X_train.empty:
    feature_values = {}
    for feature in X_train.columns:
        feature_values[feature] = st.sidebar.number_input(f"Enter {feature} value:")

    user_input_df = pd.DataFrame([feature_values])

    if st.sidebar.button("Predict"):
        prediction = model.predict(user_input_df)
        st.write(f"The predicted outcome is: {prediction[0]}")

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    st.sidebar.write(f"Training Accuracy: {train_accuracy:.2f}")

    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    st.sidebar.write(f"Test Accuracy: {test_accuracy:.2f}")

    corr_matrix = X_train.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    st.pyplot()

    importance = model.feature_importances_
    feature_names = X_train.columns
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    st.pyplot()
else:
    st.write("Please upload a CSV file to proceed.")
