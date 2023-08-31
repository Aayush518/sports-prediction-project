import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from imblearn.over_sampling import SMOTE
import numpy as np

def preprocess_data(data):
    data = data.dropna()

    # Remove outliers using Z-score
    z_scores = stats.zscore(data.select_dtypes(include=['float64', 'int64']))
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]

    # Label Encoding and Feature Scaling
    label_encoder = LabelEncoder()
    data['Outcome'] = label_encoder.fit_transform(data['Outcome'])
    scaler = StandardScaler()
    data[['Feature1', 'Feature2']] = scaler.fit_transform(data[['Feature1', 'Feature2']])

    # Handle imbalanced classes using SMOTE
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
