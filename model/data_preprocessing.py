import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from imblearn.over_sampling import SMOTE

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()

    # Remove outliers using Z-score
    z_scores = stats.zscore(data.select_dtypes(include=['float64', 'int64']))
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]

    # ... (rest of the code remains the same)

    # Handle imbalanced classes using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test
