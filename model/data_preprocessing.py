import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(file_path):
    # Load the dataset from the provided file
    data = pd.read_csv(file_path)

    # Drop rows with missing values
    data = data.dropna()

    reference_date = pd.to_datetime('2023-01-01')  # Replace with an appropriate reference date
    data['DaysSinceReference'] = (pd.to_datetime(data['Date']) - reference_date).dt.days

    

    # Feature Engineering: Extract year, month, and day of the week from the date
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    # Encode categorical variables using Label Encoding
    label_encoder = LabelEncoder()
    data['HomeTeam'] = label_encoder.fit_transform(data['HomeTeam'])
    data['AwayTeam'] = label_encoder.fit_transform(data['AwayTeam'])
    data['Venue'] = label_encoder.fit_transform(data['Venue'])
    data['Outcome'] = label_encoder.fit_transform(data['Outcome'])  # Encoding the outcome column

    # Normalize numerical features using StandardScaler
    scaler = StandardScaler()
    numerical_cols = ['HomeTotalWins', 'HomeTotalGames', 'HomeTeamPerformance', 'AwayTeamPerformance',
                      'Feature1', 'Feature2', 'Feature3']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Feature Engineering: Calculate win rate and performance differential
    data['WinRate'] = data['HomeTotalWins'] / data['HomeTotalGames']
    data['PerformanceDiff'] = data['HomeTeamPerformance'] - data['AwayTeamPerformance']

    # One-Hot Encoding for day of the week
    data = pd.get_dummies(data, columns=['DayOfWeek'], prefix='Day')

    # Split the data into features (X) and target (y)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
