import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    X = data.drop(['HomeTeamPoints', 'AwayTeamPoints', 'Date', 'HomeTeam', 'AwayTeam'], axis=1)
    y = data['HomeTeamPoints'] > data['AwayTeamPoints']  # 1 if Home Team wins, 0 otherwise
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
