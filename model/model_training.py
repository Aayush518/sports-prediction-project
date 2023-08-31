from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model

# ... (rest of the code remains the same)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the accuracy of a trained model.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (DataFrame): Features of the testing data.
        y_test (Series): Target labels of the testing data.

    Returns:
        float: Accuracy of the model on the testing data.
    """
    # Predict the target labels using the model
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
