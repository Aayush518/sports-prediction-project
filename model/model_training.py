from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.

    Args:
        X_train (DataFrame): Features of the training data.
        y_train (Series): Target labels of the training data.

    Returns:
        RandomForestClassifier: Trained model.
    """
    # Initialize a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model using the training data
    model.fit(X_train, y_train)

    return model

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
