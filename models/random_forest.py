import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=300, max_depth=20, random_state=42):
    """
    Train and evaluate a Random Forest classifier.

    Parameters:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_test (np.array): Test features.
        y_test (np.array): Test labels.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Evaluation metrics (accuracy, classification report, confusion matrix).
    """
    # Initialize Random Forest model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )

    # Train the model
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print("Random Forest Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(conf_matrix)

    # Return metrics
    return {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }
