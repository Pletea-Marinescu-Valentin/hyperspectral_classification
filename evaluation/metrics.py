import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred, average='macro'):
    """
    Calculate and return classification metrics.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        average (str): Averaging method for multi-class metrics ('macro', 'micro', 'weighted').

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    return metrics


def display_metrics(y_true, y_pred):
    """
    Print classification metrics and return the result as a dictionary.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: Classification metrics.
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    metrics = calculate_metrics(y_true, y_pred)
    print("Summary Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics


def generate_comparison_table(*model_metrics):
    """
    Generate a comparison table for multiple models.

    Args:
        model_metrics (tuple): Tuples of model name and metrics dictionary.

    Returns:
        pd.DataFrame: Comparison table as a pandas DataFrame.
    """
    print("Generating comparison table...")
    data = []
    for model_name, metrics in model_metrics:
        row = {
            "Model": model_name,
            "Accuracy": metrics.get("Accuracy", 0),
            "Precision": metrics.get("Precision", 0),
            "Recall": metrics.get("Recall", 0),
            "F1 Score": metrics.get("F1 Score", 0),
        }
        data.append(row)

    comparison_table = pd.DataFrame(data)
    print("Comparison table generated:")
    print(comparison_table)
    return comparison_table

