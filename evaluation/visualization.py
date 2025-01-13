import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_comparison(rf_metrics, nn_metrics):
    """
    Plot comparison of metrics between Random Forest and Neural Network.

    Args:
        rf_metrics (dict): Metrics for Random Forest.
        nn_metrics (dict): Metrics for Neural Network.
    """
    print("Plotting comparison of metrics...")
    
    # Data for the plot
    models = ["Random Forest", "Neural Network"]
    accuracy = [rf_metrics["Accuracy"], nn_metrics["Accuracy"]]
    precision = [rf_metrics["Precision"], nn_metrics["Precision"]]
    recall = [rf_metrics["Recall"], nn_metrics["Recall"]]
    f1_score = [rf_metrics["F1 Score"], nn_metrics["F1 Score"]]
    
    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracy, color=["blue", "orange"])
    plt.title("Comparison of Accuracy")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot Precision
    plt.figure(figsize=(8, 5))
    plt.bar(models, precision, color=["green", "purple"])
    plt.title("Comparison of Precision")
    plt.ylabel("Precision")
    plt.show()

    # Plot Recall
    plt.figure(figsize=(8, 5))
    plt.bar(models, recall, color=["cyan", "magenta"])
    plt.title("Comparison of Recall")
    plt.ylabel("Recall")
    plt.show()

    # Plot F1 Score
    plt.figure(figsize=(8, 5))
    plt.bar(models, f1_score, color=["red", "yellow"])
    plt.title("Comparison of F1 Score")
    plt.ylabel("F1 Score")
    plt.show()

    print("Comparison plots generated successfully.")

