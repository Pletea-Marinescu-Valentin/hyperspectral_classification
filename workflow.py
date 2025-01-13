import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from keras.api.models import *
from keras.api.layers import *
from sklearn.model_selection import train_test_split
from preprocessing.data_loading import load_hyperspectral_data
from preprocessing.data_cleaning import clean_data
from preprocessing.data_scaling import scale_data
from preprocessing.pca import apply_pca
from evaluation.metrics import calculate_metrics, generate_comparison_table
from evaluation.visualization import plot_comparison

# Define paths for data
DATA_PATH = "data/Salinas.mat"
GROUND_TRUTH_PATH = "data/Salinas_gt.mat"
RESULTS_PATH = "results/"

# Ensure results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    return metrics, y_pred

def train_neural_network(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Neural Network model.
    """
    num_classes = len(np.unique(y_train))
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=2)
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    metrics = calculate_metrics(y_test, y_pred)
    return metrics, y_pred

def run_workflow():
    """
    Main workflow for supervised classification on hyperspectral data.
    """
    # Load and preprocess data
    print("Loading data...")
    data, ground_truth = load_hyperspectral_data(DATA_PATH, GROUND_TRUTH_PATH)
    X, y = clean_data(data, ground_truth)
    X_scaled = scale_data(X)
    X_pca = apply_pca(X_scaled, n_components=30)

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train models and collect results
    print("Training Random Forest...")
    rf_metrics, rf_predictions = train_random_forest(X_train, y_train, X_test, y_test)

    print("Training Neural Network...")
    nn_metrics, nn_predictions = train_neural_network(X_train, y_train, X_test, y_test)

    # Generate comparison table
    comparison_table = generate_comparison_table(
        ("Random Forest", rf_metrics),
        ("Neural Network", nn_metrics),
    )
    print("\nModel Comparison:")
    print(comparison_table)

    # Save results
    comparison_table.to_csv(os.path.join(RESULTS_PATH, "comparison_results.csv"), index=False)
    print("Results saved to:", RESULTS_PATH)

    # Visualize comparison
    plot_comparison(rf_metrics, nn_metrics)
