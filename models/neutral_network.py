import numpy as np
from keras.api.layers import *
from keras.api.optimizers import *
from keras.api.models import *
from keras.api.losses import *
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def train_neural_network(X_train, y_train, X_test, y_test, input_shape, num_classes, epochs=15, batch_size=64):
    """
    Train and evaluate a Neural Network.

    Parameters:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_test (np.array): Test features.
        y_test (np.array): Test labels.
        input_shape (int): Number of input features.
        num_classes (int): Number of unique classes.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        dict: Evaluation metrics (accuracy, classification report, confusion matrix).
    """
    # Define the Neural Network model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    # Evaluate the model
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print("Neural Network Performance:")
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
