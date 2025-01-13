from sklearn.preprocessing import StandardScaler

def scale_data(X):
    """
    Scale the dataset using StandardScaler.
    :param X: Input dataset (numpy array or pandas DataFrame).
    :return: Scaled dataset.
    """
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaling completed.")
    return X_scaled
