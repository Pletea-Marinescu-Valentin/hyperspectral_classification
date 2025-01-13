from sklearn.decomposition import PCA

def apply_pca(X, n_components=30):
    """
    Apply PCA to reduce the dimensionality of the dataset.
    :param X: Input dataset (numpy array or pandas DataFrame).
    :param n_components: Number of principal components to retain.
    :return: Transformed dataset after PCA.
    """
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA completed. Explained variance: {explained_variance:.2f}%")
    return X_reduced
