import numpy as np
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

def normalize_data(data):
    """
    Normalize hyperspectral data using standard scaling (mean=0, std=1).

    Args:
        data (np.ndarray): Hyperspectral data array.

    Returns:
        np.ndarray: Normalized data.
    """
    print("Normalizing data...")
    scaler = StandardScaler()
    data_reshaped = data.reshape(-1, data.shape[-1])  # Reshape for normalization
    normalized_data = scaler.fit_transform(data_reshaped)
    print("Normalization complete.")
    return normalized_data

def filter_valid_labels(data, ground_truth):
    """
    Filter data and labels to include only valid (non-zero) labels.

    Args:
        data (np.ndarray): Normalized hyperspectral data.
        ground_truth (np.ndarray): Ground truth labels.

    Returns:
        tuple: Filtered data and labels.
    """
    print("Filtering valid labels...")
    labels = ground_truth.flatten()
    valid_mask = labels > 0  # Consider only valid labels
    data_filtered = data[valid_mask]
    labels_filtered = labels[valid_mask] - 1  # Adjust labels to start from 0
    print(f"Filtered dataset contains {data_filtered.shape[0]} samples.")
    return data_filtered, labels_filtered

def clean_data(data, ground_truth):
    """
    Normalize hyperspectral data and filter valid labels.

    Args:
        data (np.ndarray): Hyperspectral data array.
        ground_truth (np.ndarray): Ground truth labels.

    Returns:
        tuple: Cleaned and normalized data, filtered labels.
    """
    normalized_data = normalize_data(data)
    cleaned_data, cleaned_labels = filter_valid_labels(normalized_data, ground_truth)
    return cleaned_data, cleaned_labels

def parallel_cleaning(data, num_processes=4):
    """
    Cleans data in parallel using multiple processes.

    Args:
        data (list): The data to be cleaned.
        num_processes (int): Number of processes to use.

    Returns:
        list: Cleaned data.
    """
    with Pool(num_processes) as pool:
        cleaned_data = pool.map(clean_data, data)
    return cleaned_data

if __name__ == "__main__":
    # Example usage (requires data and ground_truth from data_loading module)
    from data_loading import load_hyperspectral_data

    data, ground_truth = load_hyperspectral_data()
    normalized_data = normalize_data(data)
    filtered_data, filtered_labels = filter_valid_labels(normalized_data, ground_truth)

    print(f"Filtered data shape: {filtered_data.shape}, Labels shape: {filtered_labels.shape}")
