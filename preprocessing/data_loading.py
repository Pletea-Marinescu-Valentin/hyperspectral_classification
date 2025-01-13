import os
import numpy as np
from scipy.io import loadmat
import urllib.request

# Define URLs for the dataset and ground truth
data_url = "https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat"
gt_url = "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat"

def download_file(url, path):
    """Download a file from a URL if it doesn't already exist."""
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {path}.")
    else:
        print(f"{path} already exists. Skipping download.")

def load_hyperspectral_data(data_path="Salinas.mat", gt_path="Salinas_gt.mat"):
    """
    Load hyperspectral data and ground truth labels.

    Args:
        data_path (str): Path to the hyperspectral data (.mat file).
        gt_path (str): Path to the ground truth labels (.mat file).

    Returns:
        data (np.ndarray): Hyperspectral image data.
        ground_truth (np.ndarray): Ground truth labels.
    """
    # Download files if not present
    download_file(data_url, data_path)
    download_file(gt_url, gt_path)

    # Load .mat files
    print("Loading data...")
    data = loadmat(data_path)['salinas']
    ground_truth = loadmat(gt_path)['salinas_gt']

    print("Data and ground truth loaded successfully.")
    return data, ground_truth

if __name__ == "__main__":
    data, ground_truth = load_hyperspectral_data()
    print(f"Data shape: {data.shape}, Ground truth shape: {ground_truth.shape}")
