# Hyperspectral Image Classification

## 🚀 Project Overview

This project focuses on supervised classification of hyperspectral images using advanced machine learning models, including **Random Forest** and **Neural Networks**. The dataset used is the Salinas hyperspectral scene, widely recognized for remote sensing and environmental monitoring tasks.

## 🏗️ Project Structure

``` plaintext
📂 hyperspectral_classification

├── 📁 data/                     # Dataset folder
│   └── Salinas.mat              # Hyperspectral image data
│   └── Salinas_gt.mat           # Ground truth labels
├── 📁 preprocessing/            # Preprocessing scripts
│   ├── data_cleaning.py         # Data normalization and filtering
│   ├── data_scaling.py          # Data scaling utilities
│   ├── data_loading.py          # Data loading
│   ├── pca.py                   # PCA for dimensionality reduction
├── 📁 models/                   # Model implementations
│   ├── random_forest.py         # Random Forest implementation
│   ├── neural_network.py        # Neural Network implementation
├── 📁 evaluation/               # Evaluation and visualization scripts
│   ├── metrics.py               # Metrics calculation
│   ├── visualization.py         # Graphical comparison of results
├── 📁 scripts/                  # Workflow orchestration and reporting
│   ├── batch_processing.py      # Handles end-to-end processing
│   ├── report_generator.py      # Generates detailed reports
├── 📁 results/                  # Output folder for metrics and visualizations
├── workflow.py                  # Workflow orchestration
├── main.py                      # Main entry point for the project
├── requirements.txt             # Required libraries
└── README.md                    # Project documentation
```

## 📥 Setup Instructions

Prerequisites

Ensure you have **Python 3.8+** installed.

Install Dependencies

Install required Python libraries using:
pip install -r requirements.txt

Run the Project

Open the project folder in Visual Studio Code.
Run the project from the terminal:
python -m main

## 📊 Features
- Data Preprocessing:
  - Normalizes hyperspectral data.
  - Filters valid labels.
  - Reduces dimensionality using PCA.
- Models:
  - Random Forest: Efficient for handling structured data.
  - Neural Networks: Capable of learning complex patterns in hyperspectral data.
- Evaluation:
  - Calculates metrics: Accuracy, Precision, Recall, and F1 Score.
  - Generates a detailed comparison table.
- Visualization:
  - Graphical comparison of model performance using bar charts.

## 📝 Output
Example Comparison Report
|    | Model          |   Accuracy |   Precision |   Recall |   F1 Score |
|----|----------------|------------|-------------|----------|------------|
|  0 | Random Forest  |   0.928875 |    0.964312 | 0.957836 |   0.959919 |
|  1 | Neural Network |   0.937065 |    0.971761 | 0.971187 |   0.97135  |

Outputs saved in the results/ folder:
- Comparison Table: comparison_table.csv
- Metrics Report: metrics_report.txt
- Visualization Plots: comparison_plots.png
## ⚙️ Customization
Modify the following parameters for your specific use case:

- PCA Components: Adjust the number of principal components in pca.py:

pca = PCA(n_components=30, random_state=42)

- Model Hyperparameters: Edit configurations in respective model scripts:
  - Random Forest: Adjust n_estimators, max_depth, etc., in random_forest.py.
  - Neural Networks: Modify layers, activation functions, and epochs in neural_network.py.
    
## 🌟 Key Highlights
- Hyperspectral Data Analysis: Leveraging hyperspectral image data for precise classification.
- Advanced Machine Learning: Employing state-of-the-art supervised learning models.
- Customizable and Modular: Designed for scalability and adaptability across datasets.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Ensure all dependencies are installed using `pip install -r requirements.txt`.
   - Verify Python version is 3.8 or higher.

2. **Dataset Errors**:
   - Confirm that `Salinas.mat` and `Salinas_gt.mat` are present in the `data/` folder.
   - Check file permissions if loading fails.

3. **Visualization Issues**:
   - Ensure `matplotlib` and `seaborn` are installed.
   - Verify that the `results/` folder is writable.

## Usage Examples

### Running the Project

1. Preprocess the data:
   ```bash
   python preprocessing/data_cleaning.py
   ```

2. Train models:
   ```bash
   python models/random_forest.py
   ```

3. Generate reports:
   ```bash
   python scripts/report_generator.py
   ```
