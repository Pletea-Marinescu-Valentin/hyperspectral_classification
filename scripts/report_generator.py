import pandas as pd
from tabulate import tabulate

def generate_summary_report(comparison_csv, output_path="results/summary_report.md"):
    """
    Generate a markdown summary report of the model comparison.
    """
    comparison_df = pd.read_csv(comparison_csv)
    with open(output_path, "w") as f:
        f.write("# Model Comparison Report\n\n")
        f.write("### Evaluation Metrics\n\n")
        f.write(tabulate(comparison_df, headers="keys", tablefmt="github"))
    print(f"Summary report saved to {output_path}")
