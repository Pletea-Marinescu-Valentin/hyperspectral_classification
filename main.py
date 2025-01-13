import os
from scripts.batch_processing import batch_process
from scripts.report_generator import generate_summary_report

def main():
    """
    Entry point for the supervised classification project.
    """
    results_path = "results/comparison_results.csv"
    report_path = "results/summary_report.md"

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Run batch processing
    batch_process()

    # Generate summary report
    generate_summary_report(results_path, report_path)

if __name__ == "__main__":
    main()
