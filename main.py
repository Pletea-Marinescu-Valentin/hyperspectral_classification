import logging
import os
from scripts.batch_processing import batch_process
from scripts.report_generator import generate_summary_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Entry point for the supervised classification project.
    """
    logging.info("Starting the main workflow.")

    results_path = "results/comparison_results.csv"
    report_path = "results/summary_report.md"

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    logging.info("Results directory ensured.")

    # Run batch processing
    logging.info("Starting batch processing.")
    batch_process()
    logging.info("Batch processing completed.")

    # Generate summary report
    logging.info("Generating summary report.")
    generate_summary_report(results_path, report_path)
    logging.info("Summary report generated.")

if __name__ == "__main__":
    main()
