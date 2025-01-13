import os
from workflow import run_workflow

def batch_process():
    """
    Run the supervised classification pipeline on the dataset.
    """
    print("Starting batch processing...")
    run_workflow()
    print("Batch processing completed.")
