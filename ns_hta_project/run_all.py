"""
Master runner for the NS-HTA project.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from run_experiments import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
