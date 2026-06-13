from pathlib import Path
import random
import os
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Base Directories
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
EXTERNAL_DIR = PROJECT_ROOT / "external"

# Data Subdirectories
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# Artifact Subdirectories
MODELS_DIR = ARTIFACTS_DIR / "models"

# Result Subdirectories
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
TIMINGS_DIR = RESULTS_DIR / "benchmarks"

def ensure_dir(path):
    """Helper to ensure a directory exists before saving to it."""
    path.mkdir(parents=True, exist_ok=True)
    return path