from pathlib import Path
import random
import os
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
EXTERNAL_DIR = ROOT_DIR / "external"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

def ensure_dir(path):
    """Helper to ensure a directory exists before saving to it."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def set_global_seed(seed: int = 1):
    """
    Sets a deterministic seed across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False