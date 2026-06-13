import ctypes
from pathlib import Path

try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import argparse
import torch
import random
import numpy as np
import os
from src.config import DATA_DIR, MODELS_DIR, FIGURES_DIR, ensure_dir
from src.trajectory import run_trajectory
from src.plotting import run_visualization
from src.symbolic import run_extraction
from src.de import run_de
from src.grn import run_grn
from src.preprocessing import run_preprocessing
from src.cluster import run_plot_clusters


def set_global_seed(seed: int = 1):
    """
    Sets a deterministic seed across Python, NumPy, and PyTorch 
    to guarantee reproducible weight initializations and data splits.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Safe for multi-GPU setups
    
    # Force PyTorch operations to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_global_seed(1)

    parser = argparse.ArgumentParser()
    
    # Global routing configuration switches
    parser.add_argument("--dataset", type=str, default="paul",
                        help="Target dataset directory identifier (e.g., 'paul', 'li').")
    parser.add_argument("--experiment_name", type=str, default="baseline",
                        help="Groups model checkpointers and output figures under this specific profile run.")

    # Parse command-specific sub-arguments
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Trajectory Parser
    parser_trajectory = subparsers.add_parser("trajectory", help="Train a trajectory expression prediction model.")
    parser_trajectory.add_argument("--model", type=str, choices=["effkan", "pykan", "mlp", "null"], default="effkan")
    parser_trajectory.add_argument("--gene", type=int, default=None, help="Integer target gene index (None trains all).")

    # 2. Visualize Parser
    parser_vis = subparsers.add_parser("visualize", help="Extract and render expression curve visualizations.")
    parser_vis.add_argument("--gene", type=int, default=None, help="Target gene index to plot within the loaded model layer.")
    parser_vis.add_argument("--checkpoint_name", type=str, default=None)

    # 3. Symbolic Parser
    parser_sym = subparsers.add_parser("symbolic", help="Extract mathematical equations from trained models.")
    parser_sym.add_argument("--gene", default=None, help="Filter equation mapping for a single gene index.")
    
    # 4. DE Parser
    parser_de = subparsers.add_parser("de", help="Compute differential expression tests on trained models.")
    parser_de.add_argument("--sim", type=int, default=1, help="Simulation validation target index tracker.")
    parser_de.add_argument("--lineage", type=int, default=0, help="Target lineage identifier index.")

    # 5. GRN Parser
    parser_grn = subparsers.add_parser("grn", help="Infer Gene Regulatory Network adjacency matrices.")

    # 6. Preprocessing Pre-flight Check Parser
    parser_process = subparsers.add_parser("process", help="Pre-compute data caches, normalizations, and topologies.")

    # 6. Preprocessing Pre-flight Check Parser
    parser_cluster = subparsers.add_parser("cluster")

    args = parser.parse_args()

    run_model_dir = ensure_dir(MODELS_DIR / args.dataset / args.experiment_name)
    run_fig_dir = ensure_dir(FIGURES_DIR / args.dataset / args.experiment_name)
    ensure_dir(DATA_DIR)

    adata, pseudotime, weights = run_preprocessing(dataset_name=args.dataset)    

    if args.command == "trajectory":
        run_trajectory(args, adata, pseudotime, weights, run_model_dir)
        
    elif args.command == "visualize":
        run_visualization(args, adata, pseudotime, weights, run_model_dir, run_fig_dir)
        
    elif args.command == "symbolic":
        run_extraction(args, adata, pseudotime, weights, run_model_dir, run_fig_dir)
        
    elif args.command == "de":
        run_de(args, adata, pseudotime, weights, run_model_dir)
        
    elif args.command == "grn":
        run_grn(args, adata, pseudotime, weights, run_model_dir)
        
    elif args.command == "process":
        run_preprocessing(args.dataset)
        
    elif args.command == "cluster":
        run_plot_clusters(
            adata=adata,
            pseudotime=pseudotime,
            weights=weights,
            run_model_dir=run_model_dir,
            run_fig_dir=run_fig_dir
        )


if __name__ == "__main__":
    main()