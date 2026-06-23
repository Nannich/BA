import numpy as np
import torch
from pathlib import Path
from src.trajectory.zinb_models import build_kan_model

def get_smoothed_expression(dataset, trajectory_model_dir, loss_type="zinb"):
    """
    Loads saved trajectory model checkpoints to generate a smooth and denoised
    cell-by-gene expression matrix.
    """
    n_cells = len(dataset.pseudotime)
    n_genes = len(dataset.gene_names)
    smooth_matrix = np.zeros((n_cells, n_genes), dtype=np.float32)
    
    X_tensor = torch.tensor(dataset.pseudotime, dtype=torch.float32)
    n_lineages = X_tensor.shape[1]

    for idx, gene_name in enumerate(dataset.gene_names):
        checkpoint_path = Path(trajectory_model_dir) / f"kan_{gene_name}_{loss_type}.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint at: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hidden_layers = checkpoint.get("hidden_layers", [1])
        
        model = build_kan_model(n_lineages, 1, hidden_layers)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        
        with torch.no_grad():
            mu, _, _ = model(X_tensor)
            smooth_matrix[:, idx] = mu.squeeze().numpy()
            
    return smooth_matrix


def get_lagged_expression(log_counts_in, log_counts_tgt, pseudotime, lineage_assignment, target_idx, dt=0.0):
    """
    Generates time-lagged training pairs by mapping a cell's predictor expression 
    at the current pseudotime (pt) to the target gene's interpolated expression 
    at a future timepoint (pt + dt).
    """
    n_lineages = lineage_assignment.shape[1]
    X_lagged = []
    Y_lagged = []

    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        if not np.any(mask): 
            continue
            
        lin_in = log_counts_in[mask]
        lin_tgt = log_counts_tgt[mask]
        lin_pt = pseudotime[mask, l]
        
        # Sort cells chronologically along lineage pseudotime
        sort_idx = np.argsort(lin_pt)
        lin_in_sorted = lin_in[sort_idx]
        lin_tgt_sorted = lin_tgt[sort_idx]
        lin_pt_sorted = lin_pt[sort_idx]
        
        X_branch = []
        Y_branch = []
        
        pt_min = lin_pt_sorted[0]
        pt_max = lin_pt_sorted[-1]
        branch_duration = pt_max - pt_min
        branch_dt = branch_duration * dt if branch_duration > 0 else dt
        
        for i in range(len(lin_pt_sorted)):
            t_future = lin_pt_sorted[i] + branch_dt
            if t_future > pt_max:
                break
                
            X_branch.append(lin_in_sorted[i, :])
            y_future = np.interp(t_future, lin_pt_sorted, lin_tgt_sorted[:, target_idx])
            Y_branch.append([y_future])
            
        if len(X_branch) > 0:
            X_lagged.append(np.array(X_branch))
            Y_lagged.append(np.array(Y_branch))

    if not X_lagged:
        return np.empty((0, log_counts_in.shape[1])), np.empty((0, 1))

    return np.vstack(X_lagged), np.vstack(Y_lagged)