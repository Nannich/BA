import os
import numpy as np
import pandas as pd
import torch
from efficient_kan import KAN


def load_data(data_path):
    counts = pd.read_csv(os.path.join(data_path, "counts.csv"), index_col=0)
    pseudotime = pd.read_csv(os.path.join(data_path, "pseudotime.csv"), index_col=0)
    weights = pd.read_csv(os.path.join(data_path, "weights.csv"), index_col=0)
    return counts, pseudotime, weights


def get_lineage_assignment(weights):
    # Assign cells to lineages based on their weight
    # A cell can be part of two lineages at the same time, e.g. if it's weight is [1, 1]
    sensitivity = 0.1
    max_weights = np.max(weights.values, axis=1, keepdims=True)
    lineage_assignment = np.abs(max_weights - weights.values) < sensitivity

    return lineage_assignment


def get_plotting_data(pseudotime, weights, model, gene_idx, target_lineage, n_points=200, is_single_gene=False):
    # Returns the prediction curve of the model and the corresponding input matric for the target lineage 

    n_lineages = weights.shape[1]
    lineage_assignment = get_lineage_assignment(weights)
    pt_values = pseudotime.values
    pt_min = pt_values.min(axis=0, keepdims=True)
    pt_max = pt_values.max(axis=0, keepdims=True)

    # Get the mask for the current lineage, i.e. which cells are on this lineage
    mask = lineage_assignment[:,target_lineage]
    pt_active = pseudotime.values[mask, target_lineage]

    # Sort data along the active pseudotime for fitting
    sort_idx = np.argsort(pt_active)
    pt_active_sorted = pt_active[sort_idx]
    weights_sorted = weights.values[mask][sort_idx]

    # Pseudotime where lineage ends
    pt_active_max = pseudotime.values[:, target_lineage].max()

    # Create smooth pseudotime and weight matrices as input for plotting
    # Pseudotime for the currently active lineage is evenly spread between 0 and pt_max
    # For the other lineages and the weights a function is fitted on their values for the active lineage
    #
    # Example input_matrix for Lineage 1 (Active), assuming 2 lineages:
    # [ pt_lin1, pt_lin2, weight_lin1, weight_lin2 ]
    # 
    # - pt_lin1: Linearly spaced grid (np.linspace)
    # - pt_lin2: Predicted via polyfit(pt_active, pt_inactive)
    # - Weights: Predicted via polyfit(pt_active, weight_k)
    #
    # For example:
    # [
    #   [0.00,  0.00,  1.00,  1.00],
    #   [0.10,  0.05,  0.77,  1.00],
    #   ...
    #   [0.80,  0.39,  0.16,  1.00],
    #   [1.00,  0.39,  1.00,  0.02]

    pt_grid = np.linspace(0, pt_active_max, n_points)
    pt_input = np.zeros((n_points, n_lineages))
    pt_input[:, target_lineage] = pt_grid

    for l in range(n_lineages):
        if l != target_lineage:
            pt_inactive = pseudotime.values[mask, l]
            pt_inactive_sorted = pt_inactive[sort_idx]
            pt_fitted = np.polyfit(pt_active_sorted, pt_inactive_sorted, deg=3)
            f_inactive_pt = np.poly1d(pt_fitted)
            pt_input[:, l] = np.clip(f_inactive_pt(pt_grid), 0.0, None)

    # Scale Pseudotime to [0,1] for the KAN grid
    pt_input_scaled = (pt_input - pt_min) / (pt_max - pt_min + 1e-8)
    
    w_input = np.zeros((n_points, n_lineages))
    for k in range(n_lineages):
        w = weights_sorted[:,k]
        w_fitted = np.polyfit(pt_active_sorted, w, deg=3)
        f_w = np.poly1d(w_fitted)
        w_input[:, k] = np.clip(f_w(pt_grid), 0.0, 1)

    # Run the model on the plotting data
    input_matrix = np.hstack((pt_input_scaled, w_input))
    X_tensor = torch.tensor(input_matrix, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        mu, theta, pi = model(X_tensor)
 
    if is_single_gene:
        y_line = mu[:, 0].numpy()
    else: 
        y_line = mu[:, gene_idx].numpy()
    
    # Because of the exp link function the model predicts the log count
    y_line = np.exp(y_line)                         
    y_line = np.log1p(y_line)

    return pt_grid, pt_input_scaled, y_line
