import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from efficient_kan import KAN

def get_lineage_assignment(weights):
    """
    Assigns the cells to lineages based on their weight.
    A cell can be part of two lineages at the same time, e.g. if it's weight is [1, 1].
    """
    sensitivity = 0.1
    max_weights = np.max(weights, axis=1, keepdims=True)
    lineage_assignment = np.abs(max_weights - weights) < sensitivity

    return lineage_assignment


def scale_pt(pt, pt_min, pt_max):
    return (pt - pt_min) / (pt_max - pt_min + 1e-8)


def sort_by_lineage(pseudotime, weights, lineage):
    """
    Extracts and sorts the active pseudotime and weights for a specific lineage.
    Returns the pseudotimes and weights of each lineage sorted by the given lineage's pseudtime.
    """
    lineage_assignment = get_lineage_assignment(weights)
    mask = lineage_assignment[:, lineage]

    pt_active = pseudotime[mask]
    weights_active = weights[mask]

    # Get sort indices based on the target lineage's column
    sort_idx = np.argsort(pt_active[:, lineage])
    
    # Apply sort to 2D matrices
    pt_sorted = pt_active[sort_idx]
    weights_sorted = weights_active[sort_idx]
    
    return pt_sorted, weights_sorted


def predict_lineage_trajectories(pseudotime, weights, model, gene_idx, is_single_gene, pt_min, pt_max):
    """
    Runs the model once on each lineage and returns the predicted values in a dictionary (because lineages might differ in lengt).
    Also returns the sorted and scaled pseudotimes for plotting. 
    """
    n_lineages = weights.shape[1]
    predictions = {}
    
    model.eval()

    for lineage in range(n_lineages):
        pt_sorted, weights_sorted = sort_by_lineage(pseudotime, weights, lineage)
        pt_sorted_active = pt_sorted[:, lineage]

        # Scale and prepare inputs
        pt_input_scaled = scale_pt(pt_sorted, pt_min, pt_max)
        input_matrix = np.hstack((pt_input_scaled, weights_sorted))
        X_tensor = torch.tensor(input_matrix, dtype=torch.float32)

        # Run model
        with torch.no_grad():
            mu, theta, pi = model(X_tensor)

        if is_single_gene:
            y_line = mu[:, 0].detach().cpu().numpy()        # Only gene is at index 0
        elif gene_idx is None:
            y_line = mu.detach().cpu().numpy()              # Return all genes
        else:
            y_line = mu[:, gene_idx].detach().cpu().numpy() # Gene is at it's index
        
        # Model predicts log counts but predictions should be log1p
        y_line = np.exp(y_line)                         
        y_line = np.log1p(y_line)

        # Store the results for this lineage in the dictionary
        predictions[lineage] = (pt_sorted_active, pt_input_scaled, y_line)

    return predictions


def smoothen_lineage_trajectory(pseudotime, y_line, n_bins=20):
    """
    Smoothens the predictions by averaging the predicted counts into n_bin intervals.
    """

    # Ensure y_line is 2D (samples, genes) so indexing is consistent even if just one sample is provided
    if y_line.ndim == 1:
        y_line = y_line.reshape(-1, 1)

    # Create equally spaced bin boundaries from 0 to the maximum pseudotime
    bin_edges = np.linspace(0, np.max(pseudotime), n_bins + 1)

    # Calculate the center of each bin to use as the new x-axis coordinates
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign each sample's pseudotime to a bin index
    bin_indices = np.digitize(pseudotime, bin_edges) - 1

    # Edge case: index is the exact max value
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    n_genes = y_line.shape[1]
    y_smoothed = np.full((n_bins, n_genes), np.nan)

    # Iterate through each bin and calculate the mean expression of all samples in it
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            y_smoothed[i, :] = np.mean(y_line[mask, :], axis=0)

    # Handles empty bins:
    # - Linear interpolation fills gaps between two bins
    # - ffill/bfill handles cases where the first or last bins are empty
    df_smoothed = pd.DataFrame(y_smoothed)
    y_final = df_smoothed.interpolate(method='linear', axis=0).ffill().bfill().values

    # Return the new x-coordinates and the y-values
    # If only one gene was processed flatten back to 1 dimension
    return bin_centers, y_final.flatten() if y_final.shape[1] == 1 else y_final


def predict_interpolated_trajectories(pseudotime, weights, model, gene_idx, is_single_gene, pt_min, pt_max, n_points=200):
    """
    Interpolate the datasets inputs to create smooth synthetic inputs.
    """
    n_lineages = weights.shape[1]
    predictions = {}
    
    model.eval()

    for lineage in range(n_lineages):
        # 1. Get the real data for this lineage sorted by its pseudotime
        pt_sorted, weights_sorted = sort_by_lineage(pseudotime, weights, lineage)
        pt_active_sorted = pt_sorted[:, lineage]
        
        if len(pt_active_sorted) == 0:
            continue # Skip if no cells are assigned to this lineage

        pt_active_max = pt_active_sorted.max()
        
        # Create the linear synthetic grid for the active pseudotime
        pt_grid = np.linspace(0, pt_active_max, n_points)
        pt_input = np.zeros((n_points, n_lineages))
        pt_input[:, lineage] = pt_grid
        
        # Interpolate the inactive pseudotimes
        for l in range(n_lineages):
            if l != lineage:
                pt_inactive_sorted = pt_sorted[:, l]
                # Fit 3rd degree polynomial: active_pt -> inactive_pt
                poly_fit = np.polyfit(pt_active_sorted, pt_inactive_sorted, deg=3)
                poly_func = np.poly1d(poly_fit)
                pt_input[:, l] = np.clip(poly_func(pt_grid), 0.0, None)

        # Scale synthetic pseudotimes
        pt_input_scaled = scale_pt(pt_input, pt_min, pt_max)
        
        # Interpolate the weights
        w_input = np.zeros((n_points, n_lineages))
        for k in range(n_lineages):
            w_sorted = weights_sorted[:, k]
            # Fit 3rd degree polynomial: active_pt -> weight
            poly_fit = np.polyfit(pt_active_sorted, w_sorted, deg=3)
            poly_func = np.poly1d(poly_fit)
            w_input[:, k] = np.clip(poly_func(pt_grid), 0.0, 1.0)
            
        # Run the model on the synthetic data
        input_matrix = np.hstack((pt_input_scaled, w_input))
        X_tensor = torch.tensor(input_matrix, dtype=torch.float32)

        with torch.no_grad():
            mu, theta, pi = model(X_tensor)

        if is_single_gene:
            y_line = mu[:, 0].detach().cpu().numpy()
        elif gene_idx is None:
            y_line = mu.detach().cpu().numpy()
        else:
            y_line = mu[:, gene_idx].detach().cpu().numpy()
        
        # Convert log(mu) back to log1p(raw counts)
        y_line = np.log1p(np.exp(y_line))

        # Store results (pt_grid is the true x-axis for plotting)
        predictions[lineage] = (pt_grid, pt_input_scaled, y_line)

    return predictions