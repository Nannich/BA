import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from efficient_kan import KAN
from model import (
    build_model,
    MLP_HIDDEN_LAYERS, PYKAN_HIDDEN_LAYERS, EFFKAN_HIDDEN_LAYERS, PYKAN_GRID_SIZE, EFFKAN_GRID_SIZE, EFFKAN_SPLINE_ORDER, PYKAN_SPLINE_ORDER
)
from train import LR, WEIGHT_DECAY, DATA_PATH, MODEL_PATH, SIM

# Interesting Genes
# Bifurcating 1:        0, 12, 300, 3000
# Multifurcating 9:     18, 23, 26

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
        y_line = mu[:, gene_idx].numpy()
    else: 
        y_line = mu[:, 0].numpy()
    
    # Because of the exp link function the model predicts the log count
    y_line = np.exp(y_line)                         
    y_line = np.log1p(y_line)

    return pt_grid, y_line

def plot_parameters(ax, model, model_name, is_single_gene):
    # Plot a textbox with the hyperparamters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if model_name == "effkan":
        hyperparams_text = (
            "efficient-kan\n"
            f"Layers: {EFFKAN_HIDDEN_LAYERS}\n"
            f"Parameters: {total_params}\n"
            f"Grid Size: {EFFKAN_GRID_SIZE}\n"
            f"Spline Order: {EFFKAN_SPLINE_ORDER}\n"
            f"LR: {LR} | WD: {WEIGHT_DECAY}" 
        )
    elif model_name == "pykan":
        hyperparams_text = (
            "pyKAN\n"
            f"Layers: {PYKAN_HIDDEN_LAYERS}\n"
            f"Parameters: {total_params}\n"
            f"Grid Size: {PYKAN_GRID_SIZE}\n"
            f"Spline Order: {PYKAN_SPLINE_ORDER}\n"
            f"LR: {LR} | WD: {WEIGHT_DECAY}" 
        )
    elif model_name == "mlp":
        activation_name = model.mlp[1].__class__.__name__
        hyperparams_text = (
            "MLP\n"
            f"Layers: {MLP_HIDDEN_LAYERS}\n"
            f"Parameters: {total_params}\n"
            f"Activation: {activation_name}\n"
            f"LR: {LR} | WD: {WEIGHT_DECAY}" 
        )
    
    if is_single_gene: 
        hyperparams_text = f"{hyperparams_text}\nSingle Gene"
    else:
        hyperparams_text = f"{hyperparams_text}\nAll Genes"

    # Place the text box in the bottom left corner of the plot
    ax.text(0.05, 0.1, hyperparams_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_smoothers(counts, pseudotime, weights, model, model_name, gene_idx, is_single_gene):
    # Evaluates the model along each lineage and plots the predicted gene expression count
    # against the data from the data set used during training
    model.eval()
    
    n_lineages = weights.shape[1]
    lineage_assignment = get_lineage_assignment(weights)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_lineages))

    for l in range(n_lineages):
        mask = lineage_assignment[:, l]

        # Extract pt and counts of cells assigned to this lineage for the plot
        # Each cell is one point where x is pseudotime and y is count
        pt_active = pseudotime.values[mask, l]
        log_count_active = np.log1p(counts.values[mask, gene_idx])
        ax.scatter(pt_active, log_count_active, s=16, color=colors[l], alpha=0.6)

        # Plot the curve predicted by the model for the lineage
        pt_grid, y_line = get_plotting_data(
            pseudotime, weights, model, gene_idx, target_lineage=l, is_single_gene=is_single_gene
        )
        
        ax.plot(pt_grid, y_line, linewidth=3, color=colors[l], label=f"Lineage {l+1}")

        # Plot the individual points            
        # ax.scatter(pt_grid, y_line, s=32, color=colors[l], label=f"Lineage {l+1}", marker="x")
    
    ax.set_title(f"Gene: {gene_idx}")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Log(expression + 1)")
    ax.legend(title="Lineage")

    plot_parameters(ax, model, model_name, is_single_gene)

    plt.show()

def main():
    # Get the gene index from the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("gene", type=int)
    parser.add_argument("name", type=str)
    args = parser.parse_args()
    gene_idx = args.gene

    counts, pseudotime, weights = load_data(DATA_PATH)

    load_path = f"{MODEL_PATH}{args.name}.pth"
    
    checkpoint = torch.load(load_path, weights_only=False)
    
    model_type = checkpoint ["model"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    gene = checkpoint["gene"]

    is_single_gene = gene is not None
    
    model = build_model(model_type, input_dim, output_dim)
    
    model.load_state_dict(checkpoint["state_dict"])

    plot_smoothers(counts, pseudotime, weights, model, model_type, gene_idx, is_single_gene)

if __name__ == "__main__":
    main()