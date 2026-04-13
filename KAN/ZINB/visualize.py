import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from efficient_kan import KAN
from model import build_kan, HIDDEN_LAYERS, GRID_SIZE, SPLINE_ORDER
from train import LR, WEIGHT_DECAY

DATA_PATH = "~/BA/data/bifurcating/sim_1/"
MODEL_PATH = "trained_kan_sim1.pth"
# Interesting Genes: 0, 12, 300, 3000

def load_data(data_path):
    counts = pd.read_csv(os.path.join(data_path, "counts.csv"), index_col=0)
    pseudotime = pd.read_csv(os.path.join(data_path, "pseudotime.csv"), index_col=0)
    weights = pd.read_csv(os.path.join(data_path, "weights.csv"), index_col=0)
    return counts, pseudotime, weights


def get_lineage_assignment(weights):
    sensitivity = 0.1
    max_weights = np.max(weights.values, axis=1, keepdims=True)
    lineage_assignment = np.abs(max_weights - weights.values) < sensitivity

    return lineage_assignment

def plot_smoothers(counts, pseudotime, weights, model, gene_idx):
    model.eval()
    n_cells = len(counts)
    n_lineages = weights.shape[1]

    pt_values = pseudotime.values
    pt_min = pt_values.min(axis=0, keepdims=True)
    pt_max = pt_values.max(axis=0, keepdims=True)

    # Assign cells to the lineage where they have the highest weight
    lineage_assignment = get_lineage_assignment(weights)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_lineages))

    with torch.no_grad():
        for l in range(n_lineages):
            # Get the mask for the current lineage
            mask = lineage_assignment[:,l]

            # Extract pt and counts of cells assigned to this lineage for the plot
            # Each cell is one point where x is pseudotime and y is count
            pt_active = pseudotime.values[mask, l]
            log_count_active = np.log1p(counts.values[mask, gene_idx])
            ax.scatter(pt_active, log_count_active, s=16, color=colors[l])

            # Sort weights and pseudotime of the cells of the current lineage by pt
            sort_idx = np.argsort(pt_active)
            pt_active_sorted = pt_active[sort_idx]
            weights_sorted = weights.values[mask][sort_idx]

            # Pseudotime where lineage ends
            pt_active_max = pseudotime.values[:, l].max()

            n_points = 200

            # Create smooth pseudotime matrix as input for plotting
            pt_grid = np.linspace(0, pt_active_max, n_points)
            pt_input = np.zeros((n_points, n_lineages))
            pt_input[:, l] = pt_grid
            
            for k in range(n_lineages):
                if k != l:
                    pt_inactive = pseudotime.values[mask, k]
                    pt_inactive_sorted = pt_inactive[sort_idx]
                    pt_fitted = np.polyfit(pt_active_sorted, pt_inactive_sorted, deg=3)
                    f_inactive_pt = np.poly1d(pt_fitted)
                    pt_input[:, k] = np.clip(f_inactive_pt(pt_grid), 0.0, None)

            # Scale Pseudotime to [0,1]
            pt_input_scaled = (pt_input - pt_min) / (pt_max - pt_min + 1e-8)
            
            # Create smooth weights matrix as input for plotting
            w_input = np.zeros((n_points, n_lineages))
            for k in range(n_lineages):
                w = weights_sorted[:,k]
                w_fitted = np.polyfit(pt_active_sorted, w, deg=3)
                f_w = np.poly1d(w_fitted)
                w_input[:, k] = np.clip(f_w(pt_grid), 0.0, 1)

            # Run the model on the simulated data to get a smooth line
            input_matrix = np.hstack((pt_input_scaled, w_input))
            #print("Input", input_matrix)
            X_tensor = torch.tensor(input_matrix, dtype=torch.float32)
            mu, theta, pi = model(X_tensor)
            y_line = mu[:, gene_idx].numpy()  
            y_line = np.exp(y_line)                         
            y_line = np.log1p(y_line)
            # y_line = model(X_tensor)[:, 0].numpy()                    # Single Gene

            # Plot a smooth line
            ax.plot(pt_grid, y_line, linewidth=3, color=colors[l], label=f"Lineage {l+1}")

            # Plot the individual points            
            # ax.scatter(pt_grid, y_line, s=32, color=colors[l], label=f"Lineage {l+1}", marker="x")

    ax.set_title(f"Gene: {gene_idx}")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Log(expression + 1)")
    ax.legend(title="Lineage")

    hyperparams_text = (
        f"Layers: {HIDDEN_LAYERS}\n"
        f"Grid Size: {GRID_SIZE}\n"
        f"Spline Order: {SPLINE_ORDER}\n"
        f"LR: {LR} | WD: {WEIGHT_DECAY}" 
    )
    
    # Place a text box in the top right corner of the plot
    ax.text(0.98, 0.98, hyperparams_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.show()

def main():
    # Get the gene index from the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("gene", type=int)
    args = parser.parse_args()
    gene_idx = args.gene

    counts, pseudotime, weights = load_data(DATA_PATH)
    
    input_dim = pseudotime.shape[1] + weights.shape[1]
    output_dim = counts.shape[1] * 3    # All Genes * Paramters (mu, theta, pi)
    # output_dim = 1                # Single Gene

    model = build_kan(input_dim, output_dim)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    plot_smoothers(counts, pseudotime, weights, model, gene_idx)

if __name__ == "__main__":
    main()