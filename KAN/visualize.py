import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from train import LR, WEIGHT_DECAY
from utils import get_plotting_data, load_data, get_lineage_assignment
from formulas import *
from model import (
    build_model,
    MLP_HIDDEN_LAYERS, PYKAN_HIDDEN_LAYERS, EFFKAN_HIDDEN_LAYERS, PYKAN_GRID_SIZE, EFFKAN_GRID_SIZE, EFFKAN_SPLINE_ORDER, PYKAN_SPLINE_ORDER
)


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


def plot_smoothers(counts, pseudotime, weights, model, model_name, gene_idx, is_single_gene, fig_path):
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
        pt_grid, pt_input_scaled, y_line = get_plotting_data(
            pseudotime, weights, model, gene_idx, target_lineage=l, is_single_gene=is_single_gene
        )
        
        ax.plot(pt_grid, y_line, linewidth=3, color=colors[l], label=f"Lineage {l+1}")

        # Plot the individual points            
        # ax.scatter(pt_grid, y_line, s=32, color=colors[l], label=f"Lineage {l+1}", marker="x")

        # Plot a custom formula
        y_formula_raw = pysr_pykan_sim1_gene12(pt_input_scaled[:, l], lineage=l)
        y_formula = np.log1p(np.exp(y_formula_raw))                       
        ax.plot(pt_grid, y_formula, linewidth=3, color=colors[l], linestyle="--", label=f"Lineage {l+1} (Symbolic)", alpha=0.7)
    
    ax.set_title(f"Gene: {gene_idx}")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Log(expression + 1)")
    ax.legend(title="Lineage")

    plot_parameters(ax, model, model_name, is_single_gene)

    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.show()


def run_visualization(args):
    gene = args.gene
    data_dir = args.data_dir
    model_dir = args.model_dir
    fig_dir = args.fig_dir
    model_name = args.name
    dataset = args.dataset
    sim = args.sim

    model_path = os.path.join(model_dir, dataset, model_name)
    data_path = os.path.join(args.data_dir, dataset, f"sim_{sim}/")

    counts, pseudotime, weights = load_data(data_path)

    checkpoint = torch.load(model_path, weights_only=False)
    
    model_type = checkpoint ["model"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    model_gene = checkpoint["gene"]

    fig_path = os.path.join(fig_dir, "visualize", dataset, f"{model_type}_sim{sim}_gene{gene}.png")

    is_single_gene = model_gene is not None
    
    model = build_model(model_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])

    plot_smoothers(counts, pseudotime, weights, model, model_type, gene, is_single_gene, fig_path)