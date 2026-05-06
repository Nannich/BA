import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *
from formulas import *
from dataloaders import *
from model import build_model


def plot_parameters(ax, model, checkpoint, gene_to_plot):
    """
    Plots a textbox with the hyperparamters of the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = checkpoint["model"]
    hidden_layers = checkpoint["hidden_layers"]
    lr = checkpoint["lr"]
    wd = checkpoint["wd"]
    gene = checkpoint["gene"]
    mse = checkpoint["mse"]
    aic = checkpoint["aic"]
    bic =  checkpoint["bic"]
    model_gene = checkpoint["gene"]
    
    is_single_gene = model_gene is not None
    gene_idx = 0 if is_single_gene else gene_to_plot

    mse_list = [f"L{l+1}: {mse[gene_idx, l].item():.3f}" for l in range(mse.shape[1])]
    mse_val = " | ".join(mse_list)
    aic_val = aic[gene_idx].item()
    bic_val = bic[gene_idx].item()

    text = (
        f"Model: {model_type}\n"
        f"Hidden layers: {hidden_layers}\n"
        f"Parameters: {total_params}\n"
        f"LR: {lr} | WD: {wd}\n"
        f"MSE: {mse_val}\n"
        f"AIC: {aic_val:.0f} | BIC: {bic_val:.0f}"
    )

    if gene is None:
        text = f"{text}\nAll Genes"
    else:
        text = f"{text}\nSingle Gene"

    text_box = ax.text(1.02, 0.5, text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='center', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return text_box

def plot_curves(ax, pseudotime, weights, model, gene_to_plot, checkpoint, colors):
    """
    Plots the raw and smoothed prediction curves of the model.
    """

    model.eval()

    gene = checkpoint["gene"]
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    is_single_gene = gene is not None
    
    n_lineages = weights.shape[1]

    predictions_raw = predict_lineage_trajectories(pseudotime, weights, model, gene_to_plot, is_single_gene, pt_min, pt_max)
    #predictions_smooth = predict_interpolated_trajectories(pseudotime, weights, model, gene_to_plot, is_single_gene, pt_min, pt_max)

    for l in range(n_lineages):
        x_raw, _, y_raw = predictions_raw[l]
        x_smooth, y_smooth = smoothen_lineage_trajectory(x_raw, y_raw)
        #x_smooth, _, y_smooth = predictions_smooth[l]

        ax.plot(x_raw, y_raw, linewidth=1, color=colors[l], label=f"Lineage {l+1}", alpha=0.5)
        ax.plot(x_smooth, y_smooth, linewidth=3, color=colors[l], label=f"Lineage {l+1}", alpha=1)
        

def plot_custom(ax, pseudotime, checkpoint, colors):
    """
    Plots the curve of a custom formula.
    """
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    n_lineages = pseudotime.shape[1]
    for l in range(n_lineages):
        pt_input_scaled = scale_pt(pseudotime, pt_min, pt_max)
        y_formula_raw = sigmoid_sim1_gene12(pt_input_scaled[:, l], lineage=l)
        y_formula = np.log1p(np.exp(y_formula_raw))                       
        ax.plot(pseudotime, y_formula, linewidth=3, color=colors[l], linestyle="--", label=f"Lineage {l+1} (Symbolic)", alpha=0.7)


def plot_scatter_data(ax, adata, pseudotime, weights, gene_to_plot, colors):
    """
    Plots the expression count of each cell for each lineage it is in.
    """
    lineage_assignment = get_lineage_assignment(weights)
    n_lineages = lineage_assignment.shape[1]
    
    # Get raw counts for plotting 
    raw_counts = adata.raw.X
    if hasattr(raw_counts, "toarray"):
        raw_counts = raw_counts.toarray()
        
    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        pt_active = pseudotime[mask, l]
        log_count_active = np.log1p(raw_counts[mask, gene_to_plot])
        ax.scatter(pt_active, log_count_active, s=16, color=colors[l], alpha=0.4)


def plot_everything(adata, pseudotime, weights, model, checkpoint, gene_to_plot, fig_path):
    """
    Plots the scatter data and curves.
    """
    model.eval()

    n_lineages = weights.shape[1]
    
    # Use the specific colormap for lineages
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_lineages))

    fig, ax = plt.subplots(figsize=(10, 6))

    text_box = plot_parameters(ax, model, checkpoint, gene_to_plot,)
    
    plot_scatter_data(ax, adata, pseudotime, weights, gene_to_plot, colors)
    
    plot_curves(ax, pseudotime, weights, model, gene_to_plot, checkpoint, colors)
    
    ax.set_title(f"Gene: {gene_to_plot}")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Log(expression + 1)")
    ax.legend(title="Lineage")

    fig.subplots_adjust(left=0.05, bottom=0.08, top=0.92, right=0.8)

    plt.savefig(fig_path, bbox_inches="tight", bbox_extra_artists=(text_box,), dpi=300)
    plt.show()


def run_visualization(args, adata, pseudotime, weights):
    gene_to_plot = args.gene
    model_dir = args.model_dir
    fig_dir = args.fig_dir
    model_name = args.name
    dataset = args.dataset

    model_path = os.path.join(model_dir, model_name)

    checkpoint = torch.load(model_path, weights_only=False)
    
    model_type = checkpoint["model"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]

    fig_path = os.path.join(fig_dir, "visualize", dataset, f"{model_type}_{dataset}_gene{gene_to_plot}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    model = build_model(model_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])

    plot_everything(adata, pseudotime, weights, model, checkpoint, gene_to_plot, fig_path)
