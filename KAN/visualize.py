import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *
from formulas import *
from dataset import *
from model import build_model


def plot_parameters(ax, model, checkpoint, gene_to_plot):
    # Plot a textbox with the hyperparamters
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
    # Evaluates the model along each lineage and plots the predicted gene expression count
    # against the data from the data set used during training
    model.eval()

    gene = checkpoint["gene"]
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    is_single_gene = gene is not None
    
    n_lineages = weights.shape[1]

    for l in range(n_lineages):

        # Plot the curve predicted by the model for the lineage
        pt_grid, pt_input_scaled, y = predict_lineage_curve(
            pseudotime, weights, model, gene_to_plot, l, is_single_gene, pt_min, pt_max
        )
        
        ax.plot(pt_grid, y, linewidth=3, color=colors[l], label=f"Lineage {l+1}")
        

def plot_custom(ax, pseudotime, checkpoint, colors):
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    n_lineages = pseudotime.shape[1]
    for l in range(n_lineages):
        pt_input_scaled = scale_pt(pseudotime.values, pt_min, pt_max)
        y_formula_raw = sigmoid_sim1_gene12(pt_input_scaled[:, l], lineage=l)
        y_formula = np.log1p(np.exp(y_formula_raw))                       
        ax.plot(pseudotime, y_formula, linewidth=3, color=colors[l], linestyle="--", label=f"Lineage {l+1} (Symbolic)", alpha=0.7)


def plot_scatter_data(ax, counts, pseudotime, weights, gene_to_plot, colors):
    lineage_assignment = get_lineage_assignment(weights)
    n_lineages = lineage_assignment.shape[1]
    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        pt_active = pseudotime.values[mask, l]
        log_count_active = np.log1p(counts.values[mask, gene_to_plot])
        ax.scatter(pt_active, log_count_active, s=16, color=colors[l], alpha=0.6)


def plot_everything(counts, pseudotime, weights, model, checkpoint, gene_to_plot, fig_path):
    model.eval()

    n_lineages = weights.shape[1]
    
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_lineages))

    fig, ax = plt.subplots(figsize=(10, 6))

    text_box = plot_parameters(ax, model, checkpoint, gene_to_plot,)
    plot_scatter_data(ax, counts, pseudotime, weights, gene_to_plot, colors)
    plot_curves(ax, pseudotime, weights, model, gene_to_plot, checkpoint, colors)
    #plot_custom(ax, pseudotime, checkpoint, colors)

    ax.set_title(f"Gene: {gene_to_plot}")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Log(expression + 1)")
    ax.legend(title="Lineage")

    fig.subplots_adjust(left=0.05, bottom=0.08, top=0.92, right=0.8)

    plt.savefig(fig_path, bbox_inches="tight", bbox_extra_artists=(text_box,), dpi=300)
    plt.show()


def run_visualization(args):
    gene_to_plot = args.gene
    model_dir = args.model_dir
    fig_dir = args.fig_dir
    model_name = args.name
    dataset = args.dataset
    sim = args.sim

    model_path = os.path.join(model_dir, dataset, model_name)
    data_path = os.path.join(args.data_dir, dataset, f"sim_{sim}/")

    counts, pseudotime, weights, tde = load_data(data_path)

    checkpoint = torch.load(model_path, weights_only=False)
    
    model_type = checkpoint["model"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]

    fig_path = os.path.join(fig_dir, "visualize", dataset, f"{model_type}_sim{sim}_gene{gene_to_plot}.png")
    
    model = build_model(model_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])

    plot_everything(counts, pseudotime, weights, model, checkpoint, gene_to_plot, fig_path)
