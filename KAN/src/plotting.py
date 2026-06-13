import math
import numpy as np
import networkx as nx
import torch
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.utils import *
from src.formulas import *
from src.dataloaders import *
from src.model import build_model
from src.config import FIGURES_DIR, MODELS_DIR, TABLES_DIR, DATA_RAW, ensure_dir

KNOWN_GENES = {
    1252: "Gata1", 
    1670: "Klf1",  # Erythroid
    1913: "Mpo", 
    1040: "Elane", # Myeloid
    664:  "Cebpa", 
    619:  "Cd34",  # Progenitor
    1253: "Gata2"
}


def plot_grn(adj_matrix, gene_names, edge_threshold=0.2):
    G = nx.DiGraph()
    n_genes = adj_matrix.shape[0]
    
    for i in range(n_genes):
        G.add_node(i, label=gene_names[i])
        
    for i in range(n_genes):
        for j in range(n_genes):
            weight = adj_matrix[i, j]
            if abs(weight) > edge_threshold:
                G.add_edge(i, j, weight=weight)
                
    edges = G.edges(data=True)
    colors = ['royalblue' if d['weight'] > 0 else 'crimson' for u, v, d in edges]
    
    raw_max = np.max(np.abs(adj_matrix))
    max_weight = raw_max if raw_max > 0 else 1.0
    widths = [(abs(d['weight']) / max_weight) * 5 for u, v, d in edges]
    
    plt.figure(figsize=(12, 10))
    
    pos = nx.circular_layout(G) 
    
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=600, edgecolors='white', linewidths=2)
    
    if edges:
        nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths, arrowsize=15, connectionstyle='arc3,rad=0.1', alpha=0.7)
    else:
        plt.text(0.5, 0.5, "No edges passed threshold", ha='center', va='center', color='gray', fontsize=14)
    
    labels = {i: gene_names[i] for i in range(n_genes)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    legend_handles = [
        mpatches.Patch(color='royalblue', label='Activation (+)'),
        mpatches.Patch(color='crimson', label='Repression (-)')
    ]
    plt.legend(handles=legend_handles, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_clusters(trajectory_cube, cluster_labels, lineage=None, n_bins=20, max_clusters_per_fig=8, save_path=None):
    """
    Plots the smoothed trajectories of genes grouped by their cluster assignments.
    """
    n_genes, n_lineages, _ = trajectory_cube.shape
    n_clusters = len(np.unique(cluster_labels))
    x_vals = np.linspace(0, 1, n_bins)

    if lineage is not None:
        cols = min(4, n_clusters)
        rows = math.ceil(n_clusters / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
        axes = np.array(axes).flatten() 
            
        for c in range(n_clusters):
            ax = axes[c]
            gene_indices = np.where(cluster_labels == c)[0]
            cluster_data = trajectory_cube[gene_indices, lineage, :]

            for i in range(len(gene_indices)):
                ax.plot(x_vals, cluster_data[i], alpha=0.15, color='gray')

            if len(gene_indices) > 0:
                ax.plot(x_vals, cluster_data.mean(axis=0), color='crimson', linewidth=2.5)
            
            ax.set_title(f"Cluster {c}\n(n={len(gene_indices)} genes)")
            if c >= len(axes) - cols: ax.set_xlabel("Pseudotime (Scaled)")
            if c % cols == 0:         ax.set_ylabel(f"Lineage {lineage} Expression")
                
        for c in range(n_clusters, len(axes)):
            axes[c].set_visible(False)
            
        if save_path is not None:
            dataset_name = Path(save_path).stem.split('_')[0].upper()
            fig.suptitle(f"Dataset: {dataset_name}", fontsize=14, fontweight='bold', color='#000000')
            
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
    else:
        grid_cols = 3 
        grid_rows = math.ceil(n_clusters / grid_cols)
        
        bg_colors = ["#f8f9fa", "#ffffff"]
        
        fig, axes = plt.subplots(
            grid_rows, 
            grid_cols * n_lineages, 
            figsize=(3.2 * (grid_cols * n_lineages), 2.4 * grid_rows), 
            sharex=True
        )
        
        if save_path is not None:
            dataset_name = Path(save_path).stem.split('_')[0].upper()
            fig.suptitle(f"Dataset: {dataset_name}", fontsize=16, fontweight='bold', color='#000000', y=1.02)
        
        axes_flat = np.array(axes).flatten()

        for c in range(n_clusters):
            gene_indices = np.where(cluster_labels == c)[0]
            base_col = (c % grid_cols) * n_lineages
            row = c // grid_cols
            
            cluster_bg = bg_colors[(row + (c % grid_cols)) % 2]
            
            for l in range(n_lineages):
                ax_idx = row * (grid_cols * n_lineages) + base_col + l
                ax = axes_flat[ax_idx]
                
                ax.set_facecolor(cluster_bg)
                for spine in ax.spines.values():
                    spine.set_color('#cccccc')
                    spine.set_linewidth(1.2)
                
                cluster_data = trajectory_cube[gene_indices, l, :]

                for i in range(len(gene_indices)):
                    ax.plot(x_vals, cluster_data[i], alpha=0.15, color='gray')

                if len(gene_indices) > 0:
                    ax.plot(x_vals, cluster_data.mean(axis=0), color='royalblue', linewidth=2.5)
                
                ax.set_title("") 
                
                if l == 0:
                    ax.text(
                        0.0, 1.08, f"Cluster {c} (n={len(gene_indices)}) | Lin 1", 
                        transform=ax.transAxes, fontsize=9.5, fontweight='bold', color='#000000'
                    )
                else:
                    ax.text(
                        0.0, 1.08, f"Lin {l+1}", 
                        transform=ax.transAxes, fontsize=9.5, fontweight='bold', color='#000000'
                    )
                
                if row == grid_rows - 1:
                    ax.set_xlabel("Pseudotime", fontsize=8.5, color='#000000')

        total_slots = grid_rows * grid_cols * n_lineages
        active_slots = n_clusters * n_lineages
        for empty_idx in range(active_slots, total_slots):
            axes_flat[empty_idx].set_visible(False)

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.55, wspace=0.20)
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        #plt.show()

def plot_gene_vs_gene(adata, gene_x, gene_y, run_fig_dir=None):
    """
    Plots cell-by-cell expression of gene_x against gene_y in log1p space.
    """
    gene_names = list(adata.var_names)
    
    if gene_x not in gene_names: raise ValueError(f"Gene '{gene_x}' not found in dataset var_names.")
    if gene_y not in gene_names: raise ValueError(f"Gene '{gene_y}' not found in dataset var_names.")
        
    idx_x = gene_names.index(gene_x)
    idx_y = gene_names.index(gene_y)
    
    raw_counts = get_raw_counts(adata)
    x_val = np.log1p(raw_counts[:, idx_x])
    y_val = np.log1p(raw_counts[:, idx_y])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_val, y_val, alpha=0.4, color='blue', s=15, edgecolors='none', label='Cells')
    
    plt.title(f"{gene_x} vs {gene_y}")
    plt.xlabel(f"Log({gene_x} + 1)")
    plt.ylabel(f"Log({gene_y} + 1)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    out_base = run_fig_dir if run_fig_dir is not None else FIGURES_DIR / "visualize"
    out_dir = ensure_dir(out_base / "gene_vs_gene")
    out_path = out_dir / f"{gene_x}_vs_{gene_y}.png"
    
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f" Saved gene vs gene plot to: {out_path}")
    plt.show()


def plot_ground_truth_network(network_csv_path: Path, run_fig_dir=None):
    """
    Plots the grn graph from the beeline edges list.
    """
    if not network_csv_path.exists():
        raise FileNotFoundError(f"Target network file missing: {network_csv_path}")

    df = pd.read_csv(network_csv_path)
    df.columns = [col.capitalize() for col in df.columns]

    G = nx.DiGraph()
    for _, row in df.iterrows():
        g1, g2, edge_type = str(row['Gene1']), str(row['Gene2']), str(row['Type'])
        G.add_edge(g1, g2, type=edge_type)

    edges = G.edges(data=True)
    colors = ['royalblue' if d['type'] == '+' else 'crimson' for u, v, d in edges]
    widths = [1.5 if u == v else 2.5 for u, v, d in edges]

    plt.figure(figsize=(10, 8))
    pos = nx.kamada_kawai_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths, arrowsize=18, connectionstyle='arc3,rad=0.15', alpha=0.8)
    
    legend_handles = [
        mpatches.Patch(color='royalblue', label='Activation (+)'),
        mpatches.Patch(color='crimson', label='Repression (-)')
    ]
    plt.legend(handles=legend_handles, loc='upper right')
    plt.title(f"Ground Truth GRN Architecture: {network_csv_path.parent.name}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    out_base = run_fig_dir if run_fig_dir is not None else FIGURES_DIR / "visualize"
    out_dir = ensure_dir(out_base / "ground_truth")
    out_path = out_dir / f"{network_csv_path.parent.name}_gt_network.png"
    
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved grn graph to: {out_path}")
    plt.show()

# Trajectory visualization

def plot_parameters(ax, model, checkpoint, gene_to_plot):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = checkpoint["model"]
    hidden_layers = checkpoint["hidden_layers"]
    
    mse = checkpoint["mse"]          # Shape: (n_genes, n_lineages)
    aic = checkpoint["aic"]          # Shape: (n_genes,)
    bic = checkpoint["bic"]          # Shape: (n_genes,)
    zinb_loss = checkpoint.get("zinb_loss", 0.0)
    global_aic = checkpoint.get("global_aic", 0.0)
    global_bic = checkpoint.get("global_bic", 0.0)
    
    total_avg_mse = mse.mean().item()

    if gene_to_plot is None:
        mse_lineages = [f"L{l+1}: {mse[:, l].mean().item():.3f}" for l in range(mse.shape[1])]
        aic_val = aic.mean().item()
        bic_val = bic.mean().item()
    else:
        gene_idx = 0 if mse.shape[0] == 1 else gene_to_plot
        mse_lineages = [f"L{l+1}: {mse[gene_idx, l].item():.3f}" for l in range(mse.shape[1])]
        aic_val = aic[gene_idx].item()
        bic_val = bic[gene_idx].item()

    mse_str = " | ".join(mse_lineages)

    text = (
        f"Model: {model_type.upper()} ({checkpoint.get('gene', 'all') if checkpoint.get('gene') else 'ALL'})\n"
        f"Layers: {hidden_layers}\n"
        f"Params: {total_params:,}\n"
        f"-------------------\n"
        f"ZINB Loss: {zinb_loss:.4f}\n"
        f"Avg MSE: {total_avg_mse:.4f}\n"
        f"Global AIC: {global_aic:,.0f}\n"
        f"Global BIC: {global_bic:,.0f}\n"
        f"-------------------\n"
        f"MSE: {mse_str}\n"
        f"AIC: {aic_val:.0f} | BIC: {bic_val:.0f}"
    )

    text_box = ax.text(1.02, 0.5, text, transform=ax.transAxes, 
            fontsize=8.5, verticalalignment='center', horizontalalignment='left',
            linespacing=1.4,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    return text_box


def plot_curves(ax, predictions_raw, colors):
    n_lineages = len(predictions_raw)
    for l in range(n_lineages):
        x_raw, _, y_raw = predictions_raw[l]
        x_smooth, y_smooth = smoothen_lineage_trajectory(x_raw, y_raw)
        ax.plot(x_raw, y_raw, linewidth=1, color=colors[l], label=f"Raw Predictions - Lineage {l+1}", alpha=0.4)
        ax.plot(x_smooth, y_smooth, linewidth=3, color=colors[l], label=f"Smoothed - Lineage {l+1}", alpha=0.6)
        

def plot_custom(ax, pseudotime, checkpoint, colors):
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]
    n_lineages = pseudotime.shape[1]

    for l in range(n_lineages):
        max_pt_for_lineage = np.max(pseudotime[:, l])
        x_clean = np.linspace(0, max_pt_for_lineage, 300)
        
        pt_input_scaled = scale_pt(x_clean, pt_min, pt_max)
        y_formula_raw = pykan_paul_gene1670(pt_input_scaled, lineage=l)
        y_formula = np.log1p(np.exp(y_formula_raw)).flatten()
        
        ax.plot(x_clean, y_formula, linewidth=4, color=colors[l], linestyle="--", label=f"Lineage {l+1} (Symbolic)", zorder=4)


def plot_scatter_data(ax, adata, pseudotime, weights, gene_to_plot, colors):
    lineage_assignment = get_lineage_assignment(weights)
    n_lineages = lineage_assignment.shape[1]
    raw_counts = get_raw_counts(adata)
        
    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        pt_active = pseudotime[mask, l]
        log_count_active = np.log1p(raw_counts[mask, gene_to_plot])
        ax.scatter(pt_active, log_count_active, s=16, color=colors[l], alpha=0.4)


def plot_everything(adata, pseudotime, weights, predictions_raw, model, checkpoint, gene_to_plot, fig_path):
    n_lineages = weights.shape[1]    
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_lineages))

    fig, ax = plt.subplots(figsize=(10, 6))

    text_box = plot_parameters(ax, model, checkpoint, gene_to_plot)
    plot_scatter_data(ax, adata, pseudotime, weights, gene_to_plot, colors)
    plot_curves(ax, predictions_raw, colors)
    
    ax.set_title(f"Gene: {gene_to_plot}")
    gene_name = f"{adata.var_names[gene_to_plot]} ({gene_to_plot})"
    ax.set_title(f"{gene_name} Expression Trajectory", fontsize=16, fontweight='bold')

    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Log(expression + 1)")
    ax.legend(title="Lineage")

    fig.subplots_adjust(left=0.05, bottom=0.08, top=0.92, right=0.8)

    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.savefig(fig_path, bbox_inches="tight", bbox_extra_artists=(text_box,), dpi=300)
    plt.show()


def run_visualization(args, adata, pseudotime, weights, run_model_dir, run_fig_dir):
    gene_to_plot = args.gene
    model_type = getattr(args, 'model', 'effkan')
    dataset = args.dataset
    
    if getattr(args, 'checkpoint_name', None) is not None:
        filename = f"{model_type}_{dataset}_{args.checkpoint_name}.pth"
    else:
        suffix = f"gene{gene_to_plot}" if gene_to_plot is not None else "all"
        filename = f"{model_type}_{dataset}_{suffix}.pth"
        
        if gene_to_plot is not None and not (run_model_dir / filename).exists():
            fallback_filename = f"{model_type}_{dataset}_all.pth"

    model_path = run_model_dir / filename

    if not model_path.exists():
        print(f"No checkpoint at {model_path}")
        return

    print(f"Loading model from: {filename}")
    checkpoint = torch.load(model_path, weights_only=False)
    
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    fig_suffix = f"from_{filename.replace('.pth', '')}"
    fig_path = run_fig_dir / f"{model_type}_{dataset}_plot_gene{gene_to_plot}_{fig_suffix}.png"
    
    model = build_model(model_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    predictions_raw = predict_lineage_trajectories(
        pseudotime, weights, model, gene_to_plot, pt_min, pt_max
    )

    trajectory_cube = build_smoothed_cube(predictions_raw)
    plot_everything(adata, pseudotime, weights, predictions_raw, model, checkpoint, gene_to_plot, fig_path)

    plt.show()


"""
if __name__ == "__main__":

    base_inputs = DATA_RAW / "BEELINE-data" / "inputs"
    out_base_dir = ensure_dir(FIGURES_DIR / "grn_ground_truth")

    curated_names = ["GSD", "HSC", "mCAD", "VSC"]
    for base in curated_names:
        target_csv = base_inputs / "Curated" / base / "GroundTruthNetwork.csv" # Fixed path
        if target_csv.exists():
            plot_ground_truth_network(network_csv_path=target_csv, run_fig_dir=out_base_dir)

    synthetic_base = base_inputs / "Synthetic"
    if synthetic_base.exists():
        for synth_folder in synthetic_base.iterdir():
            if synth_folder.is_dir():
                target_csv = synth_folder / "GroundTruthNetwork.csv" # Fixed path
                if target_csv.exists():
                    plot_ground_truth_network(network_csv_path=target_csv, run_fig_dir=out_base_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Lineage GRN Topology Plotter.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="GSD-2000-1", 
        help="Target dataset directory string name (e.g., GSD-2000-1, VSC-2000-1)"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="raw_to_raw_lin", 
        help="The pipeline folder where matrices are stored."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.15, 
        help="Minimum edge absolute weight filter gate."
    )
    args = parser.parse_args()
    
    target_dataset = args.dataset
    experiment = args.experiment_name
    edge_thresh = args.threshold

    target_grn_dir = TABLES_DIR / target_dataset / experiment / "grn"

    lineage_files = sorted(list(target_grn_dir.glob(f"{target_dataset}_lineage_*_grn.csv")))     

    for lin_file in lineage_files:
        lineage_id = lin_file.name.split("_lineage_")[1].split("_")[0]
        
        df_matrix = pd.read_csv(lin_file, index_col=0)
        
        genes = df_matrix.index.values
        matrix = df_matrix.to_numpy()
        
        plot_title = f"Dataset: {target_dataset.upper()} | Lineage {lineage_id} "
        
        plot_grn(
            adj_matrix=matrix, 
            gene_names=genes, 
            edge_threshold=edge_thresh
        )
"""