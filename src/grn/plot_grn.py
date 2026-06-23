import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.core.config import FIGURES_DIR, RESULTS_DIR, DATA_RAW, ensure_dir

def plot_grn(ranked_edges_path, sensitivity=0.15):
    """
    Plots an inferred gene regulatory network from a ranked edges list file,
    filtering edges below a given sensitivity threshold.
    """
    ranked_edges_path = Path(ranked_edges_path)
    if not ranked_edges_path.exists():
        raise FileNotFoundError(f"Target ranked edges file missing: {ranked_edges_path}")

    # Handle pipeline file specifications
    df = pd.read_csv(ranked_edges_path, sep="\t")
    if "EdgeWeight" not in df.columns:
        df = pd.read_csv(ranked_edges_path)

    # Standardize column casing
    df.columns = [col.capitalize() for col in df.columns]
    
    if "Edgeweight" not in df.columns:
        raise KeyError(f"Could not locate 'EdgeWeight' column inside: {ranked_edges_path.name}")

    # Filter out edges udner sensitivity
    df_filtered = df[df["Edgeweight"].abs() > sensitivity]

    G = nx.DiGraph()
    for _, row in df_filtered.iterrows():
        g1, g2, weight = str(row['Gene1']), str(row['Gene2']), float(row['Edgeweight'])
        G.add_edge(g1, g2, weight=weight)

    fig = plt.figure(figsize=(11, 9))

    if len(G.nodes) == 0:
        plt.text(0.5, 0.5, f"No inferred network edges passed sensitivity threshold (> {sensitivity})", 
                 ha='center', va='center', color='gray', fontsize=13)
        plt.axis('off')
        plt.tight_layout()
        return fig

    edges = G.edges(data=True)
    colors = ['royalblue' if d['weight'] > 0 else 'crimson' for u, v, d in edges]
    
    # Scale line width by weight
    weights_abs = [abs(d['weight']) for u, v, d in edges]
    max_weight = max(weights_abs) if weights_abs else 1.0
    widths = [(w / max_weight) * 4.5 + 1.0 for w in weights_abs]

    pos = nx.kamada_kawai_layout(G, weight=None)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths, arrowsize=18, connectionstyle='arc3,rad=0.12', alpha=0.8)
    
    legend_handles = [
        mpatches.Patch(color='royalblue', label='Activation (+)'),
        mpatches.Patch(color='crimson', label='Repression (-)')
    ]
    plt.legend(handles=legend_handles, loc='upper right')
    
    # Dynamically extract tracking variables out of folder segments for headers
    dataset_name = ranked_edges_path.parent.parent.name if len(ranked_edges_path.parts) >= 3 else "Network"
    arch_name = ranked_edges_path.parent.name if len(ranked_edges_path.parts) >= 2 else "Inferred"
    
    plt.title(f"Inferred GRN: {dataset_name} ({arch_name})\nSensitivity > {sensitivity}", 
              fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return fig


def plot_ground_truth_network(dataset_name):
    """
    Automatically traverses up the raw dataset filesystem tree to locate the reference
    GroundTruthNetwork.csv file, and plots the grn.
    """
    matched_dirs = list(DATA_RAW.rglob(f"**/{dataset_name}/ExpressionData.csv"))
    if not matched_dirs:
        raise FileNotFoundError(f"Could not locate dataset: {dataset_name}")
        
    ground_truth_path = None
    for parent in matched_dirs[0].parents:
        check_path = parent / "GroundTruthNetwork.csv"
        if check_path.exists():
            ground_truth_path = check_path
            break
        if parent == DATA_RAW:
            break

    if not ground_truth_path:
        raise FileNotFoundError(f"GroundTruthNetwork.csv missing above: {matched_dirs[0]}")

    df = pd.read_csv(ground_truth_path)
    df.columns = [col.capitalize() for col in df.columns]

    G = nx.DiGraph()
    for _, row in df.iterrows():
        g1, g2, edge_type = str(row['Gene1']), str(row['Gene2']), str(row['Type'])
        G.add_edge(g1, g2, type=edge_type)

    fig = plt.figure(figsize=(11, 9))

    edges = G.edges(data=True)
    colors = ['royalblue' if d['type'] == '+' else 'crimson' for u, v, d in edges]
    widths = [1.5 if u == v else 2.5 for u, v, d in edges]

    pos = nx.kamada_kawai_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths, arrowsize=18, connectionstyle='arc3,rad=0.12', alpha=0.8)
    
    legend_handles = [
        mpatches.Patch(color='royalblue', label='Activation (+)'),
        mpatches.Patch(color='crimson', label='Repression (-)')
    ]
    plt.legend(handles=legend_handles, loc='upper right')
    plt.title(f"Ground Truth Network: {dataset_name}", fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return fig


def run_plot_grn(args):
    """
    Unpacks command line interface arguments, checks for ground truth visualization flag
    and saves figure.
    """
    dataset = args.dataset
    sensitivity = getattr(args, "sensitivity", 0.15)
    use_ground_truth = getattr(args, "ground_truth", False)
    
    run_fig_dir = ensure_dir(FIGURES_DIR / dataset / "grn")
    
    if use_ground_truth:
        out_path = run_fig_dir / f"{dataset}_ground_truth_network.png"
        fig = plot_ground_truth_network(dataset_name=dataset)
    else:
        if getattr(args, "experiment_name", None) is not None:
            experiment_name = args.experiment_name
        elif getattr(args, "arch", None) is not None:
            experiment_name = args.arch
        else:
            input_mode = getattr(args, "input_mode", "log")
            target_mode = getattr(args, "target_mode", "log")
            dt_val = getattr(args, "lag", 0.0)
            deep_config = getattr(args, "deep", False)
            
            in_token = "smo" if input_mode == "smooth" else "log"
            tgt_token = "smo" if target_mode == "smooth" else "log"
            
            if dt_val > 0 and deep_config:
                suffix = "_ld"
            elif dt_val > 0:
                suffix = "_l"
            elif deep_config:
                suffix = "_d"
            else:
                suffix = ""
                
            experiment_name = f"{in_token}_{tgt_token}{suffix}"

        ranked_edges_path = RESULTS_DIR / "grn" / dataset / experiment_name / "rankedEdges.csv"
        out_path = run_fig_dir / f"{dataset}_{experiment_name}_network.png"
        
        print(f"Extracting edges from: {ranked_edges_path}")
        fig = plot_grn(
            ranked_edges_path=ranked_edges_path,
            sensitivity=sensitivity
        )
    
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {out_path}")