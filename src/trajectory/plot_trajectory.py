import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans

from src.trajectory.train_trajectory import build_kan_model, build_mlp_model, build_null_model

def cluster_trajectories(trajectory_cube, n_clusters=9):
    """
    Clusters genes based on their smoothed trajectories across all available lineages.
    """
    n_genes, n_lineages, n_bins = trajectory_cube.shape
    X = trajectory_cube.reshape(n_genes, n_lineages * n_bins)

    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True) + 1e-8
    X_scaled = (X - X_mean) / X_std

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    return labels

def plot_distribution(dataset, gene, colors, ax=None, bins=25):
    """
    Plots a histogram showing the frequency of log-normalized expression values across cells.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    n_lineages = dataset.lineage_assignment.shape[1]
    gene_name = dataset.gene_names[gene]

    for l in range(n_lineages):
        mask = dataset.lineage_assignment[:, l]
        
        if not np.any(mask):
            continue
            
        log_count_active = dataset.log_counts[mask, gene]
        
        ax.hist(
            log_count_active, 
            bins=bins, 
            color=colors[l], 
            alpha=0.6, 
            edgecolor=colors[l],
            linewidth=1.2,
            histtype='stepfilled',
            label=f"Lineage {l + 1}"
        )

    ax.set_title(f"(b) {gene_name} Expression Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel(r"ln(Expression + 1)", fontsize=11)
    ax.set_ylabel("Number of Cells", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    if n_lineages > 1:
        ax.legend(loc="upper right")
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    return ax

def plot_scatter(dataset, gene, colors, ax=None, alpha=0.6, s=16):
    """
    Plots cell-by-cell log gene expression against pseudotime for each lineage.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    n_lineages = dataset.lineage_assignment.shape[1]
    gene_name = dataset.gene_names[gene]

    for l in range(n_lineages):
        mask = dataset.lineage_assignment[:, l]
        
        if not np.any(mask):
            continue
            
        pt_active = dataset.pseudotime[mask, l]
        log_count_active = dataset.log_counts[mask, gene]
        
        ax.scatter(
            pt_active, 
            log_count_active, 
            s=s, 
            color=colors[l], 
            alpha=alpha, 
            edgecolors='none',
            label=f"Lineage {l + 1}"
        )

    ax.set_title(f"(a) {gene_name} Expression Across Cells", fontsize=14, fontweight='bold')
    ax.set_xlabel("Pseudotime", fontsize=11)
    ax.set_ylabel(r"ln(Expression + 1)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    if n_lineages > 1:
        ax.legend(loc="upper right")
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    return ax

def plot_trajectory(dataset, gene, colors, model_dir, model_type="kan", loss="mse", ax=None):
    """
    Extracts trajectory curves from a saved single-gene model checkpoint
    and overlays them on the scatter plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    plot_scatter(dataset=dataset, gene=gene, colors=colors, ax=ax)
    
    gene_name = dataset.gene_names[gene]
    ax.set_title(f"{gene_name} Trajectory ({model_type.upper()})", fontsize=14, fontweight='bold')
    
    filename = f"{model_type}_{gene_name}_{loss}.pth"
    model_path = Path(model_dir) / filename

    if not model_path.exists():
        print(f"No checkpoint found at {model_path}")
        return ax

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    hidden_layers = checkpoint.get("hidden_layers", [])
    n_lineages = dataset.pseudotime.shape[1]
    output_dim = 1  

    if model_type == "kan":
        model = build_kan_model(n_lineages, output_dim, hidden_layers)
    elif model_type == "mlp":
        model = build_mlp_model(n_lineages, output_dim, hidden_layers)
    elif model_type == "null":
        model = build_null_model(output_dim)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        for l in range(n_lineages):
            mask = dataset.lineage_assignment[:, l]
            if not np.any(mask):
                continue

            max_pt = np.max(dataset.pseudotime[mask, l])
            pt_dense = np.linspace(0, max_pt, 300)

            X_eval = np.zeros((300, n_lineages))
            X_eval[:, l] = pt_dense
            X_tensor = torch.tensor(X_eval, dtype=torch.float32)

            mu, _, _ = model(X_tensor)
            y_preds = torch.log1p(mu[:, 0]).cpu().numpy()

            ax.plot(
                pt_dense, 
                y_preds, 
                color=colors[l], 
                linewidth=3.5, 
                solid_capstyle="round"
            )

    ax.legend(loc="upper right")
    return ax

def plot_clusters(trajectory_cube, cluster_labels, n_bins=32):
    """
    Plots the trajectory curves in clusters.
    """
    unique_clusters = np.sort(np.unique(cluster_labels))
    n_clusters = len(unique_clusters)

    n_cols = int(np.ceil(np.sqrt(n_clusters)))
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), sharex=True)
    if n_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    n_genes, n_lineages, _ = trajectory_cube.shape
    x_grid = np.linspace(0, 1, n_bins)

    cmap_name = "viridis"
    if n_lineages == 2:
        colors = plt.get_cmap(cmap_name)(np.linspace(0, 0.66, n_lineages))
    else:
        colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, n_lineages))

    for idx, c in enumerate(unique_clusters):
        ax = axes[idx]
        cluster_mask = (cluster_labels == c)
        cluster_cube = trajectory_cube[cluster_mask]
        n_cluster_genes = cluster_cube.shape[0]

        for l in range(n_lineages):
            for g in range(n_cluster_genes):
                ax.plot(
                    x_grid, 
                    cluster_cube[g, l, :], 
                    color=colors[l], 
                    alpha=0.12, 
                    linewidth=0.8
                )

            if n_cluster_genes > 0:
                mean_profile = cluster_cube[:, l, :].mean(axis=0)
                ax.plot(
                    x_grid, 
                    mean_profile, 
                    color=colors[l], 
                    linewidth=3.0, 
                    label=f"Mean Fit Lin {l + 1}" if idx == 0 else ""
                )

        ax.set_title(f"Cluster {c + 1} (n={n_cluster_genes})", fontsize=12, fontweight='bold')
        ax.set_ylabel(r"ln(Expression + 1)", fontsize=10)
        ax.set_xlabel("Normalized Pseudotime", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_lineages > 1:
        axes[0].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


def generate_cluster_figure(dataset, model_dir, model_type="kan", loss_type="mse"):
    """
    Assembles expression trajectories across trained checkpoints and clusters them.
    """
    run_model_dir = Path(model_dir)
    n_lineages = dataset.lineage_assignment.shape[1]
    n_bins = 32  

    X_eval = np.zeros((n_lineages * n_bins, n_lineages))
    for l in range(n_lineages):
        mask = dataset.lineage_assignment[:, l]
        if np.any(mask):
            X_eval[l * n_bins : (l + 1) * n_bins, l] = np.linspace(0, np.max(dataset.pseudotime[mask, l]), n_bins)
    X_tensor = torch.tensor(X_eval, dtype=torch.float32)

    model_builders = {"kan": build_kan_model, "mlp": build_mlp_model, "null": build_null_model}
    valid_trajectory_list = []

    print(f"Clustering checkpoints in: {run_model_dir}")

    for gene_name in dataset.gene_names:
        model_path = run_model_dir / f"{model_type}_{gene_name}_{loss_type}.pth"
        if not model_path.exists():
            continue

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        hidden_layers = checkpoint.get("hidden_layers", [])
        
        model = model_builders[model_type](1) if model_type == "null" else \
                model_builders[model_type](n_lineages, 1, hidden_layers)
        
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        with torch.no_grad():
            mu, _, _ = model(X_tensor)
            y_preds = torch.log1p(mu[:, 0]).cpu().numpy()
            
        valid_trajectory_list.append(y_preds.reshape(n_lineages, n_bins))

    if not valid_trajectory_list:
        print(f"No {model_type} checkpoints found in {run_model_dir}.")
        return None

    trajectory_cube = np.array(valid_trajectory_list)
    cluster_labels = cluster_trajectories(trajectory_cube, n_clusters=min(len(trajectory_cube), 9))
    
    return plot_clusters(trajectory_cube, cluster_labels, n_bins=n_bins)


def run_plotting(args, dataset):
    """
    Resolves command flags, and saves plots.
    """
    mode = args.mode
    cmap = "viridis"
    n_lineages = dataset.lineage_assignment.shape[1]

    if n_lineages == 2:
        colors = plt.get_cmap(cmap)(np.linspace(0, 0.66, n_lineages))
    else:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_lineages))

    fig = None

    if mode == "scatter":
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_scatter(dataset=dataset, gene=args.gene, ax=ax, colors=colors)
        
    elif mode == "distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_distribution(dataset=dataset, gene=args.gene, ax=ax, colors=colors)
        
    elif mode == "trajectory":
        fig, ax = plt.subplots(figsize=(10, 6))
        model_type = getattr(args, "model", "kan")
        loss = getattr(args, "loss", "mse")
        model_dir = args.model_dir
        
        plot_trajectory(
            dataset=dataset, 
            gene=args.gene, 
            colors=colors, 
            model_dir=model_dir, 
            model_type=model_type, 
            loss=loss, 
            ax=ax
        )
        
    elif mode == "cluster":
        model_type = getattr(args, "model", "kan")
        loss_type = getattr(args, "loss", "mse")
        model_dir = args.model_dir
        fig = generate_cluster_figure(
            dataset=dataset,
            model_dir=model_dir,
            model_type=model_type,
            loss_type=loss_type
        )

    if fig is None:
        return

    tokens = [args.dataset, mode]
    
    if mode in ["scatter", "distribution", "trajectory"]:
        gene_idx = getattr(args, "gene", None)
        if gene_idx is not None:
            tokens.append(dataset.gene_names[gene_idx])
    elif mode == "cluster":
        model_type = getattr(args, "model", "kan")
        tokens.append(model_type)

    loss_type = getattr(args, "loss", None)
    if loss_type is not None:
        tokens.append(loss_type)
    
    fig_name = f"{'_'.join(tokens)}.png"
    fig_path = args.fig_dir / fig_name

    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    print(f"Saved plot to: {fig_path}")