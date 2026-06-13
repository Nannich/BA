import numpy as np
from sklearn.cluster import KMeans
import torch

from src.utils import *
from src.dataloaders import *
from src.config import ensure_dir
from src.plotting import plot_clusters
from src.model import build_model

def cluster_trajectories(trajectory_cube, n_clusters=20, lineage=None):
    """
    Clusters genes based on their smoothed trajectories.
    """
    n_genes, n_lineages, n_bins = trajectory_cube.shape

    if lineage is not None:
        # Cluster based on a specific lineage -> Shape: (n_genes, n_bins)
        X = trajectory_cube[:, lineage, :]
    else:
        # Cluster based on all lineages combined -> Shape: (n_genes, n_lineages * n_bins)
        X = trajectory_cube.reshape(n_genes, n_lineages * n_bins)

    # Standardize each genes trajectory (Z-score normalization)
    # So genes with similar expression shapes cluster together 
    # regardless of differences in absolute expression levels
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True) + 1e-8
    X_scaled = (X - X_mean) / X_std

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X_scaled)

    return labels


def run_plot_clusters(adata, pseudotime, weights, run_model_dir, run_fig_dir):
    model_type = "effkan"
    loss_type = "mse"
    n_bins = 32

    dataset = run_model_dir.parent.name
    model_path = run_model_dir / f"{model_type}_{dataset}_all_{loss_type}.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint at: {model_path}")

    checkpoint = torch.load(model_path, weights_only=False)
    
    model = build_model(model_type, checkpoint["input_dim"], checkpoint["output_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    
    predictions_raw = predict_lineage_trajectories(
        pseudotime, weights, model, None, checkpoint["pt_min"], checkpoint["pt_max"]
    )
    n_genes = checkpoint["output_dim"] // 3
    
    n_clusters = min(n_genes, 9)
    
    filtered_preds = filter_predictions(predictions_raw, np.ones(n_genes, dtype=bool))
    trajectory_cube = build_smoothed_cube(filtered_preds, n_bins=n_bins)
    
    cluster_labels = cluster_trajectories(trajectory_cube, n_clusters=n_clusters, lineage=None)
    
    out_dir = ensure_dir(run_fig_dir / "clusters")
    fig_path = out_dir / f"{dataset}_clusters_all_lineages.png"
    
    plot_clusters(
        trajectory_cube=trajectory_cube, 
        cluster_labels=cluster_labels, 
        lineage=None, 
        n_bins=n_bins, 
        save_path=fig_path
    )
    
    print(f"Cluster plot saved to: {fig_path}")