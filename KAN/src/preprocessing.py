from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from sklearn.cluster import KMeans

from src.utils import *
from src.config import DATA_RAW, DATA_PROCESSED, FIGURES_DIR, ensure_dir

# Preprocessing & Data Loading

def preprocess(adata):
    """
    Preprocesses the count data for trajectory inference
    """
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor='igraph', resolution=0.3, n_iterations=2)    
    return adata

def load_paul15():
    """
    Downloads, processes, caches the paul dataset and exports raw counts to CSV.
    """
    ensure_dir(DATA_PROCESSED)
    cache_path = DATA_PROCESSED / "paul15_processed.h5ad"
    
    paul_csv_dir = ensure_dir(DATA_RAW / "paul15")
    csv_path = paul_csv_dir / "ExpressionData.csv"

    if cache_path.exists():
        return sc.read_h5ad(cache_path)
        
    adata = sc.datasets.paul15()
    
    adata = preprocess(adata)
    
    raw_adata = adata.raw.to_adata()
    counts_df = raw_adata.to_df().T    
    counts_df.to_csv(csv_path)

    adata.write_h5ad(cache_path)
    return adata

def load_local_dataset(dataset_name):
    ensure_dir(DATA_PROCESSED)
    cache_path = DATA_PROCESSED / f"{dataset_name}_processed.h5ad"
    
    if cache_path.exists():
        return sc.read_h5ad(cache_path)

    matched_files = list(DATA_RAW.rglob(f"**/{dataset_name}/ExpressionData.csv"))
    if not matched_files:
        raise FileNotFoundError(f"Missing ExpressionData.csv for {dataset_name}")
        
    df = pd.read_csv(matched_files[0], index_col=0).T
    adata = sc.AnnData(X=df.values)
    adata.obs_names = df.index
    adata.var_names = df.columns

    adata = preprocess(adata)
    adata.write_h5ad(cache_path)
    return adata


# Trajectory Inference

def extract_topology(adata, dataset_name, num_lineages=2):
    """
    Returns the start and end cluster of the dataset based on pseudotime or known
    clusters.
    """
    if dataset_name == "paul":
        root_idx = adata.uns.get('iroot', 0)
        original_clusters = adata.obs["paul15_clusters"].astype(str).values
        prog_clusters = ["7MEP", "9GMP", "10GMP"]
        ery_clusters = ["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "8Mk"]
        
        coarse_clusters = []
        for c in original_clusters:
            if c in prog_clusters: 
                coarse_clusters.append("Progenitor")
            elif c in ery_clusters: 
                coarse_clusters.append("Erythroid")
            else: 
                coarse_clusters.append("Myeloid")
                
        cluster_labels = np.array(coarse_clusters)
        start_cluster = "Progenitor" 
        end_clusters = ["Erythroid", "Myeloid"] 
        
        return root_idx, cluster_labels, start_cluster, end_clusters

    else:
        # Beeline datasets
        pca_coords = adata.obsm['X_pca'][:, :3]
        kmeans = KMeans(n_clusters=4).fit(pca_coords)
        cluster_labels = kmeans.labels_.astype(str)
        
        # Read the ground truth time strictly from RAW
        matched_pt = list(DATA_RAW.rglob(f"**/{dataset_name}/PseudoTime.csv"))
        if not matched_pt:
            raise FileNotFoundError(f"Could not locate PseudoTime.csv for {dataset_name}")
        pt_path = matched_pt[0]
        df_pt = pd.read_csv(pt_path, index_col=0)
        
        # Find the start cluster
        min_times = df_pt.min(axis=1)
        root_cell_id = min_times.idxmin()
        root_idx = adata.obs_names.get_loc(root_cell_id)
        
        start_cluster = cluster_labels[root_idx]
        
        # Find the end clusters
        end_clusters = []
        for col in df_pt.columns:
            if len(end_clusters) >= num_lineages:
                break 
                
            lineage_times = df_pt[col].dropna()
            if not lineage_times.empty:
                terminal_cell_id = lineage_times.idxmax()
                terminal_idx = adata.obs_names.get_loc(terminal_cell_id)
                terminal_cluster = cluster_labels[terminal_idx]
                
                # Ensure it's unique and not the start cluster
                if terminal_cluster != start_cluster and terminal_cluster not in end_clusters:
                    end_clusters.append(terminal_cluster)

        return root_idx, cluster_labels, start_cluster, end_clusters


def run_trajectory(adata, dataset_name="paul"):
    """
    Runs slingshot for trajectory inference.
    """
    ensure_dir(DATA_PROCESSED)
    
    # Define file paths using Pathlib
    cache_path = DATA_PROCESSED / f"{dataset_name}_trajectory.npz"
    csv_path = DATA_PROCESSED / f"{dataset_name}_pseudotime.csv"
    weights_csv_path = DATA_PROCESSED / f"{dataset_name}_weights.csv"
    
    # Route the plot securely into the centralized figures directory
    plot_dir = ensure_dir(FIGURES_DIR / "preprocessing")
    plot_path = plot_dir / f"{dataset_name}_slingshot_plot.png"

    # Check cache (Pathlib uses .exists())
    if cache_path.exists() and csv_path.exists() and plot_path.exists():
        print(f"Loading cached Slingshot data from {cache_path}...")
        cached_data = np.load(cache_path)
        return cached_data['pseudotime'], cached_data['weights']
    
    print(f"Sending data to R to run Slingshot for {dataset_name}...")
    
    # Extract start and end clusters
    root_idx, cluster_labels, start_node, end_nodes = extract_topology(adata, dataset_name)
    
    counts = get_raw_counts(adata)

    # Slingshot R script
    r_script = """
    library(slingshot)

    FQnorm <- function(counts){
        rk <- apply(counts,2,rank,ties.method='min')
        counts.sort <- apply(counts,2,sort)
        refdist <- apply(counts.sort,1,median)
        norm <- apply(rk,2,function(r){ refdist[r] })
        rownames(norm) <- rownames(counts)
        return(norm)
    }

    ti_slingshot <- function(counts_matrix, plot_path, cluster_labels, start_node, end_nodes) {
        
        # Normalize & Reduce
        norm_counts <- FQnorm(counts_matrix)
        pca <- prcomp(t(log1p(norm_counts)), scale. = FALSE)
        rd <- pca$x[, 1:3]

        
        cl <- as.character(cluster_labels)
        cl_factor <- as.factor(cl)
        
        lin <- getLineages(rd, cl, start.clus = start_node, end.clus = as.character(end_nodes))
        crv <- getCurves(lin)

        # Save plot
        png(plot_path, width=800, height=800, res=100)
        colors <- hcl.colors(nlevels(cl_factor), palette = "Set 2") 
        plot(rd[,1:2], col = colors[cl_factor], pch=16, asp = 1, 
             main=paste("Slingshot Trajectory"), xlab="PC1", ylab="PC2")
        lines(SlingshotDataSet(crv), lwd=3, col='black')
        dev.off()
        
        return(list(
            pseudotime = as.matrix(slingPseudotime(crv, na = FALSE)),
            weights = as.matrix(slingCurveWeights(crv))
        ))
    }
    """
    
    # Execute R Code
    robjects.r(r_script)
    ti_slingshot_r = robjects.globalenv['ti_slingshot']
    
    with localconverter(robjects.default_converter + pandas2ri.converter + numpy2ri.converter):
        # Convert Python types to R Vectors
        end_nodes_r = robjects.StrVector(end_nodes) if len(end_nodes) > 0 else robjects.StrVector([])
        
        # Pass everything into R (R needs the path as a standard string)
        result_r = ti_slingshot_r(counts.T, str(plot_path), cluster_labels, str(start_node), end_nodes_r)        
        
        pseudotime = np.array(result_r[0])
        weights = np.array(result_r[1])

    # Clean up NANs & Cache
    pseudotime = np.nan_to_num(pseudotime, nan=0.0)
    weights = np.nan_to_num(weights, nan=0.0)
    
    # Save processed numpy arrays to DATA_PROCESSED
    np.savez(cache_path, pseudotime=pseudotime, weights=weights)

    # Export the Pseudotime and Weights Matrix to a CSV in DATA_PROCESSED
    num_lineages = pseudotime.shape[1]
    col_names = [f"Lineage_{i+1}" for i in range(num_lineages)]
    
    df_pt = pd.DataFrame(pseudotime, index=adata.obs_names, columns=col_names)
    df_pt.to_csv(csv_path)

    df_weights = pd.DataFrame(weights, index=adata.obs_names, columns=col_names)
    df_weights.to_csv(weights_csv_path)

    # Re-fill NaNs to 0.0 before returning to PyTorch
    pseudotime = np.nan_to_num(pseudotime, nan=0.0)
    
    return pseudotime, weights

def parse_beeline_ground_truth(adata, dataset_name):
    """
    Directly extracts true benchmark trajectories.
    Explicitly catches both literal strings like 'NA' and empty missing values,
    safely cleaning them to 0.0 to match original matrix dimensions.
    """
    ensure_dir(DATA_PROCESSED)
    
    # We alter the cache name slightly based on the toggle to prevent overwrite conflicts
    cache_suffix = "gt_trajectory"
    cache_path = DATA_PROCESSED / f"{dataset_name}_{cache_suffix}.npz"
    csv_path = DATA_PROCESSED / f"{dataset_name}_{cache_suffix}_pseudotime.csv"
    weights_csv_path = DATA_PROCESSED / f"{dataset_name}_{cache_suffix}_weights.csv"

    if cache_path.exists() and csv_path.exists():
        cached_data = np.load(cache_path)
        return cached_data['pseudotime'], cached_data['weights']

    matched_pt = list(DATA_RAW.rglob(f"**/{dataset_name}/PseudoTime.csv"))
    if not matched_pt:
        raise FileNotFoundError(f"Missing ground truth PseudoTime.csv for {dataset_name}")
    
    # Standardize missing value tokens to intercept literal 'NA' strings alongside empty commas
    df_pt = pd.read_csv(matched_pt[0], index_col=0, na_values=["NA", "na", ""])
    
    # Reindex matching observations carefully to align with current cell indices
    df_pt = df_pt.reindex(adata.obs_names)

    # 1. Weights array: 1.0 where a valid time value is mapped, 0.0 where it's blank/NA
    weights_matrix = df_pt.notna().astype(float).values
    
    # 2. Pseudotime array: Fill all blank positions or NA records cleanly with 0.0
    pseudotime_matrix = df_pt.fillna(0.0).values

    # Cache outputs exactly where downstream functions look for them
    np.savez(cache_path, pseudotime=pseudotime_matrix, weights=weights_matrix)
    
    col_names = [f"Lineage_{i+1}" for i in range(df_pt.shape[1])]
    pd.DataFrame(pseudotime_matrix, index=adata.obs_names, columns=col_names).to_csv(csv_path)
    pd.DataFrame(weights_matrix, index=adata.obs_names, columns=col_names).to_csv(weights_csv_path)

    return pseudotime_matrix, weights_matrix

# Main Wrapper

def run_preprocessing(dataset_name):
    
    if dataset_name == "paul":
        adata = load_paul15()
    else:
        adata = load_local_dataset(dataset_name)

    #pseudotime, weights = run_trajectory(adata, dataset_name)
    pseudotime, weights = parse_beeline_ground_truth(adata, dataset_name)

    return adata, pseudotime, weights