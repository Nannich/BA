from pathlib import Path
import pandas as pd
import numpy as np

from src.core.types import SingleCellDataset
from src.core.config import DATA_RAW, DATA_PROCESSED, ensure_dir

def load_and_align_raw_data(dataset_name):
    """
    Loads raw expression and pseudotime CSV files, aligns cells via Pandas, 
    and returns the individual arrays.
    """
    matched_dirs = list(DATA_RAW.rglob(dataset_name))

    if not matched_dirs:
        raise FileNotFoundError(f"No dataset folder found for: {dataset_name}")

    dataset_dir = matched_dirs[0]
    counts_path = dataset_dir / "ExpressionData.csv"
    pt_path = dataset_dir / "PseudoTime.csv"

    # Verify that the files exists
    if not counts_path.exists():
        raise FileNotFoundError(f"Missing ExpressionData.csv at: {counts_path}")
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing PseudoTime.csv at: {pt_path}")
        
    # Load expression counts (Transposing to cells x genes)
    df_counts = pd.read_csv(counts_path, index_col=0).T
    cell_names = df_counts.index.values.astype(str)
    gene_names = df_counts.columns.values.astype(str)

    # Load pre-computed pseudotime trajectories
    df_pt = pd.read_csv(pt_path, index_col=0, na_values=["NA", "na", ""])
    
    # Ensures cell alignment across both files
    df_pt = df_pt.reindex(df_counts.index)

    # Create counts array and log1p normalize it
    raw_counts = df_counts.values.astype(np.float32)
    log_counts = np.log1p(raw_counts)
    
    # Extract boolean lineage assignment mask from the pseudotime dataframe by 
    # replacing na with False and any numerical value with True
    lineage_assignment = df_pt.notna().values

    # Replace every na in the pseudotime dataframe with 0
    pseudotime = df_pt.fillna(0.0).values.astype(np.float32)

    return raw_counts, log_counts, pseudotime, lineage_assignment, gene_names, cell_names


def run_preprocessing(dataset_name):
    """
    Main entry point for pipeline data preprocessing. 
    Manages dataset caching and returns a SingleCellDataset object.
    """
    ensure_dir(DATA_PROCESSED)
    cache_path = DATA_PROCESSED / f"{dataset_name}.npz"

    # If cached file exists, load it
    if cache_path.exists():
        print(f"Loaded cached data from: {cache_path}")
        with np.load(cache_path, allow_pickle=True) as data:
            return SingleCellDataset(
                raw_counts=data['raw_counts'],
                log_counts=data['log_counts'],
                pseudotime=data['pseudotime'],
                lineage_assignment=data['lineage_assignment'],
                gene_names=data['gene_names'],
                cell_names=data['cell_names']
            )

    # Otherwise, parse raw matrices and compute transformations
    raw_counts, log_counts, pseudotime, lineage_assignment, gene_names, cell_names = load_and_align_raw_data(dataset_name)

    # Cache all arrays
    np.savez_compressed(
        cache_path,
        raw_counts=raw_counts,
        log_counts=log_counts,
        pseudotime=pseudotime,
        lineage_assignment=lineage_assignment,
        gene_names=gene_names,
        cell_names=cell_names
    )

    print(f"Finished preprocessing {dataset_name} and cached it at: {cache_path}")
    
    # Wrap inside the data container
    return SingleCellDataset(
        raw_counts=raw_counts,
        log_counts=log_counts,
        pseudotime=pseudotime,
        lineage_assignment=lineage_assignment,
        gene_names=gene_names,
        cell_names=cell_names
    )