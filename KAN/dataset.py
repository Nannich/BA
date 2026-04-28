import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SingleCellDataset(Dataset):
    def __init__(self, trajectories, counts):
        self.inputs = torch.tensor(trajectories, dtype=torch.float32)
        self.targets = torch.tensor(counts, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    

def get_dataloaders(data_path, target_gene=None, batch_size=64, train_ratio=0.8):
    # Load the raw data
    counts = pd.read_csv(os.path.join(data_path, "counts.csv"), index_col=0)
    pseudotime = pd.read_csv(os.path.join(data_path, "pseudotime.csv"), index_col=0)
    weights = pd.read_csv(os.path.join(data_path, "weights.csv"), index_col=0)
    
    n_cells = len(counts)
    indices = np.random.permutation(n_cells)
    split_idx = int(n_cells * train_ratio)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    # Scale Pseudotime to [0, 1] to match the KAN grid
    pt_values = pseudotime.values
    train_pt_values = pt_values[train_indices]
    
    pt_min = train_pt_values.min(axis=0, keepdims=True)
    pt_max = train_pt_values.max(axis=0, keepdims=True)
    
    # Apply the training scale to the entire dataset
    pt_scaled = (pt_values - pt_min) / (pt_max - pt_min + 1e-8)

    # Organize input (pseudotime, weights) and target values (gene counts)
    trajectories = np.hstack((pt_scaled, weights.values))

    if target_gene is not None:
        count_values = np.round(counts.values[:, [target_gene]])
    else:
        count_values = np.round(counts.values)
    
    # Create the Datasets
    train_dataset = SingleCellDataset(trajectories[train_indices], count_values[train_indices])
    test_dataset = SingleCellDataset(trajectories[val_indices], count_values[val_indices])

    # Wrap in DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate input and output dimension for the KAN
    input_dim = trajectories.shape[1]
    output_dim = count_values.shape[1] * 3  # Genes * three parameters (mu, theta, pi) for ZINBloss

    return train_dataloader, test_dataloader, input_dim, output_dim, pt_min, pt_max


def get_eval_dataloader(data_path, pt_min, pt_max, target_gene=None, batch_size=256):
    counts = pd.read_csv(os.path.join(data_path, "counts.csv"), index_col=0)
    pseudotime = pd.read_csv(os.path.join(data_path, "pseudotime.csv"), index_col=0)
    weights = pd.read_csv(os.path.join(data_path, "weights.csv"), index_col=0)
    
    # Use the same bounds used during training
    pt_scaled = (pseudotime.values - pt_min) / (pt_max - pt_min + 1e-8)

    trajectories = np.hstack((pt_scaled, weights.values))

    if target_gene is not None:
        count_values = np.round(counts.values[:, [target_gene]])
    else:
        count_values = np.round(counts.values)
    
    full_dataset = SingleCellDataset(trajectories, count_values)
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    return full_dataloader