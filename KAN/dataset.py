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
    

def get_dataloaders(data_path, batch_size=64, train_ratio=0.8):
    
    # 1. Load the raw data
    counts = pd.read_csv(os.path.join(data_path, "counts.csv"), index_col=0)
    pseudotime = pd.read_csv(os.path.join(data_path, "pseudotime.csv"), index_col=0)
    weights = pd.read_csv(os.path.join(data_path, "weights.csv"), index_col=0)
    
    # 2. Organize input (pseudotime, weights) and target values (gene counts)
    trajectories = np.hstack((pseudotime.values, weights.values))
    count_values = counts.values
    
    # 3. Shuffle and split into training and validation
    n_cells = len(trajectories)
    indices = np.random.permutation(n_cells)
    split_idx = int(n_cells * train_ratio)
    
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    # 4. Create the Datasets
    train_dataset = SingleCellDataset(trajectories[train_indices], count_values[train_indices])
    val_dataset = SingleCellDataset(trajectories[val_indices], count_values[val_indices])
    
    # 5. Wrap in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
    