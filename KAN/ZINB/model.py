from efficient_kan import KAN
from torch import nn
import torch.nn.functional as F

HIDDEN_LAYERS = [64] 
GRID_SIZE = 10
SPLINE_ORDER = 3

class PositiveKAN(nn.Module):
    """
    A wrapper that forces the KAN to output strictly positive values
    becaue gene expression can not be less than zero in reality.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Initialize the standard KAN
        layers = [input_dim] + HIDDEN_LAYERS + [output_dim]
        self.kan = KAN(
            layers, 
            grid_size=GRID_SIZE, 
            spline_order=SPLINE_ORDER,
            grid_range=[0, 1]  # Both weights and pt are scaled to [0, 1]
        )

    def forward(self, x):
        raw_output = self.kan(x)
        
        n_genes = len(raw_output[0]) // 3

        mu =    raw_output[:, :n_genes]
        theta = raw_output[:, n_genes:n_genes*2]
        pi =    raw_output[:, n_genes * 2:]

        return F.softplus(mu), F.softplus(theta), F.sigmoid(pi)

def build_kan(input_dim, output_dim):
    return PositiveKAN(input_dim, output_dim)