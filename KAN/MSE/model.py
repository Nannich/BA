from efficient_kan import KAN
from torch import nn
import torch.nn.functional as F

HIDDEN_LAYERS = [4] 
GRID_SIZE = 5
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
        self.kan = KAN(layers, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER)

    def forward(self, x):
        raw_output = self.kan(x)
        
        # Softplus forces the value to be positive
        return F.softplus(raw_output)

def build_kan(input_dim, output_dim):
    return PositiveKAN(input_dim, output_dim)