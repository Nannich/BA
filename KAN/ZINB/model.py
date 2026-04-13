from efficient_kan import KAN
from torch import nn

HIDDEN_LAYERS = [8] 
GRID_SIZE = 5
SPLINE_ORDER = 3

class ZINBKAN(nn.Module):
    """
    A wrapper that makes the KAN work with ZINBloss by predicting three parameters per gene:
    - mu: Predicted true biological mean expression
    - theta: Dispersion parameter (variance)
    - pi: Dropout probability (zero-inflation)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Initialize the standard KAN
        layers = [input_dim] + HIDDEN_LAYERS + [output_dim]
        self.kan = KAN(
            layers, 
            grid_size=GRID_SIZE, 
            spline_order=SPLINE_ORDER,
            grid_range=[0, 1]  # Both lineage weights and pseudtime are scaled to [0, 1]
        )

    def forward(self, x):
        raw_output = self.kan(x)
        
        # Because three parameters are predicted per gene the output dim is n_genes * 3
        # The first third holds all mu values for each gene, the second third all theta
        # values and the third one all pi values
        n_genes = raw_output.shape[1] // 3

        mu =    raw_output[:, :n_genes]
        theta = raw_output[:, n_genes:n_genes*2]
        pi =    raw_output[:, n_genes * 2:]

        return mu, theta, pi

def build_kan(input_dim, output_dim):
    return ZINBKAN(input_dim, output_dim)