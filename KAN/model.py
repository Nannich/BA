from torch import nn
from kan import KAN as PyKAN
from efficient_kan import KAN as EffKAN

# Hyperparamters
# Efficient-KAN
EFFKAN_HIDDEN_LAYERS = [32]
EFFKAN_GRID_SIZE = 5
EFFKAN_SPLINE_ORDER = 3

# PyKAN
PYKAN_HIDDEN_LAYERS = [1]
PYKAN_GRID_SIZE = 2
PYKAN_SPLINE_ORDER = 3

# MLP
# MLP_HIDDEN_LAYERS = [320, 320, 320]
MLP_HIDDEN_LAYERS = [32]


class ZINB_EFFKAN(nn.Module):
    """
    A wrapper that makes the KAN work with ZINBloss by predicting three parameters per gene:
    - mu: Predicted true biological mean expression
    - theta: Dispersion parameter (variance)
    - pi: Dropout probability (zero-inflation)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        layers = [input_dim] + EFFKAN_HIDDEN_LAYERS + [output_dim]
        self.kan = EffKAN(
            layers,
            grid_size=EFFKAN_GRID_SIZE,
            spline_order=EFFKAN_SPLINE_ORDER,
            # Both lineage weights and pseudtime are scaled to [0, 1]
            # Small buffer dampens the line curving up at the end
            grid_range=[-1.2, 1.2],
        )

    def forward(self, x, update_grid=False):
        raw_output = self.kan(x, update_grid=update_grid)
        
        # Because three parameters are predicted per gene the output dim is n_genes * 3
        # The first third holds all mu values for each gene, the second third all theta
        # values and the third one all pi values
        n_genes = raw_output.shape[1] // 3

        mu =    raw_output[:, :n_genes]
        theta = raw_output[:, n_genes:n_genes*2]
        pi =    raw_output[:, n_genes * 2:]

        return mu, theta, pi


class ZINB_PYKAN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        width = [input_dim] + PYKAN_HIDDEN_LAYERS + [output_dim]
        self.kan = PyKAN(
            width=width,
            grid=PYKAN_GRID_SIZE,
            k=PYKAN_SPLINE_ORDER,
            grid_range=[-1.2, 1.2],
            device="cpu"  
        )

    def forward(self, x, update_grid=False):
        raw_output = self.kan(x)
        
        n_genes = raw_output.shape[1] // 3

        mu =    raw_output[:, :n_genes]
        theta = raw_output[:, n_genes:n_genes*2]
        pi =    raw_output[:, n_genes * 2:]

        return mu, theta, pi


class ZINB_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        layers = []
        current_dim = input_dim
        
        for hidden_dim in MLP_HIDDEN_LAYERS:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        raw_output = self.mlp(x)

        n_genes = raw_output.shape[1] // 3

        mu =    raw_output[:, :n_genes]
        theta = raw_output[:, n_genes:n_genes*2]
        pi =    raw_output[:, n_genes * 2:]

        return mu, theta, pi

def build_model(model_type, input_dim, output_dim):
    if model_type == "effkan":
        return ZINB_EFFKAN(input_dim, output_dim)
    elif model_type == "pykan":
        return ZINB_PYKAN(input_dim, output_dim)
    else:
        return ZINB_MLP(input_dim, output_dim)