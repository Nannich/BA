import torch
from torch import nn
from kan import KAN

def activate_zinb_output(raw_output, n_genes):
    """
    Clamps and activates raw high-dimensional network layers into 
    clean biological metrics (mu, theta, pi) to avoid overflow.
    """
    # Clamp raw network layers to prevent exponential overflow/underflow
    raw_mu = torch.clamp(raw_output[:, :n_genes], min=-10.0, max=12.0)
    raw_theta = torch.clamp(raw_output[:, n_genes:n_genes*2], min=-10.0, max=12.0)
    raw_pi = raw_output[:, n_genes*2:]

    # Natively activate outputs into parameter profiles
    mu = torch.exp(raw_mu)
    theta = torch.exp(raw_theta)
    pi = torch.sigmoid(raw_pi)

    return mu, theta, pi


class ZINB_KAN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, grid_size, spline_order):
        super().__init__()
        width = [input_dim] + list(hidden_layers) + [output_dim]
        self.kan = KAN(width=width, grid=grid_size, k=spline_order, device="cpu", auto_save=False)

    def forward(self, x, update_grid=False):
        if update_grid and hasattr(self.kan, "update_grid_from_samples"):
            self.kan.update_grid_from_samples(x)

        raw_output = self.kan(x)
        n_genes = raw_output.shape[1] // 3
        return activate_zinb_output(raw_output, n_genes)


class ZINB_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        raw_output = self.mlp(x)
        n_genes = raw_output.shape[1] // 3
        return activate_zinb_output(raw_output, n_genes)
    

class NullModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        n_genes = output_dim // 3
        self.mu_bias = nn.Parameter(torch.zeros(1, n_genes))
        self.theta_bias = nn.Parameter(torch.zeros(1, n_genes))
        self.pi_bias = nn.Parameter(torch.zeros(1, n_genes))

    def forward(self, x):
        batch_size = x.shape[0]
        raw_output = torch.cat([
            self.mu_bias.expand(batch_size, -1),
            self.theta_bias.expand(batch_size, -1),
            self.pi_bias.expand(batch_size, -1)
        ], dim=1)
        n_genes = self.mu_bias.shape[1]
        return activate_zinb_output(raw_output, n_genes)
    

def build_kan_model(input_features, output_features, hidden_layers, grid_size=3, spline_order=3):
    """Constructs ZINB KAN.
    Args:
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        hidden_layers (list of int): Specifies node distribution per hidden layer.
        grid_size (int): Initial grid size.
        spline_order (int): Polynomial degree of the splines.

    Returns:
        ZINB_KAN: An initialized instance of the ZINB KAN.
    """
    output_dim = output_features * 3
    return ZINB_KAN(
        input_dim=input_features,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        grid_size=grid_size,
        spline_order=spline_order
    )


def build_mlp_model(input_features, output_features, hidden_layers):
    """Constructs a fully connected ZINB Multi-Layer Perceptron (MLP).
    Args:
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        hidden_layers (list of int): Specifies node distribution per hidden layer.
    Returns:
        ZINB_MLP: An initialized instance of the ZINB MLP.
    """
    output_dim = output_features * 3
    return ZINB_MLP(
        input_dim=input_features,
        output_dim=output_dim,
        hidden_layers=hidden_layers
    )


def build_null_model(output_features):
    """Constructs a static, bias-only ZINB tracking baseline model.
    Args:
        output_features (int): Number of output features.
    Returns:
        NullModel: An initialized instance of the bias-only baseline model.
    """
    output_dim = output_features * 3
    return NullModel(output_dim=output_dim)