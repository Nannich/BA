import torch
import os
import sympy
from kan.utils import ex_round
import numpy as np
import matplotlib.pyplot as plt
from model import build_model
from visualize import load_data, get_plotting_data


def symbolic_pykan(model, gene, counts, pseudotime, weights, sim, fig_path):
    pt_values = pseudotime.values
    pt_min = pt_values.min(axis=0, keepdims=True)
    pt_max = pt_values.max(axis=0, keepdims=True)
    pt_scaled = (pt_values - pt_min) / (pt_max - pt_min + 1e-8)

    trajectories = np.hstack((pt_scaled, weights.values))
    
    # Populate model
    X_sample = torch.tensor(trajectories[:100], dtype=torch.float32)
    
    with torch.no_grad():
        model(X_sample)

    # Pruning
    model.kan.remove_node(2, 1, mode='down')
    model.kan.remove_node(2, 2, mode='down')

    model.kan = model.kan.prune()
    model.kan.prune_edge(threshold=0.05)

    # Plotting the KAN
    n_lineages = weights.shape[1]
    input_names = [f"pt{i+1}" for i in range(n_lineages)] + [f"w{i+1}" for i in range(n_lineages)]
    
    output_names = ["$\\mu$", "$\\theta$", "$\\pi$"]

    model.kan.plot(
        folder='./figures', 
        beta=3, 
        scale=2.0, 
        varscale=0.33,
        in_vars=input_names, 
        out_vars=output_names,
    )
    
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Run symbolic regression to replace splines with math functions
    model.kan.auto_symbolic()

    # Symbols for input variables
    pt1, pt2, w1, w2 = sympy.symbols('pt1 pt2 w1 w2')
    input_symbols = [pt1, pt2, w1, w2] 

    # Generate the formula
    formulas = model.kan.symbolic_formula(var=input_symbols)

    # Extract the formula for mu
    mu_formula = formulas[0][0] # type: ignore
    rounded_mu = ex_round(mu_formula, 1)
    
    print(rounded_mu)

    

def symbolic_pysr(model, counts, pseudotime, weights, gene):
    from pysr import PySRRegressor
    
    lineage = 1

    model.eval()
   
    X_grid, y_pred = get_plotting_data(
        pseudotime, weights, model, gene, lineage
    )

    X_pysr = X_grid.reshape(-1, 1) # Format for PySR
    
    pysr_model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[
            "exp", 
            "inv(x) = 1/x",
            "sigmoid(x) = 1 / (1 + exp(-x))"  
        ],
        extra_sympy_mappings={
            "inv": lambda x: 1 / x,
            "sigmoid": lambda x: 1 / (1 + sympy.exp(-x)) # type: ignore
        },

        model_selection="best",
    )

    pysr_model.fit(X_pysr, y_pred)


def run_extraction(args):
    sim = args.sim
    gene = args.gene
    data_dir = args.data_dir
    model_dir = args.model_dir
    fig_dir = args.fig_dir
    model_name = args.name
    dataset = args.dataset

    data_path = os.path.join(data_dir, dataset, f"sim_{sim}")
    model_path = os.path.join(model_dir, dataset, model_name)
    
    checkpoint = torch.load(model_path, weights_only=False)
    model_type = checkpoint ["model"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    model_gene = checkpoint["gene"]

    fig_path = os.path.join(fig_dir, "symbolic", dataset, f"sim{sim}_gene{gene}.png")

    counts, pseudotime, weights = load_data(data_path)
    
    model = build_model(model_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    symbolic_pykan(model, model_gene, counts, pseudotime, weights, sim, fig_path)

