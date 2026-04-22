import torch
import os
import sympy
from kan.utils import ex_round
import numpy as np
import matplotlib.pyplot as plt
from model import build_model
from visualize import load_data, get_plotting_data
from kan.utils import SYMBOLIC_LIB

def torch_sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Prevent sigmoid from being simplified/expanded
class sigmoid(sympy.Function):
    nargs = 1
    @classmethod
    def eval(cls, x):
        return None

def sympy_sigmoid(x):
    return sigmoid(x)

SYMBOLIC_LIB['sigmoid'] = (torch_sigmoid, sympy_sigmoid, 1, lambda x, y: (x, y))

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
    custom_lib = [
        'x', 'x^2', 'x^3', 'x^4', 'x^5', 
        '1/x', '1/x^2', '1/x^3', '1/x^4', '1/x^5', 
        'sqrt', 'x^0.5', 'x^1.5', '1/sqrt(x)', '1/x^0.5', 
        'exp', 'log', 'abs', '0', 'gaussian', 'sgn',
        'sigmoid',
        #'sin', 'cos', 
        'tan', 'tanh', 'arcsin', 'arccos', 'arctan', 'arctanh'
    ]

    model.kan.auto_symbolic(lib=custom_lib, weight_simple=0.8)
    

    # Symbols for input variables
    pt1, pt2, w1, w2 = sympy.symbols('pt1 pt2 w1 w2')
    input_symbols = [pt1, pt2, w1, w2] 

    # Generate the formula
    formulas = model.kan.symbolic_formula(var=input_symbols)
    model.kan.suggest_symbolic(0, 1, 0, lib=custom_lib, topk=5)

    # Extract the formula for mu
    mu_formula = formulas[0][0] # type: ignore
    rounded_mu = ex_round(mu_formula, 1)
    
    print(rounded_mu)

    

def symbolic_pysr(model, counts, pseudotime, weights, gene, model_gene):
    from pysr import PySRRegressor
    
    lineage = 1

    gene_idx = 0 if model_gene is not None else gene

    model.eval()
   
    pt_input, pt_input_scaled, y_pred = get_plotting_data(
        pseudotime, weights, model, gene_idx, lineage
    )


    X_pysr = pt_input_scaled[:, lineage].reshape(-1, 1) # Format for PySR
    
    pysr_model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[
            "exp", 
            "inv(x) = 1/x",
            "sigmoid(x) = 1 / (1 + exp(-x))",
            #"gaussian(x) = exp(-x^2)",
            #"square(x) = x^2",
            #"log1p(x) = log(x + 1)", 
            #"relu(x) = max(0, x)"
        ],
        extra_sympy_mappings={
            "inv": lambda x: 1 / x,
            "sigmoid": lambda x: 1 / (1 + sympy.exp(-x)), 
            #"gaussian": lambda x: sympy.exp(-x**2),
            #"square": lambda x: x**2,
            #"log1p": lambda x: sympy.log(x + 1),
            #"relu": lambda x: sympy.Max(0, x)
        },
        variable_names=["x"],
        model_selection="best",
        random_state=0
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
    #symbolic_pysr(model, counts, pseudotime, weights, gene, model_gene)
