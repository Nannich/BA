import torch
import torch.nn as nn
import torch.optim as optim
from kan import KAN as PyKAN
import pandas as pd
import numpy as np
import sympy
import re
from kan.utils import ex_round, SYMBOLIC_LIB
import matplotlib.pyplot as plt
from pathlib import Path

from src.model import build_model
from src.utils import *
from src.config import DATA_RAW, TABLES_DIR, ensure_dir

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

SYMBOLIC_LIB['sigmoid'] = (torch_sigmoid, sympy_sigmoid, 3, lambda x, y: (x, y))


def train_symbolic_kan(X_tensor, Y_tensor, hidden_layers=[1], epochs=400, lr=0.01, lamb_l1=0.00, grid=3, k=3):
    """
    Trains a KAN specifically for symbolic formula extraction.
    """
    if hidden_layers is None:
        hidden_layers = []
        
    in_dim = X_tensor.shape[1]
    width = [in_dim] + hidden_layers + [1]
    
    model = PyKAN(
        width=width, 
        grid=grid, 
        k=k,
        device="cpu", 
        auto_save=False
    )
    model.update_grid_from_samples(X_tensor)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        mse_loss = criterion(predictions, Y_tensor)
        
        # L1 regularization strips away weak connections
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                
        loss = mse_loss + (lamb_l1 * l1_loss)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def extract_trajectory_symbolic(model, pseudotime, weights, fig_path, pt_min, pt_max):
    """
    Uses pykans built in functions to extract a symbolic formula from the KAN.
    """
    pt_scaled = (pseudotime - pt_min) / (pt_max - pt_min + 1e-8)
    trajectories = np.hstack((pt_scaled, weights))
    
    # Populate model
    model.eval()
    X_sample = torch.tensor(trajectories, dtype=torch.float32)
    
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

    # Save tracking file safely to parent string directory required by KAN plot internal
    model.kan.plot(
        folder=str(fig_path.parent), 
        beta=3, 
        scale=2.0, 
        varscale=0.33,
        in_vars=input_names, 
        out_vars=output_names,
    )
    
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Run symbolic regression to replace splines with math functions
    custom_lib = ['x', 'x^2', '1/x', '1/x^2', 'sqrt', 'exp', 'log', '0', 'sigmoid']
    model.kan.auto_symbolic(lib=custom_lib, weight_simple=0.5, r2_threshold=0.01)

    # Generate the formula
    pt1, pt2, w1, w2 = sympy.symbols('pt1 pt2 w1 w2')
    input_symbols = [pt1, pt2, w1, w2] 
    formulas = model.kan.symbolic_formula(var=input_symbols)

    mu_formula = formulas[0][0] # type: ignore
    rounded_mu = ex_round(mu_formula, 1)
    print(rounded_mu)

def extract_symbolic_from_model(adata, dataset, base_model_name, run_model_dir, run_fig_dir, target_gene=None):
    """ Trains / loads models and extracts equations. Returns output text block. """
    gene_names = list(adata.var_names)
    n_genes = len(gene_names)
    
    kan_dir = run_model_dir / "grn_models"
    if not kan_dir.exists():
        print(f"Error: No GRN models found at {kan_dir}. Run 'grn' first.")
        return ""

    out_fig_dir = ensure_dir(run_fig_dir / "symbolic" / "from_model")
    custom_lib = ['x', 'x^2', '1/x', '1/x^2', 'sqrt', 'exp', 'log', '0', 'sigmoid']
    genes_to_process = [target_gene] if target_gene and target_gene != "all" else gene_names

    # Accumulate equations in memory
    output_lines = [
        f"Symbolic Gene Interactions: {dataset} ({base_model_name})",
        "=" * 60,
        ""
    ]

    for target_g in genes_to_process:
        model_path = kan_dir / f"{target_g}_kan.pth"
        if not model_path.exists(): 
            continue
            
        print(f"Extracting symbolic formula for {target_g}...")
        checkpoint = torch.load(model_path, weights_only=False)
        grid_size = checkpoint.get("grid", 3)
        
        kan_model = PyKAN(width=[n_genes - 1, 1], grid=grid_size, k=3, device="cpu", auto_save=False)
        kan_model.load_state_dict(checkpoint["state_dict"])
        
        X_tensor = torch.tensor(checkpoint["X_numpy"], dtype=torch.float32)
        kan_model.eval()
        with torch.no_grad():
            kan_model(X_tensor)
        
        # Plotting remains safe here because it goes to its specialized runtime fig dir
        fig_path = out_fig_dir / f"{target_g}_interaction.png"
        input_gene_names = [name for name in gene_names if name != target_g]
        kan_model.plot(folder=str(out_fig_dir), beta=3, scale=2.0, in_vars=input_gene_names, out_vars=[target_g])
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()

        kan_model.auto_symbolic(lib=custom_lib, weight_simple=0.5, r2_threshold=0.01)
        input_symbols = [sympy.Symbol(name) for name in input_gene_names]
        formulas = kan_model.symbolic_formula(var=input_symbols)
        rounded_formula = ex_round(formulas[0][0], 2)
        
        output_lines.append(f"{target_g} = {rounded_formula}\n")

    return "\n".join(output_lines)


def convert_grn_csv_to_edge_list(grn_csv_path: Path, output_dir: Path = None, threshold: float = 0.0) -> pd.DataFrame:
    """
    Converts the adjacency matrix to the BEELINE like list of edges.
    """
    df = pd.read_csv(grn_csv_path, index_col=0)
    edges = []
    
    for gene1 in df.index:
        for gene2 in df.columns:
            weight = df.loc[gene1, gene2]
            if abs(weight) > threshold:
                edge_type = "+" if weight > 0 else "-"
                edges.append({"Gene1": gene1, "Gene2": gene2, "Type": edge_type})
                
    final_out_dir = ensure_dir(output_dir if output_dir is not None else TABLES_DIR / "grn_list")
    out_name = grn_csv_path.stem.replace("_grn", "_edges") + ".csv"
    out_path = final_out_dir / out_name
    
    edge_df = pd.DataFrame(edges)
    edge_df.to_csv(out_path, index=False)
    print(f"Saved edge list to: {out_path}")
    
    return edge_df

def prep_tensors_for_target(adata, target_gene, predictor_genes):
    """Extracts expression matrices and converts them to PyTorch tensors."""
    gene_names = list(adata.var_names)
    expression_matrix = np.log1p(get_raw_counts(adata))
    
    target_idx = gene_names.index(target_gene)
    predictor_indices = [gene_names.index(p) for p in predictor_genes]
    
    Y_numpy = expression_matrix[:, [target_idx]]
    X_numpy = expression_matrix[:, predictor_indices]
    
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
    
    return X_tensor, Y_tensor


def extract_symbolic_from_edges(adata, edges_df, dataset, run_fig_dir):
    """ Trains KAN models on mapped edges. Returns output text block. """
    out_fig_dir = ensure_dir(run_fig_dir / "symbolic" / "from_edges")
    custom_lib = ['x', '1/x', '1/x^2', 'sqrt', 'exp', 'log', '0', 'sigmoid']
    gene_names = list(adata.var_names)
    targets = edges_df['Gene2'].unique()
    
    output_lines = [
        f"Symbolic Gene Interactions from Edges: {dataset}",
        "=" * 60,
        ""
    ]

    for target_g in targets:
        predictors = list(set(edges_df[edges_df['Gene2'] == target_g]['Gene1'].tolist()))
        valid_predictors = [p for p in predictors if p in gene_names and p != target_g]
        
        if target_g not in gene_names or not valid_predictors:
            print(f"Skipping '{target_g}': No valid predictors (or only self-edges).")
            continue
            
        X_tensor, Y_tensor = prep_tensors_for_target(adata, target_g, valid_predictors)
        print(f"Training Symbolic KAN for target '{target_g}' with predictors: {valid_predictors}...")
        
        kan_model = train_symbolic_kan(X_tensor, Y_tensor)
        with torch.no_grad():
            kan_model(X_tensor)
        
        fig_path = out_fig_dir / f"{target_g}_interaction.png"
        kan_model.plot(folder=str(out_fig_dir), beta=3, scale=2.0, in_vars=valid_predictors, out_vars=[target_g])
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        kan_model.auto_symbolic(lib=custom_lib, weight_simple=0.5, r2_threshold=0.01)
        
        input_symbols = [sympy.Symbol(name) for name in valid_predictors]
        formulas = kan_model.symbolic_formula(var=input_symbols)
        rounded_formula = ex_round(formulas[0][0], 2)
        
        output_lines.append(f"{target_g} = {rounded_formula}\n")
            
    return "\n".join(output_lines)

class sigmoid(sympy.Function):
    nargs = 1
    def _eval_derivative(self, symbol):
        return self.func(self.args[0]) * (1 - self.func(self.args[0]))
        
    def _eval_evalf(self, prec):
        arg_val = float(self.args[0].evalf(prec))
        import math
        return sympy.Float(1 / (1 + math.exp(-arg_val)), prec)

def evaluate(eq_file_path: Path, ground_truth_path: Path, output_report_path: Path):
    if not eq_file_path.exists() or not ground_truth_path.exists():
        return

    gt_df = pd.read_csv(ground_truth_path)
    gt_df.columns = [col.capitalize() for col in gt_df.columns]
    gt_edges = {(str(row['Gene1']), str(row['Gene2'])): str(row['Type']) for _, row in gt_df.iterrows()}

    with open(eq_file_path, "r") as f:
        lines = f.readlines()

    summary_data = []
    text_blocks = []

    for line in lines:
        if "=" not in line or line.startswith("="):
            continue
            
        parts = line.split("=")
        target_gene = parts[0].strip()
        expr_string = parts[1].strip()
        
        expr_string = re.sub(r'([\d\)])([a-zA-Z])', r'\1*\2', expr_string)
        
        gene_block = [
            f"Target Gene: {target_gene}",
            f"Equation: {target_gene} = {expr_string}"
        ]
        has_edges = False
        
        try:
            local_dict = {"sigmoid": sigmoid}
            parsed_expr = sympy.parse_expr(expr_string, local_dict=local_dict)
            predictor_symbols = [s for s in parsed_expr.free_symbols if s.name != 'True']
            
            for symbol in predictor_symbols:
                pred_gene = symbol.name
                
                if (pred_gene, target_gene) not in gt_edges:
                    continue
                    
                true_sign = gt_edges[(pred_gene, target_gene)]
                has_edges = True
                
                # Expose the symbolic custom sigmoid class cleanly to NumPy lambda layers
                free_vars = list(parsed_expr.free_symbols)
                modules_dict = {"sigmoid": lambda x: 1 / (1 + np.exp(-x))}
                fast_numpy_func = sympy.lambdify(free_vars, parsed_expr, modules=[modules_dict, "numpy"])

                grid_size = 50
                sampled_points = np.linspace(0.0, 2.0, grid_size)

                input_matrix = []
                for var in free_vars:
                    if var == symbol:
                        input_matrix.append(sampled_points)
                    else:
                        input_matrix.append(np.ones(grid_size))

                grid_outputs = fast_numpy_func(*input_matrix)

                global_slope, _ = np.polyfit(sampled_points, grid_outputs, 1)

                pred_sign = "+" if global_slope > 0 else "-"
                is_correct = (pred_sign == true_sign)
                status_str = "CORRECT" if is_correct else "WRONG"
                
                gene_block.append(
                    f"  -> {pred_gene} : True [{true_sign}] | Pred [{pred_sign}] => {status_str} (Delta: {global_slope:.3f})"
                )
                summary_data.append({"true": true_sign, "correct": is_correct})
                
        except Exception as e:
            gene_block.append(f"  Error parsing equation behavior: {e}")

        if has_edges:
            text_blocks.append("\n".join(gene_block))

    if not summary_data:
        return

    total_checked = len(summary_data)
    total_correct = sum(1 for e in summary_data if e["correct"])
    
    act_total = sum(1 for e in summary_data if e["true"] == "+")
    act_corr = sum(1 for e in summary_data if e["true"] == "+" and e["correct"])
    
    rep_total = sum(1 for e in summary_data if e["true"] == "-")
    rep_corr = sum(1 for e in summary_data if e["true"] == "-" and e["correct"])

    output = []
    output.append("SYMBOLIC INTERACTION VALIDATION REPORT")
    output.append(f"Dataset: {output_report_path.parts[-3]}")
    output.append(f"Profile: {output_report_path.parts[-2]}\n")
    output.append(f"Total Checked Edges: {total_checked}")
    output.append(f"Overall Accuracy: {total_correct / total_checked:.1%} ({total_correct}/{total_checked})")
    
    val_act = f"{act_corr / act_total:.1%} ({act_corr}/{act_total})" if act_total else "N/A"
    output.append(f"Activators (+) Accuracy: {val_act}")
    
    val_rep = f"{rep_corr / rep_total:.1%} ({rep_corr}/{rep_total})" if rep_total else "N/A"
    output.append(f"Repressors (-) Accuracy: {val_rep}\n")
    output.append("GENE EQUATIONS AND CLASSIFICATIONS\n")
    output.append("\n\n".join(text_blocks))

    ensure_dir(output_report_path.parent)
    with open(output_report_path, "w") as f:
        f.write("\n".join(output))


USE_GROUND_TRUTH = True

def run_extraction(args, adata, pseudotime, weights, run_model_dir, run_fig_dir):
    experiment_name = getattr(args, 'experiment_name')
    dataset = args.dataset

    out_res_dir = ensure_dir(TABLES_DIR.parent / "symbolic" / dataset / experiment_name)
    eq_file_path = out_res_dir / "symbolic_equations.txt"
    report_file_path = out_res_dir / "evaluation.txt"

    if USE_GROUND_TRUTH:
        matched_dirs = list(DATA_RAW.rglob(f"**/{dataset}/ExpressionData.csv"))
        if not matched_dirs:
            print(f"Could not locate dataset folder {dataset} under {DATA_RAW}")
            return
            
        edge_file_path = None
        for parent in matched_dirs[0].parents:
            check_path = parent / "GroundTruthNetwork.csv"
            if check_path.exists():
                edge_file_path = check_path
                break
            if parent == DATA_RAW:
                break

        if not edge_file_path or not edge_file_path.exists():
            print(f"GroundTruthNetwork.csv file not found above: {matched_dirs[0]}")
            return
            
        edges_df = pd.read_csv(edge_file_path)
        
        equations_text = extract_symbolic_from_edges(adata, edges_df, dataset, run_fig_dir)
        
    else:
        base_inputs = DATA_RAW / "BEELINE-data" / "inputs"
        matched_dirs = list(base_inputs.rglob(f"**/{dataset}/ExpressionData.csv"))
        
        grn_matrix_path = TABLES_DIR / dataset / experiment_name / "grn" / f"{dataset}_grn.csv"
        
        if grn_matrix_path.exists():
            print(f"GRN matrix at: {grn_matrix_path}")
            edges_df = convert_grn_csv_to_edge_list(grn_matrix_path)
            equations_text = extract_symbolic_from_edges(adata, edges_df, dataset, run_fig_dir)
        else:
            equations_text = extract_symbolic_from_model(
                adata=adata, 
                dataset=dataset, 
                experiment_name=experiment_name, 
                run_model_dir=run_model_dir, 
                run_fig_dir=run_fig_dir, 
                target_gene=args.gene
            )

    if equations_text:
        with open(eq_file_path, "w") as f:
            f.write(equations_text)
        print(f"Equations file saved to: {eq_file_path}")
        
        matched_dirs = list(DATA_RAW.rglob(f"**/{dataset}/ExpressionData.csv"))
        if matched_dirs:
            final_gt_path = None
            for parent in matched_dirs[0].parents:
                check_path = parent / "GroundTruthNetwork.csv"
                if check_path.exists():
                    final_gt_path = check_path
                    break
                if parent == DATA_RAW:
                    break
                    
            if final_gt_path:
                evaluate(eq_file_path, final_gt_path, report_file_path)