import sympy
import torch
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from kan.utils import ex_round, SYMBOLIC_LIB

from src.core.config import MODELS_DIR, RESULTS_DIR, DATA_RAW, ensure_dir
from src.trajectory.zinb_models import build_kan_model
from src.core.preprocessing import run_preprocessing
from src.symbolic.eval_symbolic import evaluate_equations
from src.symbolic.train_symbolic import train_deep_symbolic_kan

# Custom Sigmoid Function Registration for PyKAN Symbolic Library
def torch_sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class sigmoid(sympy.Function):
    nargs = 1
    @classmethod
    def eval(cls, x):
        return None
    def _eval_derivative(self, symbol):
        return self.func(self.args[0]) * (1 - self.func(self.args[0]))
    def _eval_evalf(self, prec):
        arg_val = float(self.args[0].evalf(prec))
        import math
        return sympy.Float(1 / (1 + math.exp(-arg_val)), prec)

def sympy_sigmoid(x):
    return sigmoid(x)

# Register custom sigmoid function to PyKAN global math library
SYMBOLIC_LIB['sigmoid'] = (torch_sigmoid, sympy_sigmoid, 3, lambda x, y: (x, y))


def extract_symbolic_grn(checkpoint, checkpoint_path, dataset_name, tmp_dir, custom_lib, output_fig_path=None, deep_config=None):
    """Handles symbolic formula extraction from GRN models."""
    target_gene = checkpoint["target_gene"]
    predictor_names = checkpoint["predictor_names"]
    loss_mode = checkpoint["loss_mode"]
    X_numpy = checkpoint["X_numpy"]
    Y_numpy = checkpoint["Y_numpy"]

    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)

    in_dim = X_numpy.shape[1]
    hidden_layers = deep_config if deep_config is not None else [in_dim]

    print(f"  Training symbolic KAN: {[in_dim] + hidden_layers + [3 if loss_mode == 'zinb' else 1]}")

    kan_model = train_deep_symbolic_kan(
        X_tensor, Y_tensor, hidden_layers=hidden_layers, ckpt_dir=tmp_dir,
        loss_mode=loss_mode, epochs=350, lr=0.01,
    )
    
    with torch.no_grad():
        kan_model(X_tensor)
        
    final_layer_idx = len(hidden_layers) + 1
    if loss_mode == "zinb":
        kan_model.remove_node(final_layer_idx, 1, mode='down')
        kan_model.remove_node(final_layer_idx, 2, mode='down')
        
    if output_fig_path:
        output_fig_path = Path(output_fig_path)
        ensure_dir(output_fig_path.parent)
        kan_model.plot(folder=str(output_fig_path.parent), beta=3, scale=2.0, in_vars=predictor_names, out_vars=[target_gene])
        plt.savefig(output_fig_path, bbox_inches="tight", dpi=300)
        plt.close()

    arch_dir = checkpoint_path.parent.name
    symbolic_model_dir = ensure_dir(MODELS_DIR / dataset_name / "symbolic" / arch_dir)
    deep_ckpt_path = symbolic_model_dir / f"deep_{checkpoint_path.name}"
    
    torch.save({
        "state_dict": {k: v.cpu() for k, v in kan_model.state_dict().items()},
        "target_gene": target_gene,
        "predictor_names": predictor_names,
        "loss_mode": loss_mode,
        "X_numpy": X_numpy,
        "width": kan_model.width,
        "grid": kan_model.grid,
        "k": kan_model.k,
        "is_symbolic_pruned": True
    }, deep_ckpt_path)
    print(f"  Saved symbolic KAN checkpoint to: {deep_ckpt_path}")

    for param in kan_model.parameters():
        param.requires_grad = False
        
    kan_model.auto_symbolic(lib=custom_lib, weight_simple=0.5, r2_threshold=0.01)
    input_symbols = [sympy.Symbol(name) for name in predictor_names]
    formulas = kan_model.symbolic_formula(var=input_symbols)
    
    return target_gene, str(ex_round(formulas[0][0], 2))


def extract_symbolic_trajectory(checkpoint, checkpoint_path, dataset_name, tmp_dir, custom_lib, dataset=None, output_fig_path=None):
    """Handles symbolic formula extraction from trajectory models."""
    if dataset is None:
        raise ValueError("A SingleCellDataset object must be provided to extract equations from a trajectory checkpoint.")
        
    gene_label = checkpoint["gene_names"][0]
    hidden_layers = checkpoint["hidden_layers"]
    loss_mode = checkpoint["loss"]
    
    X_numpy = dataset.pseudotime
    n_lineages = X_numpy.shape[1]
    
    wrapper_model = build_kan_model(n_lineages, 1, hidden_layers)
    wrapper_model.load_state_dict(checkpoint["state_dict"])
    
    kan_model = wrapper_model.kan
    kan_model.ckpt_path = tmp_dir
    
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    
    wrapper_model.eval()
    with torch.no_grad():
        wrapper_model(X_tensor)
        
    if loss_mode == "zinb":
        final_layer_idx = len(kan_model.width) - 1
        kan_model.remove_node(final_layer_idx, 1, mode='down')
        kan_model.remove_node(final_layer_idx, 2, mode='down')
        
    kan_model = kan_model.prune()
    kan_model.prune_edge(threshold=0.05)
    
    if output_fig_path:
        output_fig_path = Path(output_fig_path)
        ensure_dir(output_fig_path.parent)
        input_names = [f"pt{i+1}" for i in range(n_lineages)]
        kan_model.plot(folder=str(output_fig_path.parent), beta=3, scale=2.0, in_vars=input_names, out_vars=[gene_label])
        plt.savefig(output_fig_path, bbox_inches="tight", dpi=300)
        plt.close()

    arch_dir = checkpoint_path.parent.name
    symbolic_model_dir = ensure_dir(MODELS_DIR / dataset_name / "symbolic" / arch_dir)
    deep_ckpt_path = symbolic_model_dir / f"deep_{checkpoint_path.name}"
    
    torch.save({
        "state_dict": {k: v.cpu() for k, v in kan_model.state_dict().items()},
        "gene_names": [gene_label],
        "loss": loss_mode,
        "width": kan_model.width,
        "grid": kan_model.grid,
        "k": kan_model.k,
        "is_symbolic_pruned": True
    }, deep_ckpt_path)
    print(f"  Saved pruned symbolic trajectory KAN checkpoint to: {deep_ckpt_path}")

    for param in kan_model.parameters():
        param.requires_grad = False
        
    kan_model.auto_symbolic(lib=custom_lib, weight_simple=0.5, r2_threshold=0.01)
    input_symbols = [sympy.Symbol(f"pt{i+1}") for i in range(n_lineages)]
    formulas = kan_model.symbolic_formula(var=input_symbols)
    
    return gene_label, str(ex_round(formulas[0][0], 2))


def extract_symbolic_from_checkpoint(checkpoint_path, dataset=None, output_fig_path=None, deep_config=None):
    """
    Loads the checkpoint, extracts data and calls either the grn or trajectory symbolic KAN train function.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    is_grn = "predictor_names" in checkpoint
    custom_lib = ['x', 'x^2', '1/x', '1/x^2', 'sqrt', 'exp', 'log', '0', 'sigmoid']

    dataset_name = checkpoint_path.parents[2].name if is_grn else checkpoint_path.parents[1].name

    with tempfile.TemporaryDirectory() as tmp_dir:
        if is_grn:
            return extract_symbolic_grn(
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                dataset_name=dataset_name,
                tmp_dir=tmp_dir,
                custom_lib=custom_lib,
                output_fig_path=output_fig_path,
                deep_config=deep_config
            )
        else:
            return extract_symbolic_trajectory(
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                dataset_name=dataset_name,
                tmp_dir=tmp_dir,
                custom_lib=custom_lib,
                dataset=dataset,
                output_fig_path=output_fig_path
            )


def run_symbolic_pipeline(dataset_name, mode="grn", arch_name="smo_log_l"):
    """
    Processes checkpoints, builds (deep) models for more accurate symbolic formula extraction
    and saves extracted equations.
    """
    print(f"Symbolic Formula Extraction: {dataset_name} ({mode} | {arch_name})")
    
    dataset_obj = run_preprocessing(dataset_name=dataset_name) if mode == "trajectory" else None
    
    target_model_dir = MODELS_DIR / dataset_name / mode
    if mode == "grn":
        target_model_dir = target_model_dir / arch_name
        
    checkpoint_paths = sorted(list(target_model_dir.glob("*_checkpoint.pth"))) if mode == "grn" else \
                       sorted(list(target_model_dir.glob("kan_*_mse.pth")))
                       
    if not checkpoint_paths:
        print(f"No checkpoint found at: {target_model_dir}")
        return

    formulas_out_dir = ensure_dir(RESULTS_DIR / "symbolic" / dataset_name / "formulas")
    file_token = arch_name if mode == "grn" else "trajectory"
    eq_csv_path = formulas_out_dir / f"{file_token}_equations.csv"
    
    output_fig_dir = ensure_dir(RESULTS_DIR / "figures" / dataset_name / "symbolic" / file_token)
    extracted_records = []

    for cp_path in checkpoint_paths:
        try:
            gene_symbol = cp_path.name.split("_checkpoint")[0] if mode == "grn" else \
                          cp_path.name.split("kan_")[1].split("_mse")[0]
            
            fig_out_path = output_fig_dir / f"{gene_symbol}_symbolic_graph.png"
            
            print(f" Processing checkpoint weights for gene: {gene_symbol}...")
            gene_name, formula = extract_symbolic_from_checkpoint(
                checkpoint_path=cp_path, 
                dataset=dataset_obj, 
                output_fig_path=fig_out_path,
                deep_config=[1]
            )
            
            extracted_records.append({
                "TargetGene": gene_name,
                "Equation": formula
            })
            
        except Exception as err:
            print(f"  Error {cp_path.name}: {err}")
            continue

    df_equations = pd.DataFrame(extracted_records)
    df_equations.to_csv(eq_csv_path, index=False)
    print(f"Saved equations at: {eq_csv_path}")

    # Evaluate signs on ground-truth
    matched_dirs = list(DATA_RAW.rglob(f"**/{dataset_name}/ExpressionData.csv"))
    if not matched_dirs:
        print(f"Evaluation skipped: Dataset details for '{dataset_name}' missing from raw data paths.")
        return
        
    final_gt_path = None
    for parent in matched_dirs[0].parents:
        check_path = parent / "GroundTruthNetwork.csv"
        if check_path.exists():
            final_gt_path = check_path
            break
        if parent == DATA_RAW:
            break

    if not final_gt_path:
        print(f"Evaluation skipped: No GroundTruthNetwork.csv found for {dataset_name}.")
        return

    eval_out_dir = ensure_dir(RESULTS_DIR / "symbolic" / dataset_name / "eval")
    
    print(f"Using ground truth: {final_gt_path.name} for Evaluation")
    evaluate_equations(
        eq_csv_path=eq_csv_path,
        ground_truth_path=final_gt_path,
        eval_out_dir=eval_out_dir,
        file_token=file_token
    )