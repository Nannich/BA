import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F

from src.trajectory.zinb_models import build_kan_model, build_mlp_model, build_null_model
from src.trajectory.zinb_loss import ZINBLoss

def calculate_mse_per_curve(model, X, Y, lineage_mask):
    """
    Calculates the Mean Squared Error per gene, isolated each lineage.
    """
    model.eval()
    
    with torch.no_grad():
        mu, _, pi_logits = model(X)
        
        pi = torch.sigmoid(pi_logits)
        expected_value = (1.0 - pi) * mu
        
        y_true_log1p = torch.log1p(Y.float())
        y_pred_log1p = torch.log1p(expected_value)
        
        sq_err = F.mse_loss(y_pred_log1p, y_true_log1p, reduction='none')

        n_genes = sq_err.shape[1]
        n_lineages = lineage_mask.shape[1]

        total_squared_error = torch.zeros((n_genes, n_lineages), device=X.device)
        samples_per_lineage = torch.zeros(n_lineages, device=X.device)

        # Slice along lineage
        for l in range(n_lineages):
            l_mask = lineage_mask[:, l]
            n_active = l_mask.sum()
            
            if n_active > 0:
                total_squared_error[:, l] = sq_err[l_mask].sum(dim=0)
                samples_per_lineage[l] = n_active

        # Calculate localized mean values
        mse_per_curve = total_squared_error / (samples_per_lineage + 1e-10)

    return mse_per_curve


def calculate_nll_per_gene(model, X, Y, loss_fn):
    """
    Computes the Negative Log-Likelihood accumulated per individual gene.
    """
    model.eval()
    
    with torch.no_grad():
        mu, theta, pi = model(X)
        loss = loss_fn(Y, mu, theta, pi) # Shape: (n_cells, n_genes)
        total_nll = loss.sum(dim=0)
        
    return total_nll


def calculate_aic(n_params, neg_log_likelihood):
    """
    Calculates the Akaike Information Criterion (AIC).
    AIC = 2k + 2 * NLL
    """
    return 2 * n_params + 2 * neg_log_likelihood


def calculate_bic(n_params, neg_log_likelihood, n_samples):
    """
    Calculates the Bayesian Information Criterion (BIC).
    BIC = k * ln(n) + 2 * NLL
    """
    return n_params * np.log(n_samples) + 2 * neg_log_likelihood

def run_eval(args, dataset):
    """
    Benchmarks across all metrics.
    """
    model_dir = Path(args.model_dir)
    checkpoint_paths = list(model_dir.glob("*.pth"))
    
    if not checkpoint_paths:
        print(f"No checkpoint at: {model_dir}")
        return

    evaluation_records = []
    
    # Initialize an loss function instance for per-gene likelihood calculation
    loss_fn_zinb = ZINBLoss(ridge_lambda=0.0, reduction='none')
    
    X = torch.tensor(dataset.pseudotime, dtype=torch.float32)
    lineage_mask = torch.tensor(dataset.lineage_assignment, dtype=torch.bool)
    n_lineages = lineage_mask.shape[1]
    n_samples = X.shape[0]

    for cp_path in checkpoint_paths:
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)

        model_type = checkpoint.get("model_type", "unknown")
        loss_type = checkpoint.get("loss", "unknown")
        target_gene_idx = checkpoint.get("target_gene_idx")
        hidden_layers = checkpoint.get("hidden_layers", [])
        gene_label = checkpoint.get("gene_names", ["unknown"])[0]

        if target_gene_idx is None:
            continue

        Y = torch.tensor(dataset.raw_counts[:, [target_gene_idx]], dtype=torch.float32)

        if model_type == "kan":
            model = build_kan_model(n_lineages, 1, hidden_layers)
        elif model_type == "mlp":
            model = build_mlp_model(n_lineages, 1, hidden_layers)
        elif model_type == "null":
            model = build_null_model(1)
        else:
            continue

        model.load_state_dict(checkpoint["state_dict"])
        
        # Compute MSE across all lineages
        mse_tensor = calculate_mse_per_curve(model, X, Y, lineage_mask)
        mse_np = mse_tensor.flatten().cpu().numpy()
        avg_mse = np.mean(mse_np)
        
        # Sum up number of parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Compute information criteria statistics
        nll_val, aic_val, bic_val = np.nan, np.nan, np.nan
        if loss_type == "zinb":
            total_nll = calculate_nll_per_gene(model, X, Y, loss_fn_zinb)
            nll_val = total_nll.item()
            aic_val = calculate_aic(n_params, nll_val)
            bic_val = calculate_bic(n_params, nll_val, n_samples)

        record = {
            "Gene": gene_label,
            "Model": model_type.upper(),
            "Loss": loss_type.upper(),
            "Parameters": n_params,
            "Final_Train_Loss": checkpoint.get("final_loss", np.nan),
            "Avg_MSE": avg_mse,
            "Negative_LogLikelihood": nll_val,
            "AIC": aic_val,
            "BIC": bic_val
        }
        
        for l in range(n_lineages):
            record[f"MSE_Lineage_{l+1}"] = mse_np[l]
            
        evaluation_records.append(record)

    if not evaluation_records:
        print("No metrics records were generated from folder parsing execution.")
        return

    df_eval = pd.DataFrame(evaluation_records)
    df_eval = df_eval.sort_values(by=["Gene", "Model", "Loss"])
    
    out_csv_path = model_dir / f"{args.dataset}_trajectory_evaluation.csv"
    df_eval.to_csv(out_csv_path, index=False)
    print(f"Saced metrics at: {out_csv_path}")