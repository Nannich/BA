import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from kan import KAN as PyKAN

from src.utils import *
from src.de import *
from src.model import build_model
from src.plotting import *
from src.trajectory import run_trajectory
from src.config import DATA_RAW, TABLES_DIR, EXTERNAL_DIR, ensure_dir
from src.dataloaders import get_eval_dataloader

TRAJECTORY_MODEL_TYPE = "effkan"
N_BINS = 512

def train_n_to_1_kan(X_tensor, Y_tensor, epochs=600, lr=0.01, lamb_l1=0.1):
    in_dim = X_tensor.shape[1]
    model = PyKAN(
        width=[in_dim, 1], 
        grid=3, 
        k=3,
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
        
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                
        loss = mse_loss + (lamb_l1 * l1_loss)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_predictions = model(X_tensor)
        
    return model, final_predictions.detach().numpy()


def get_correlation_signs(X_numpy, Y_numpy):
    signs = np.zeros(X_numpy.shape[1])
    y_flat = Y_numpy.flatten()
    
    for i in range(X_numpy.shape[1]):
        x_col = X_numpy[:, i]
        if np.std(x_col) == 0 or np.std(y_flat) == 0:
            corr = 0
        else:
            corr = np.corrcoef(x_col, y_flat)[0, 1]
        signs[i] = 1 if corr >= 0 else -1
        
    return signs


def train_grn_models(input_matrix, target_matrix, gene_names, save_dir):
    """
    Trains and saves a checkpoint for each gene.
    """
    ensure_dir(save_dir)
    n_samples, n_genes = input_matrix.shape

    epochs = 600
    lamb_l1 = 0.02

    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        model_save_path = save_dir / f"{target_gene}_kan.pth"
                
        print(f"Training Gene {target_idx+1}/{n_genes}: {target_gene}")
        
        all_predictor_names = [name for name in gene_names if name != target_gene]
        predictor_indices = [list(gene_names).index(name) for name in all_predictor_names]
        
        X_numpy = input_matrix[:, predictor_indices]
        Y_numpy = target_matrix[:, [target_idx]]
        
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
        
        kan_model, _ = train_n_to_1_kan(X_tensor, Y_tensor, lr=0.01, lamb_l1=lamb_l1, epochs=epochs)
        
        checkpoint = {
            "state_dict": kan_model.state_dict(),
            "X_numpy": X_numpy,
            "Y_numpy": Y_numpy,
            "grid": kan_model.grid,
            "k": kan_model.k
        }
        torch.save(checkpoint, model_save_path)


def train_grn_models_deep(input_matrix, target_matrix, gene_names, save_dir, hidden_dim=2):
    """ Trains and saves a self-contained deep checkpoint for each gene. """
    ensure_dir(save_dir)
    n_samples, n_genes = input_matrix.shape
    epochs = 600
    lamb_l1 = 0.02

    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        model_save_path = save_dir / f"{target_gene}_kan_deep.pth"
                
        print(f"Deep Training Gene {target_idx+1}/{n_genes}: {target_gene}")
        
        all_predictor_names = [name for name in gene_names if name != target_gene]
        predictor_indices = [list(gene_names).index(name) for name in all_predictor_names]
        
        X_numpy = input_matrix[:, predictor_indices]
        Y_numpy = target_matrix[:, [target_idx]]
        
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
        
        in_dim = X_tensor.shape[1]
        model = PyKAN(
            width=[in_dim, hidden_dim, 1],
            grid=3, 
            k=3,
            device="cpu", 
            auto_save=False
        )
        model.update_grid_from_samples(X_tensor)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            predictions = model(X_tensor)
            mse_loss = criterion(predictions, Y_tensor)
            l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            loss = mse_loss + (lamb_l1 * l1_loss)
            loss.backward()
            optimizer.step()
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "X_numpy": X_numpy,
            "Y_numpy": Y_numpy,
            "grid": model.grid,
            "k": model.k,
            "hidden_dim": hidden_dim
        }
        torch.save(checkpoint, model_save_path)


def extract_grn_matrix(gene_names, save_dir):
    """ Extracts adjacency matrix directly from saved checkpoints. """
    n_genes = len(gene_names)
    adj_matrix = np.zeros((n_genes, n_genes))

    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        model_save_path = save_dir / f"{target_gene}_kan.pth"
        
        checkpoint = torch.load(model_save_path, weights_only=False)
        grid_size = checkpoint.get("grid", 3)
        k_order = checkpoint.get("k", 3)
        
        kan_model = PyKAN(width=[n_genes - 1, 1], grid=grid_size, k=k_order, device="cpu", auto_save=False)
        kan_model.load_state_dict(checkpoint["state_dict"])
        
        X_numpy = checkpoint["X_numpy"]
        Y_numpy = checkpoint["Y_numpy"]
        
        kan_model.eval()
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        with torch.no_grad():
            kan_model(X_tensor)
        
        kan_model.attribute()
        edge_magnitudes = kan_model.edge_scores[0].detach().cpu().numpy().flatten()
        edge_signs = get_correlation_signs(X_numpy, Y_numpy)
        
        edge_weights = edge_magnitudes * edge_signs
        adj_matrix[:, target_idx] = np.insert(edge_weights, target_idx, 0.0)
    
    return adj_matrix


def extract_grn_matrix_deep(gene_names, save_dir):
    """ Extracts adjacency matrix directly from saved deep checkpoints. """
    n_genes = len(gene_names)
    adj_matrix = np.zeros((n_genes, n_genes))

    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        model_save_path = save_dir / f"{target_gene}_kan_deep.pth"
        
        checkpoint = torch.load(model_save_path, weights_only=False)
        grid_size = checkpoint.get("grid", 3)
        k_order = checkpoint.get("k", 3)
        hidden_dim = checkpoint.get("hidden_dim", 2)
        
        X_numpy = checkpoint["X_numpy"]
        Y_numpy = checkpoint["Y_numpy"]
        
        kan_model = PyKAN(width=[n_genes - 1, hidden_dim, 1], grid=grid_size, k=k_order, device="cpu", auto_save=False)
        kan_model.load_state_dict(checkpoint["state_dict"])
        kan_model.eval()
        
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32, requires_grad=True)
        predictions = kan_model(X_tensor)
        
        loss = torch.sum(predictions)
        loss.backward()
        
        gradients = X_tensor.grad.abs().mean(dim=0).detach().cpu().numpy().flatten()
        edge_signs = get_correlation_signs(X_numpy, Y_numpy)
        
        edge_weights = gradients * edge_signs
        adj_matrix[:, target_idx] = np.insert(edge_weights, target_idx, 0.0)
    
    return adj_matrix


def save_beeline_ranked_edges(adj_matrix, gene_names, dataset_name, base_model_name):
    """ Saves ranked edges into the BEELINE dir structure. """
    edges = []
    n_genes = len(gene_names)
    
    for i in range(n_genes):
        for j in range(n_genes):
            if i != j:
                weight = adj_matrix[i, j]
                abs_weight = abs(weight)
                if abs_weight > 0:
                    edges.append({
                        'Gene1': gene_names[i],
                        'Gene2': gene_names[j],
                        'EdgeWeight': abs_weight,
                        'Sign': 1 if weight > 0 else -1
                    })
                    
    df = pd.DataFrame(edges)
    if not df.empty:
        df = df.sort_values(by='EdgeWeight', ascending=False)
        
    base_inputs = DATA_RAW / "BEELINE-data" / "inputs"
    
    root_dataset = dataset_name.split("_lineage_")[0]
    matched_paths = list(base_inputs.rglob(f"**/{root_dataset}/ExpressionData.csv"))
    if not matched_paths:
        print(f"Could not resolve BEELINE input paths for {root_dataset}.")
        return
        
    path_dir = matched_paths[0].parent
    
    if "Synthetic" in path_dir.parts:
        dataset_group = "Synthetic/" + str(path_dir.parent.relative_to(base_inputs / "Synthetic"))
    else:
        pure_group = str(path_dir.parent.relative_to(base_inputs / "Curated"))
        dataset_group = f"Curated/{pure_group}" if pure_group and pure_group != "." else "Curated"
        
    dataset_group = dataset_group.strip("/")

    out_dir = ensure_dir(EXTERNAL_DIR / "Beeline" / "outputs" / dataset_group / dataset_name / base_model_name)
    out_path = out_dir / "rankedEdges.csv"
    
    df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved ranked edges for BEELINE to: {out_path}")


def raw_to_raw_pykan(args, adata, pseudotime, weights):
    raw_counts = np.log1p(get_raw_counts(adata))
    lineage_assignment = get_lineage_assignment(weights)
    n_lineages = weights.shape[1]
    n_genes = raw_counts.shape[1]
    gene_names = adata.var_names.values
    
    raw_matrices = []
    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        if not np.any(mask): 
            continue
        pt_active = pseudotime[mask, l]
        counts_active = raw_counts[mask]
        sort_idx = np.argsort(pt_active)
        raw_matrices.append(counts_active[sort_idx])
        
    expression_matrix = np.vstack(raw_matrices)
    adj_matrix = np.zeros((n_genes, n_genes))
    
    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        
        all_predictor_names = [name for name in gene_names if name != target_gene]
        predictor_indices = [list(gene_names).index(name) for name in all_predictor_names]
        
        X_numpy = expression_matrix[:, predictor_indices]
        Y_numpy = expression_matrix[:, [target_idx]]
        
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
        
        kan_model = PyKAN(
            width=[n_genes - 1, 1], 
            grid=3, 
            k=3, 
            device="cpu", 
            auto_save=False
        )
        kan_model.update_grid_from_samples(X_tensor)
        
        dataset_dict = {
            'train_input': X_tensor, 'train_label': Y_tensor,
            'test_input': X_tensor, 'test_label': Y_tensor
        }
        
        kan_model.fit(
            dataset_dict, 
            opt='Adam', 
            steps=600, 
            lamb=0.01,          
            lamb_l1=1.0,        
            lamb_entropy=2.0,   
            update_grid=True, 
            display_metrics=None
        )
        
        kan_model.attribute(plot=False)
        edge_magnitudes = kan_model.edge_scores[0].detach().cpu().numpy().flatten()
        edge_signs = get_correlation_signs(X_numpy, Y_numpy)
        
        edge_weights = edge_magnitudes * edge_signs
        adj_matrix[:, target_idx] = np.insert(edge_weights, target_idx, 0.0)
        
    return adj_matrix


def smooth_to_smooth_ZINB(args, adata, pseudotime, weights, run_model_dir):
    print("Executings smooth to smooth ZINB")
    dataset = args.dataset

    class TrainArgsNamespace:
        model = TRAJECTORY_MODEL_TYPE
        gene = None 
        dataset = args.dataset
        
    run_trajectory(TrainArgsNamespace(), adata, pseudotime, weights, run_model_dir)
    
    generated_model_name = f"{TRAJECTORY_MODEL_TYPE}_{dataset}_all_zinb.pth"
    model_path = run_model_dir / generated_model_name
    
    checkpoint = torch.load(model_path, weights_only=False)
    pt_min, pt_max = checkpoint["pt_min"], checkpoint["pt_max"]
    architecture_type, input_dim, output_dim = checkpoint["model"], checkpoint["input_dim"], checkpoint["output_dim"]
    
    model = build_model(architecture_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    predictions = predict_lineage_trajectories(pseudotime, weights, model, None, pt_min, pt_max)
    n_genes = output_dim // 3
    dummy_mask = np.ones(n_genes, dtype=bool)
    predictions_de = filter_predictions(predictions, dummy_mask)
    
    predictions_smooth = build_smoothed_cube(predictions_de, n_bins=N_BINS)
    expression_matrix = predictions_smooth.reshape(n_genes, -1).T 
    
    return expression_matrix


def smooth_to_smooth_MSE(args, adata, pseudotime, weights, run_model_dir):
    print("Executing Smooth-to-Smooth MSE")
    dataset = args.dataset

    class TrainArgsNamespace:
        model = TRAJECTORY_MODEL_TYPE
        gene = None 
        dataset = args.dataset
        
    run_trajectory(TrainArgsNamespace(), adata, pseudotime, weights, run_model_dir, loss_type="mse")
    
    generated_model_name = f"{TRAJECTORY_MODEL_TYPE}_{dataset}_all_mse.pth"
    model_path = run_model_dir / generated_model_name
    
    checkpoint = torch.load(model_path, weights_only=False)
    pt_min, pt_max = checkpoint["pt_min"], checkpoint["pt_max"]
    architecture_type, input_dim, output_dim = checkpoint["model"], checkpoint["input_dim"], checkpoint["output_dim"]
    
    model = build_model(architecture_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    predictions = predict_lineage_trajectories(pseudotime, weights, model, None, pt_min, pt_max)
    n_genes = output_dim // 3
    dummy_mask = np.ones(n_genes, dtype=bool)
    predictions_de = filter_predictions(predictions, dummy_mask)
    
    predictions_smooth = build_smoothed_cube(predictions_de, n_bins=N_BINS)
    expression_matrix = predictions_smooth.reshape(n_genes, -1).T 
    
    return expression_matrix


def smooth_to_smooth_MSE_deep(args, adata, pseudotime, weights, run_model_dir):
    print("Executing Deep Smooth-to-Smooth MSE")
    dataset = args.dataset

    class TrainArgsNamespace:
        model = TRAJECTORY_MODEL_TYPE
        gene = None 
        dataset = args.dataset
        
    run_trajectory(TrainArgsNamespace(), adata, pseudotime, weights, run_model_dir, loss_type="mse")
    
    generated_model_name = f"{TRAJECTORY_MODEL_TYPE}_{dataset}_all_mse.pth"
    model_path = run_model_dir / generated_model_name
    
    checkpoint = torch.load(model_path, weights_only=False)
    pt_min, pt_max = checkpoint["pt_min"], checkpoint["pt_max"]
    architecture_type, input_dim, output_dim = checkpoint["model"], checkpoint["input_dim"], checkpoint["output_dim"]
    
    model = build_model(architecture_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    predictions = predict_lineage_trajectories(pseudotime, weights, model, None, pt_min, pt_max)
    n_genes = output_dim // 3
    dummy_mask = np.ones(n_genes, dtype=bool)
    predictions_de = filter_predictions(predictions, dummy_mask)
    
    predictions_smooth = build_smoothed_cube(predictions_de, n_bins=N_BINS)
    expression_matrix = predictions_smooth.reshape(n_genes, -1).T 
    
    return expression_matrix


def smooth_to_raw_ZINB(args, adata, pseudotime, weights, run_model_dir):
    print("Executing Smooth-to-Raw ZINB")
    dataset = args.dataset

    class TrainArgsNamespace:
        model = TRAJECTORY_MODEL_TYPE
        gene = None 
        dataset = args.dataset
        
    run_trajectory(TrainArgsNamespace(), adata, pseudotime, weights, run_model_dir)
    
    generated_model_name = f"{TRAJECTORY_MODEL_TYPE}_{dataset}_all_zinb.pth"
    model_path = run_model_dir / generated_model_name
    
    checkpoint = torch.load(model_path, weights_only=False)
    pt_min, pt_max = checkpoint["pt_min"], checkpoint["pt_max"]
    architecture_type, input_dim, output_dim = checkpoint["model"], checkpoint["input_dim"], checkpoint["output_dim"]
    
    trajectory_model = build_model(architecture_type, input_dim, output_dim)
    trajectory_model.load_state_dict(checkpoint["state_dict"])
    
    eval_dataloader = get_eval_dataloader(adata, pseudotime, weights, pt_min, pt_max, batch_size=256)
    mu_list = []
    trajectory_model.eval()
    with torch.no_grad():
        for X_b, _ in eval_dataloader:
            mu_b, _, _ = trajectory_model(X_b)
            mu_list.append(mu_b.cpu().numpy())
            
    smooth_matrix = np.vstack(mu_list)
    smooth_matrix = np.exp(smooth_matrix)
    smooth_matrix = np.log1p(smooth_matrix)

    raw_matrix = np.log1p(get_raw_counts(adata))
    return smooth_matrix, raw_matrix


def smooth_to_raw_MSE(args, adata, pseudotime, weights, run_model_dir):
    print("Executing Smooth-to-Raw MSE")
    dataset = args.dataset

    class TrainArgsNamespace:
        model = TRAJECTORY_MODEL_TYPE
        gene = None 
        dataset = args.dataset
        
    run_trajectory(TrainArgsNamespace(), adata, pseudotime, weights, run_model_dir, loss_type="mse")
    
    generated_model_name = f"{TRAJECTORY_MODEL_TYPE}_{dataset}_all_mse.pth"
    model_path = run_model_dir / generated_model_name
    
    checkpoint = torch.load(model_path, weights_only=False)
    pt_min, pt_max = checkpoint["pt_min"], checkpoint["pt_max"]
    architecture_type, input_dim, output_dim = checkpoint["model"], checkpoint["input_dim"], checkpoint["output_dim"]
    
    trajectory_model = build_model(architecture_type, input_dim, output_dim)
    trajectory_model.load_state_dict(checkpoint["state_dict"])
    
    eval_dataloader = get_eval_dataloader(adata, pseudotime, weights, pt_min, pt_max, batch_size=256)
    mu_list = []
    trajectory_model.eval()
    with torch.no_grad():
        for X_b, _ in eval_dataloader:
            mu_b, _, _ = trajectory_model(X_b)
            mu_list.append(mu_b.cpu().numpy())
            
    smooth_matrix = np.vstack(mu_list)
    smooth_matrix = np.exp(smooth_matrix)
    smooth_matrix = np.log1p(smooth_matrix)

    raw_matrix = np.log1p(get_raw_counts(adata))
    return smooth_matrix, raw_matrix


def smooth_to_raw_MSE_deep(args, adata, pseudotime, weights, run_model_dir):
    print("Executing Deep Smooth-to-Raw MSE")
    dataset = args.dataset

    class TrainArgsNamespace:
        model = TRAJECTORY_MODEL_TYPE
        gene = None 
        dataset = args.dataset
        
    run_trajectory(TrainArgsNamespace(), adata, pseudotime, weights, run_model_dir, loss_type="mse")
    
    generated_model_name = f"{TRAJECTORY_MODEL_TYPE}_{dataset}_all_mse.pth"
    model_path = run_model_dir / generated_model_name
    
    checkpoint = torch.load(model_path, weights_only=False)
    pt_min, pt_max = checkpoint["pt_min"], checkpoint["pt_max"]
    architecture_type, input_dim, output_dim = checkpoint["model"], checkpoint["input_dim"], checkpoint["output_dim"]
    
    trajectory_model = build_model(architecture_type, input_dim, output_dim)
    trajectory_model.load_state_dict(checkpoint["state_dict"])
    
    eval_dataloader = get_eval_dataloader(adata, pseudotime, weights, pt_min, pt_max, batch_size=256)
    mu_list = []
    trajectory_model.eval()
    with torch.no_grad():
        for X_b, _ in eval_dataloader:
            mu_b, _, _ = trajectory_model(X_b)
            mu_list.append(mu_b.cpu().numpy())
            
    smooth_matrix = np.vstack(mu_list)
    smooth_matrix = np.exp(smooth_matrix)
    smooth_matrix = np.log1p(smooth_matrix)

    raw_matrix = np.log1p(get_raw_counts(adata))
    return smooth_matrix, raw_matrix


def raw_to_raw(args, adata, pseudotime, weights):
    print("Executing Raw to Raw")
    raw_counts = np.log1p(get_raw_counts(adata))
    lineage_assignment = get_lineage_assignment(weights)
    n_lineages = weights.shape[1]
    
    raw_matrices = []
    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        if not np.any(mask): 
            continue
            
        pt_active = pseudotime[mask, l]
        counts_active = raw_counts[mask]
        
        sort_idx = np.argsort(pt_active)
        raw_matrices.append(counts_active[sort_idx])
        
    expression_matrix = np.vstack(raw_matrices)
    return expression_matrix


def raw_to_raw_lag(args, adata, pseudotime, weights):
    print("Executing Time-Lagged Raw-to-Raw")
    n_genes = adata.shape[1]
    gene_names = adata.var_names.values

    adj_matrix = np.zeros((n_genes, n_genes))
    
    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        print(f"{target_idx+1}/{n_genes}: {target_gene}")
        
        X_numpy, Y_numpy = get_lagged_expression(
            adata, pseudotime, weights, target_idx
        )
        
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
        
        kan_model, _ = train_n_to_1_kan(X_tensor, Y_tensor, epochs=400, lr=0.01, lamb_l1=0.02)
        
        kan_model.attribute()
        edge_magnitudes = kan_model.edge_scores[0].detach().cpu().numpy().flatten()
        edge_signs = get_correlation_signs(X_numpy, Y_numpy)
        
        edge_weights = edge_magnitudes * edge_signs
        adj_matrix[:, target_idx] = np.insert(edge_weights, target_idx, 0.0)
        
    return adj_matrix


def raw_to_raw_deep(args, adata, pseudotime, weights):
    print("Executing raw to raw deep")
    raw_counts = np.log1p(get_raw_counts(adata))
    lineage_assignment = get_lineage_assignment(weights)
    n_lineages = weights.shape[1]
    
    raw_matrices = []
    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        if not np.any(mask): 
            continue
            
        pt_active = pseudotime[mask, l]
        counts_active = raw_counts[mask]
        
        sort_idx = np.argsort(pt_active)
        raw_matrices.append(counts_active[sort_idx])
        
    expression_matrix = np.vstack(raw_matrices)
    return expression_matrix


def raw_to_raw_lin(args, adata, pseudotime, weights):
    print("Executing raw to raw lineage")
    raw_counts = np.log1p(get_raw_counts(adata))
    lineage_assignment = get_lineage_assignment(weights)
    n_lineages = weights.shape[1]
    
    lineage_matrices = []
    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        if not np.any(mask):
            continue
            
        pt_active = pseudotime[mask, l]
        counts_active = raw_counts[mask]
        sort_idx = np.argsort(pt_active)
        raw_sorted = counts_active[sort_idx]
        
        lineage_matrices.append((raw_sorted, raw_sorted))
        
    return lineage_matrices


def raw_to_raw_lag_lin(args, adata, pseudotime, weights):
    print("Executing Lineage-Specific Time-Lagged Raw-to-Raw")
    dataset = args.dataset
    experiment_name = getattr(args, 'experiment_name', 'temp')
    
    raw_counts = np.log1p(get_raw_counts(adata))
    lineage_assignment = get_lineage_assignment(weights)
    n_lineages = weights.shape[1]
    n_genes = raw_counts.shape[1]
    gene_names = adata.var_names.values
    
    if experiment_name:
        import re
        match = re.search(r'lag_(\d+)-(\d+)', experiment_name)
        if match:
            dt = float(f"{match.group(1)}.{match.group(2)}")
        elif 'lag_2' in experiment_name:
            dt = 0.20

    lineage_matrices = []
    
    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        if not np.any(mask): 
            continue
            
        print(f"Lag Lineage Branch {l + 1}/{n_lineages}")
        lin_adj = np.zeros((n_genes, n_genes))
        
        for target_idx in range(n_genes):
            lin_counts = raw_counts[mask]
            lin_pt = pseudotime[mask, l]
            sort_idx = np.argsort(lin_pt)
            lin_counts_sorted = lin_counts[sort_idx]
            lin_pt_sorted = lin_pt[sort_idx]
            
            X_branch = []
            Y_branch = []
            max_pt_on_branch = lin_pt_sorted[-1]
            
            for i in range(len(lin_pt_sorted)):
                t = lin_pt_sorted[i]
                t_future = t + dt
                if t_future > max_pt_on_branch:
                    break
                X_branch.append(lin_counts_sorted[i, :])
                y_future = np.interp(t_future, lin_pt_sorted, lin_counts_sorted[:, target_idx])
                Y_branch.append([y_future])
                
            if len(X_branch) == 0:
                continue
                
            X_numpy = np.array(X_branch)
            Y_numpy = np.array(Y_branch)
            
            X_final = np.delete(X_numpy, target_idx, axis=1)
            X_tensor = torch.tensor(X_final, dtype=torch.float32)
            Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
            
            kan_model, _ = train_n_to_1_kan(X_tensor, Y_tensor, epochs=400, lr=0.01, lamb_l1=0.02)
            kan_model.attribute()
            edge_magnitudes = kan_model.edge_scores[0].detach().cpu().numpy().flatten()
            edge_signs = get_correlation_signs(X_final, Y_numpy)
            
            edge_weights = edge_magnitudes * edge_signs
            lin_adj[:, target_idx] = np.insert(edge_weights, target_idx, 0.0)
            
        lineage_matrices.append(lin_adj)
        
        lin_output_dir = ensure_dir(TABLES_DIR / dataset / experiment_name / "grn")
        lin_filename = lin_output_dir / f"{dataset}_lineage_{l + 1}_grn.csv"
        lin_df = pd.DataFrame(lin_adj, index=gene_names, columns=gene_names)
        lin_df.to_csv(lin_filename)
        
        save_beeline_ranked_edges(lin_adj, gene_names, f"{dataset}_lineage_{l + 1}", experiment_name)
        
    if not lineage_matrices:
        return np.zeros((n_genes, n_genes))
        
    adj_matrix = np.mean(lineage_matrices, axis=0)
    return adj_matrix

def raw_to_raw_lag_deep(args, adata, pseudotime, weights, hidden_dim=[8, 8, 8]):
    print("Executing Deep Time-Lagged Raw-to-Raw")
    n_genes = adata.shape[1]
    gene_names = adata.var_names.values
    
    adj_matrix = np.zeros((n_genes, n_genes))
    epochs = 600
    lamb_l1 = 0.02

    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        print(f"Deep Lag Training Gene {target_idx+1}/{n_genes}: {target_gene}")
        
        X_numpy, Y_numpy = get_lagged_expression(
            adata, pseudotime, weights, target_idx
        )
        
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32, requires_grad=True)
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
        
        in_dim = X_tensor.shape[1]
        kan_model = PyKAN(
            width=[in_dim, hidden_dim, 1],
            grid=3, 
            k=3,
            device="cpu", 
            auto_save=False
        )
        kan_model.update_grid_from_samples(X_tensor.detach())
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(kan_model.parameters(), lr=0.01)

        kan_model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            predictions = kan_model(X_tensor)
            mse_loss = criterion(predictions, Y_tensor)
            l1_loss = sum(torch.sum(torch.abs(param)) for param in kan_model.parameters())
            loss = mse_loss + (lamb_l1 * l1_loss)
            loss.backward()
            optimizer.step()
            
        kan_model.eval()
        X_tensor_grad = torch.tensor(X_numpy, dtype=torch.float32, requires_grad=True)
        preds = kan_model(X_tensor_grad)
        loss_grad = torch.sum(preds)
        loss_grad.backward()
        
        gradients = X_tensor_grad.grad.abs().mean(dim=0).detach().cpu().numpy().flatten()
        edge_signs = get_correlation_signs(X_numpy, Y_numpy)
        
        edge_weights = gradients * edge_signs
        adj_matrix[:, target_idx] = np.insert(edge_weights, target_idx, 0.0)
        
    return adj_matrix

def train_n_to_1_mlp(X_tensor, Y_tensor, hidden_dim=[64, 64, 64], epochs=400, lr=0.01, lamb_l1=0.02):
    in_dim = X_tensor.shape[1]
    
    # Dynamically build the layers based on the hidden_dim list
    layers = []
    current_dim = in_dim
    
    for h_dim in hidden_dim:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(nn.SiLU())
        current_dim = h_dim
        
    # Append final output layer
    layers.append(nn.Linear(current_dim, 1))
    
    # Unpack the list into nn.Sequential
    model = nn.Sequential(*layers)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        mse_loss = criterion(predictions, Y_tensor)
        
        # Apply L1 weight regularization to keep feature paths sparse
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                
        loss = mse_loss + (lamb_l1 * l1_loss)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def raw_to_raw_lag_mlp(args, adata, pseudotime, weights, hidden_dim=[64, 64, 64]):
    print("Executing Time-Lagged Raw-to-Raw MLP")
    n_genes = adata.shape[1]
    gene_names = adata.var_names.values

    adj_matrix = np.zeros((n_genes, n_genes))
    
    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        print(f"MLP Lag Training Gene {target_idx+1}/{n_genes}: {target_gene}")
        
        X_numpy, Y_numpy = get_lagged_expression(
            adata, pseudotime, weights, target_idx
        )
        
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
        
        mlp_model = train_n_to_1_mlp(X_tensor, Y_tensor, hidden_dim=hidden_dim, epochs=400, lr=0.01, lamb_l1=0.02)
        
        # Saliency Mapping: Take gradients of inputs to track feature contributions
        X_tensor_grad = torch.tensor(X_numpy, dtype=torch.float32, requires_grad=True)
        preds = mlp_model(X_tensor_grad)
        loss_grad = torch.sum(preds)
        loss_grad.backward()
        
        gradients = X_tensor_grad.grad.abs().mean(dim=0).detach().cpu().numpy().flatten()
        edge_signs = get_correlation_signs(X_numpy, Y_numpy)
        
        edge_weights = gradients * edge_signs
        adj_matrix[:, target_idx] = np.insert(edge_weights, target_idx, 0.0)
        
    return adj_matrix

def run_grn(args, adata, pseudotime, weights, run_model_dir):
    dataset = args.dataset
    experiment_name = getattr(args, 'experiment_name', 'temp')

    input_matrix = None
    target_matrix = None
    adj_matrix = None

    n_genes = adata.shape[1]
    is_de = np.ones(n_genes, dtype=bool)

    if "smooth_to_smooth_ZINB" in experiment_name:
        expression_matrix = smooth_to_smooth_ZINB(args, adata, pseudotime, weights, run_model_dir)
        input_matrix, target_matrix = expression_matrix, expression_matrix
        
    elif "smooth_to_smooth_MSE_deep" in experiment_name:
        expression_matrix = smooth_to_smooth_MSE_deep(args, adata, pseudotime, weights, run_model_dir)
        input_matrix, target_matrix = expression_matrix, expression_matrix
        
    elif "smooth_to_smooth_MSE" in experiment_name:
        expression_matrix = smooth_to_smooth_MSE(args, adata, pseudotime, weights, run_model_dir)
        input_matrix, target_matrix = expression_matrix, expression_matrix
        
    elif "smooth_to_raw_ZINB" in experiment_name:
        smooth_m, raw_m = smooth_to_raw_ZINB(args, adata, pseudotime, weights, run_model_dir)
        input_matrix, target_matrix = smooth_m, raw_m
        
    elif "smooth_to_raw_MSE_deep" in experiment_name:
        smooth_m, raw_m = smooth_to_raw_MSE_deep(args, adata, pseudotime, weights, run_model_dir)
        input_matrix, target_matrix = smooth_m, raw_m
        
    elif "smooth_to_raw_MSE" in experiment_name:
        smooth_m, raw_m = smooth_to_raw_MSE(args, adata, pseudotime, weights, run_model_dir)
        input_matrix, target_matrix = smooth_m, raw_m
        
    elif "raw_to_raw_deep" in experiment_name:
        expression_matrix = raw_to_raw_deep(args, adata, pseudotime, weights)
        input_matrix, target_matrix = expression_matrix, expression_matrix
    
    elif "raw_to_raw_lag_deep" in experiment_name:
        adj_matrix = raw_to_raw_lag_deep(args, adata, pseudotime, weights)
    
    elif "raw_to_raw_lag_mlp" in experiment_name:
        adj_matrix = raw_to_raw_lag_mlp(args, adata, pseudotime, weights)
        
    elif "raw_to_raw_lag_lin" in experiment_name:
        adj_matrix = raw_to_raw_lag_lin(args, adata, pseudotime, weights)

    elif "raw_to_raw_lag" in experiment_name:
        adj_matrix = raw_to_raw_lag(args, adata, pseudotime, weights)
        
    elif "raw_to_raw_lin" in experiment_name:
        input_matrix = raw_to_raw_lin(args, adata, pseudotime, weights)
        
    elif "raw_to_raw_pykan" in experiment_name:
        adj_matrix = raw_to_raw_pykan(args, adata, pseudotime, weights)
        
    elif "raw_to_raw" in experiment_name:
        expression_matrix = raw_to_raw(args, adata, pseudotime, weights)
        input_matrix, target_matrix = expression_matrix, expression_matrix

    de_gene_names = adata.var_names.values[is_de]

    if "raw_to_raw_pykan" not in experiment_name and "raw_to_raw_lag" not in experiment_name:
        if isinstance(input_matrix, list):
            lineage_matrices = []
            
            for l_idx, (lin_input, lin_target) in enumerate(input_matrix):
                print(f"--- Inferring Lineage Branch {l_idx + 1}/{len(input_matrix)} ---")
                save_dir = ensure_dir(run_model_dir / f"grn_models_lineage_{l_idx + 1}")
                
                if "_deep" in experiment_name:
                    train_grn_models_deep(lin_input, lin_target, de_gene_names, save_dir, hidden_dim=2)
                    lin_adj = extract_grn_matrix_deep(de_gene_names, save_dir)
                else:
                    train_grn_models(lin_input, lin_target, de_gene_names, save_dir)
                    lin_adj = extract_grn_matrix(de_gene_names, save_dir)
                    
                lineage_matrices.append(lin_adj)
                
                lin_output_dir = ensure_dir(TABLES_DIR / dataset / experiment_name / "grn")
                lin_filename = lin_output_dir / f"{dataset}_lineage_{l_idx + 1}_grn.csv"
                lin_df = pd.DataFrame(lin_adj, index=de_gene_names, columns=de_gene_names)
                lin_df.to_csv(lin_filename)
                
                save_beeline_ranked_edges(lin_adj, de_gene_names, f"{dataset}_lineage_{l_idx + 1}", experiment_name)
            
            adj_matrix = np.mean(lineage_matrices, axis=0)
        else:
            save_dir = ensure_dir(run_model_dir / "grn_models")
            
            if "_deep" in experiment_name:
                train_grn_models_deep(input_matrix, target_matrix, de_gene_names, save_dir, hidden_dim=2)
                adj_matrix = extract_grn_matrix_deep(de_gene_names, save_dir)
            else:
                train_grn_models(input_matrix, target_matrix, de_gene_names, save_dir)
                adj_matrix = extract_grn_matrix(de_gene_names, save_dir)

    run_output_dir = ensure_dir(TABLES_DIR / dataset / experiment_name / "grn")
    output_filename = run_output_dir / f"{dataset}_grn.csv"
    
    adj_df = pd.DataFrame(adj_matrix, index=de_gene_names, columns=de_gene_names)
    adj_df.to_csv(output_filename)

    save_beeline_ranked_edges(adj_matrix, de_gene_names, dataset, experiment_name)