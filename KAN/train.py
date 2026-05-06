import torch
import os
from dataloaders import get_dataloaders, get_eval_dataloader
from de import calculate_mse_per_curve, calculate_nll_per_gene, calculate_aic, calculate_bic
from model import build_model
from loss import ZINBLoss

from model import MLP_HIDDEN_LAYERS, PYKAN_HIDDEN_LAYERS, EFFKAN_HIDDEN_LAYERS

BATCH_SIZE = 256
EPOCHS = 2000
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_LIMIT = 5
PATIENCE = 12

def train_loop(dataloader, model, loss_fn, optimizer, device, epoch, is_kan):
    model.train()
    total_loss = 0.0
    
    avg_mu, avg_theta, avg_pi = 0.0, 0.0, 0.0

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Update the KAN grid on the first batch of early epochs
        update_grid = is_kan and (epoch < 5 and batch_idx == 0)

        if is_kan:
            mu_logits, theta_logits, pi_logits = model(X, update_grid=update_grid)
        else:
            mu_logits, theta_logits, pi_logits = model(X)

        loss = loss_fn(y, mu_logits, theta_logits, pi_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_LIMIT)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Track raw network outputs
        with torch.no_grad():
            avg_mu += mu_logits.mean().item()
            avg_theta += theta_logits.mean().item()
            avg_pi += torch.sigmoid(pi_logits).mean().item()

    n_batches = len(dataloader)
    return total_loss / n_batches, avg_mu / n_batches, avg_theta / n_batches, avg_pi / n_batches


def test_loop(dataloader, model, loss_fn, device, is_kan):
    model.eval()
    total_test_loss = 0.0  

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            if is_kan:
                mu, theta, pi = model(X, update_grid=False)
            else:
                mu, theta, pi = model(X)
                
            loss = loss_fn(y, mu, theta, pi)
            total_test_loss += loss.item() 

    return total_test_loss / len(dataloader)


def run_training(args, adata, pseudotime, weights):
    model_type = args.model
    model_dir = args.model_dir
    target_gene = args.gene
    dataset = args.dataset

    train_dataloader, test_dataloader, input_dim, output_dim, pt_min, pt_max = get_dataloaders(
        adata, pseudotime, weights, target_gene, BATCH_SIZE
    )
    
     # Initialize the model
    model = build_model(model_type, input_dim, output_dim)
    checkpoint = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "model": model_type,
        "gene": target_gene,
        "state_dict": model.state_dict(),
        "pt_min": pt_min,
        "pt_max": pt_max,
        "hidden_layers": MLP_HIDDEN_LAYERS if model_type == "mlp" else (
                         EFFKAN_HIDDEN_LAYERS if model_type == "effkan" else PYKAN_HIDDEN_LAYERS),
        "wd": WEIGHT_DECAY,
        "lr": LR,
        "mse": 0
    }

    gene_str = f"gene{target_gene}" if target_gene is not None else "all"
    filename = f"{model_type}_{dataset}_{gene_str}.pth"
    model_path = os.path.join(model_dir, filename)

    device = "cpu"
    model.to(device)
    print(f"Starting training on: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = ZINBLoss(ridge_lambda=0.11)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    is_kan = model_type in ["effkan", "pykan"]

    # Early Stopping Setup
    for t in range(EPOCHS):
        train_loss, a_mu, a_th, a_pi = train_loop(train_dataloader, model, loss_fn, optimizer, device, t, is_kan)
        val_loss = test_loop(test_dataloader, model, loss_fn, device, is_kan)

        if t % 5 == 0:
            print(f"Epoch [{t+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test: {val_loss:.4f} | "
                  f"Raw μ: {a_mu:.2f}, Raw θ: {a_th:.2f}, π: {a_pi:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint["state_dict"] = model.state_dict()
            torch.save(checkpoint, model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {t+1} epochs.")
            break

    # Add Evaluation metrics to checkpoint
    model.load_state_dict(checkpoint["state_dict"])
    full_dataloader = get_eval_dataloader(
        adata, pseudotime, weights, pt_min, pt_max, target_gene, batch_size=256
    )
    mse_per_curve = calculate_mse_per_curve(full_dataloader, model, device)
    checkpoint["mse"] = mse_per_curve.cpu()

    unreduced_loss_fn = ZINBLoss(reduction='none') 
    total_nll_tensor = calculate_nll_per_gene(full_dataloader, model, unreduced_loss_fn, device)
    
    # Number of parameters: k
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_genes = total_nll_tensor.shape[0]

    # Distribute k across the number of genes
    k_per_gene = n_params / n_genes
    n_samples = len(full_dataloader.dataset)

    checkpoint["aic"] = calculate_aic(k_per_gene, total_nll_tensor).cpu()
    checkpoint["bic"] = calculate_bic(k_per_gene, total_nll_tensor, n_samples).cpu()
    
    torch.save(checkpoint, model_path)