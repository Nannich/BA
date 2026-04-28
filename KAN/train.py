import torch
import os
from dataset import get_dataloaders, get_eval_dataloader
from de import calculate_mse_per_curve, calculate_nll_per_gene, calculate_aic, calculate_bic
from model import build_model
from loss import ZINBLoss

from model import MLP_HIDDEN_LAYERS, PYKAN_HIDDEN_LAYERS, EFFKAN_HIDDEN_LAYERS


BATCH_SIZE = 64
EPOCHS = 2000
TARGET_GENE = 12
LR = 5e-3
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_LIMIT = 5
PATIENCE = 16


def train_loop(dataloader, model, loss_fn, optimizer, device):
    # Set the model to training mode
    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # The ZINBKAN outputs 3 parameters per gene
        # - mu: Predicted true biological mean expression
        # - theta: Dispersion parameter (variance)
        # - pi: Dropout probability (zero-inflation)

        mu, theta, pi = model(X)

        # Compute the ZINB negative log-likelihood
        loss = loss_fn(y, mu, theta, pi)

        # Backpropagation
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients caused by extreme values
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_LIMIT)

        # Update the weights
        optimizer.step()
        
        # Add loss to running total
        total_loss += loss.item()

    # Calculate and return the average loss for this epoch
    return total_loss / len(dataloader)


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    total_test_loss = 0.0  

    # no_grad saves RAM
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            mu, theta, pi = model(X)
            loss = loss_fn(y, mu, theta, pi)
            total_test_loss += loss.item() 

    return total_test_loss / len(dataloader)


def run_training(args):
    sim = args.sim
    data_dir = args.data_dir
    model_type = args.model
    model_dir = args.model_dir
    target_gene = args.gene
    dataset = args.dataset
    
    data_path = os.path.join(data_dir, dataset, f"sim_{sim}/")

    # Load data
    train_dataloader, test_dataloader, input_dim, output_dim, pt_min, pt_max = get_dataloaders(data_path, target_gene, BATCH_SIZE)
    
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
    filename = f"{model_type}_sim{sim}_{gene_str}.pth"
    model_path = os.path.join(model_dir, dataset, filename)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    print(f"Starting training on: {device}")

    # Optimizer and Loss setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = ZINBLoss()

    # Early Stopping Setup
    # Stops training if validation loss doesn't improve for patience consecutive epochs
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training Loop
    for t in range(EPOCHS):
        
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        val_loss = test_loop(test_dataloader, model, loss_fn, device)

        print(f"Epoch [{t+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f}")

        # Check whether validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Only save the model when it actually improves
            checkpoint["state_dict"] = model.state_dict()
            torch.save(checkpoint, model_path)
        else:
            epochs_no_improve += 1

        # Trigger early stopping    
        if epochs_no_improve >= PATIENCE:
            break

    # Add MSE to checkpoint
    model.load_state_dict(checkpoint["state_dict"])
    full_dataloader = get_eval_dataloader(data_path, pt_min, pt_max, target_gene, batch_size=256)
    mse_per_curve = calculate_mse_per_curve(full_dataloader, model, device)
    checkpoint["mse"] = mse_per_curve.cpu()

    # Add AIC and BIC to checkpoint
    unreduced_loss_fn = ZINBLoss(reduction='none') 

    # Negative Log-Likelihood for each gene
    total_nll_tensor = calculate_nll_per_gene(full_dataloader, model, unreduced_loss_fn, device)
    
    # Number of parameters: k
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Distribute k across the number of genes
    n_genes = total_nll_tensor.shape[0]
    k_per_gene = n_params / n_genes

    # Number of cells: n
    n_samples = len(full_dataloader.dataset)

    checkpoint["aic"] = calculate_aic(k_per_gene, total_nll_tensor).cpu()
    checkpoint["bic"] = calculate_bic(k_per_gene, total_nll_tensor, n_samples).cpu()
    
    torch.save(checkpoint, model_path)