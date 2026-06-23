import time
import torch
from pathlib import Path

from src.trajectory.zinb_models import build_kan_model, build_mlp_model, build_null_model
from src.trajectory.zinb_loss import ZINBLoss, MSEWrapperLoss


def prepare_data(dataset, target_gene_idx):
    """Extracts the pseudotime matrix (X) and a single gene's raw expression counts (Y)."""
    X_numpy = dataset.pseudotime
    input_features = X_numpy.shape[1]
    output_features = 1
    gene_label = dataset.gene_names[target_gene_idx]

    Y_numpy = dataset.raw_counts[:, [target_gene_idx]]
    
    X = torch.tensor(X_numpy, dtype=torch.float32)
    Y = torch.tensor(Y_numpy, dtype=torch.float32)
    
    return X, Y, input_features, output_features, gene_label


def run_optimization_loop(model, X, Y, loss_fn, optimizer, epochs, loss_name, gradient_clip_limit, update_grid_epochs=0):
    """Executes a full-batch backpropagation routine."""
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if update_grid_epochs > 0:
            update_grid = (epoch < update_grid_epochs)
            mu, theta, pi = model(X, update_grid=update_grid)
        else:
            mu, theta, pi = model(X)

        loss = loss_fn(Y, mu, theta, pi)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_limit)
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            if loss_name == "zinb":
                with torch.no_grad():
                    mean_pi = torch.sigmoid(pi).mean().item()
                print(f"  Epoch [{epoch+1:03d}/{epochs}] - Loss: {loss.item():.4f} | Mean μ: {mu.mean().item():.3f} | Dropout π: {mean_pi:.3f}")
            else:
                print(f"  Epoch [{epoch+1:03d}/{epochs}] - MSE Loss: {loss.item():.4f}")
                
    return loss.item()


def save_checkpoint(model, model_type, loss, target_gene_idx, hidden_layers, gene_label, final_loss, save_dir):
    """Saves the model checkpoint"""

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f"{model_type}_{gene_label}_{loss}.pth"
    model_path = save_path / filename
    
    checkpoint = {
        "model_type": model_type,
        "loss": loss,
        "target_gene_idx": target_gene_idx,
        "hidden_layers": hidden_layers,
        "state_dict": model.state_dict(),
        "gene_names": [gene_label],
        "final_loss": final_loss
    }
    torch.save(checkpoint, model_path)
    print(f"Saved checkpoint to: {model_path}")



def train_trajectory(
    dataset, 
    loss="mse", 
    target_gene_idx=None, 
    hidden_layers=None, 
    epochs=500, 
    lr=0.02, 
    weight_decay=1e-06, 
    gradient_clip_limit=5, 
    save_dir=None
):
    """
    Trains a Kolmogorov-Arnold Network to model a single gene's expression trajectory.

    Parameters:
        dataset: Single-cell dataset object containing pseudotime and counts.
        loss (str): Loss function type, either "mse" or "zinb".
        target_gene_idx (int): Index of the target gene to model.
        hidden_layers (list): List defining the KAN hidden layer architecture.
        epochs (int): Number of full-batch optimization epochs.
        lr (float): Learning rate for the AdamW optimizer.
        weight_decay (float): Weight decay coefficient.
        gradient_clip_limit (float): Maximum norm limit for gradient clipping.
        save_dir (str/Path): Directory path to save the model checkpoint.

    Returns:
        torch.nn.Module: The trained KAN model instance.
    """
    X, Y, in_f, out_f, gene_label = prepare_data(dataset, target_gene_idx)
    hidden_layers = hidden_layers if hidden_layers is not None else [1]

    model = build_kan_model(in_f, out_f, hidden_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = MSEWrapperLoss() if loss == "mse" else ZINBLoss(ridge_lambda=0.1)

    print(f"Model: KAN | Loss: {loss.upper()} | Target: {gene_label}")
    start_time = time.time()

    final_loss = run_optimization_loop(
        model=model, X=X, Y=Y, loss_fn=loss_fn, optimizer=optimizer, 
        epochs=epochs, loss_name=loss, gradient_clip_limit=gradient_clip_limit, 
        update_grid_epochs=20
    )

    print(f"Training finished in: {time.time() - start_time:.2f}s")
    save_checkpoint(model, "kan", loss, target_gene_idx, hidden_layers, gene_label, final_loss, save_dir)
    return model


def train_trajectory_mlp(
    dataset, 
    loss="mse", 
    target_gene_idx=None, 
    hidden_layers=None, 
    epochs=500, 
    lr=0.02, 
    weight_decay=1e-05, 
    gradient_clip_limit=5, 
    save_dir=None
):
    """Trains a MLP to model a single gene's expression trajectory."""
    X, Y, in_f, out_f, gene_label = prepare_data(dataset, target_gene_idx)
    hidden_layers = hidden_layers if hidden_layers is not None else [489, 489, 489]

    model = build_mlp_model(in_f, out_f, hidden_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = MSEWrapperLoss() if loss == "mse" else ZINBLoss(ridge_lambda=0.11)

    print(f"Model: MLP | | Loss: {loss.upper()} | Target: {gene_label}")
    start_time = time.time()

    final_loss = run_optimization_loop(
        model=model, X=X, Y=Y, loss_fn=loss_fn, optimizer=optimizer, 
        epochs=epochs, loss_name=loss, gradient_clip_limit=gradient_clip_limit, 
        update_grid_epochs=0
    )

    print(f"Total Runtime: {time.time() - start_time:.2f}s")
    save_checkpoint(model, "mlp", loss, target_gene_idx, hidden_layers, gene_label, final_loss, save_dir)
    return model


def train_trajectory_null(
    dataset, 
    loss="mse", 
    target_gene_idx=None, 
    epochs=500, 
    lr=0.02, 
    gradient_clip_limit=5, 
    save_dir=None
):
    """Fits a bias-only static model to serve as a baseline background control."""
    X, Y, in_f, out_f, gene_label = prepare_data(dataset, target_gene_idx)

    model = build_null_model(out_f)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = MSEWrapperLoss() if loss == "mse" else ZINBLoss(ridge_lambda=0.01)

    print(f"Model: NULL | Target: {gene_label} | Loss: {loss.upper()}")
    start_time = time.time()

    final_loss = run_optimization_loop(
        model=model, X=X, Y=Y, loss_fn=loss_fn, optimizer=optimizer, 
        epochs=epochs, loss_name=loss, gradient_clip_limit=gradient_clip_limit, 
        update_grid_epochs=0
    )
    print(f"Finished in: {time.time() - start_time:.2f}s")
    save_checkpoint(model, "null", loss, target_gene_idx, [], gene_label, final_loss, save_dir)
    return model


def run_trajectory(args, dataset):
    """Unpacks command-line args and routes execution."""
    model_type = getattr(args, "model", "kan")
    loss = getattr(args, "loss", "mse")
    target_gene_idx = getattr(args, "gene", None)
    hidden_layers = getattr(args, "hidden_layers", None)
    epochs = getattr(args, "epochs", 500)
    
    save_dir = getattr(args, "model_dir")

    if model_type == "kan":
        return train_trajectory(
            dataset=dataset, loss=loss, target_gene_idx=target_gene_idx,
            hidden_layers=hidden_layers, epochs=epochs, save_dir=save_dir
        )
    elif model_type == "mlp":
        return train_trajectory_mlp(
            dataset=dataset, loss=loss, target_gene_idx=target_gene_idx,
            hidden_layers=hidden_layers, epochs=epochs, save_dir=save_dir
        )
    elif model_type == "null":
        return train_trajectory_null(
            dataset=dataset, loss=loss, target_gene_idx=target_gene_idx,
            epochs=epochs, save_dir=save_dir
        )