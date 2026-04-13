from efficient_kan import KAN
import torch
from dataset import get_dataloaders
from model import build_kan
from loss import ZINBLoss

DATA_PATH = "~/BA/data/bifurcating/sim_1/"
BATCH_SIZE = 64
EPOCHS = 2000
TARGET_GENE = 12
LR = 1e-2
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_LIMIT = 5

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
    avg_loss = total_loss / len(dataloader)
    return avg_loss

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
    
    avg_test_loss = total_test_loss / len(dataloader)

    return avg_test_loss

def main():
    # Load data
    train_dataloader, test_dataloader, input_dim, output_dim = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE)
    
    # Initialize the KAN model
    model = build_kan(input_dim, output_dim)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    print(f"Starting training on: {device}")

    # Optimizer and Loss setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = ZINBLoss()

    # Early Stopping Setup
    # Stops training if validation loss doesn't improve for patience consecutive epochs
    patience = 16
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
            torch.save(model.state_dict(), "trained_kan_sim1.pth")
        else:
            epochs_no_improve += 1

        # Trigger early stopping    
        if epochs_no_improve >= patience:
            break

if __name__ == "__main__":
    main()