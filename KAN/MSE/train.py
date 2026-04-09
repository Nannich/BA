from efficient_kan import KAN
import torch
from torch import nn
from dataset import get_dataloaders
from model import build_kan

DATA_PATH = "~/BA/data/bifurcating/sim_1/"
BATCH_SIZE = 64
EPOCHS = 2000
TARGET_GENE = 0

def train_loop(dataloader, model, loss_fn, optimizer, device):
    # Set the model to training mode
    model.train()
    
    total_loss = 0.0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # For only fitting a single gene
        y_target = y[:, TARGET_GENE].unsqueeze(1)

        # Clear old gradients
        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)            # All genes
        #loss = loss_fn(pred, y_target)      # Only target gene

        # Backpropagation
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients caused by extreme values
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        # Update the weights
        optimizer.step()
        
        # Add loos to running total
        total_loss += loss.item()

    # Calculate and return the average loss for this epoch
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test_loop(dataloader, model, loss_fn, device):
    # Set the model to testing mode
    model.eval()

    total_test_loss = 0.0  

    # no_grad saves RAM
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # For only fitting a single gene
            y_target = y[:, TARGET_GENE].unsqueeze(1)

            pred = model(X)
            loss = loss_fn(pred, y)            # All genes
            #loss = loss_fn(pred, y_target)      # Only target gene
            total_test_loss += loss.item()
    
    # Calculate the average test loss for the whole dataset
    avg_test_loss = total_test_loss / len(dataloader)

    return avg_test_loss

def main():
    train_dataloader, test_dataloader, input_dim, output_dim = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE)
    model = build_kan(input_dim, output_dim)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    print(f"Starting training on: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)

    loss_fn = nn.MSELoss()

    # Early Stopping
    patience = 16
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for t in range(EPOCHS):
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        val_loss = test_loop(test_dataloader, model, loss_fn, device)
        print(f"Epoch [{t+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Only save the model when it actually improves
            torch.save(model.state_dict(), "trained_kan_sim1.pth")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

if __name__ == "__main__":
    main()