from efficient_kan import KAN
import torch
from torch import nn
from dataset import get_dataloaders

DATA_PATH = "~/BA/data/bifurcating/sim_1/"
BATCH_SIZE = 64
EPOCHS = 32

def train_loop(dataloader, model, loss_fn, optimizer, device):
    # Set the model to training mode
    model.train()
    
    total_loss = 0.0 
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        
        # TODO: Taken from scKAN, get reason for doing this
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        
        # Update the weights
        optimizer.step()
        
        # Add loos to running total
        total_loss += loss.item()

    # Calculate and return the average loss for this entire epoch
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
            pred = model(X)
            loss = loss_fn(pred, y)
            total_test_loss += loss.item()
    
    # Calculate the average test loss for the whole dataset
    avg_test_loss = total_test_loss / len(dataloader)

    return avg_test_loss

def main():
    train_dataloader, test_dataloader, input_dim, output_dim = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE)

    model = KAN([input_dim, 64, output_dim], grid_size=5, spline_order=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Starting training on: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    loss_fn = nn.MSELoss()

    for t in range(EPOCHS):
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        val_loss = test_loop(test_dataloader, model, loss_fn, device)
        print(f"Epoch [{t+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()