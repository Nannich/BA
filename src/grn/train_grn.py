import torch
import torch.optim as optim
from torch import nn
from kan import KAN
from src.trajectory.zinb_loss import ZINBLoss
from src.trajectory.zinb_models import activate_zinb_output

def train_n_to_1_kan(X_tensor, Y_tensor, device, loss_mode="mse", epochs=400, lr=0.01, lamb_l1=0.02):
    in_dim = X_tensor.shape[1]
    out_dim = 3 if loss_mode == "zinb" else 1
    network_width = [in_dim, out_dim]
    
    model = KAN(width=network_width, grid=3, k=3, device=device, auto_save=False)
    model.update_grid_from_samples(X_tensor)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = ZINBLoss(ridge_lambda=0.1, reduction='mean') if loss_mode == "zinb" else nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        
        if loss_mode == "zinb":
            raw_output = model(X_tensor)
            mu, theta, pi = activate_zinb_output(raw_output, n_genes=1)
            loss = loss_fn(Y_tensor, mu, theta, pi)
        else:
            mu = model(X_tensor)
            loss = loss_fn(mu, Y_tensor)
            
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        total_loss = loss + (lamb_l1 * l1_loss)
        
        total_loss.backward()
        optimizer.step()

    model.eval()
    return model