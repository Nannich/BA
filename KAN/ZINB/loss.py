import torch
from torch import nn

class ZINBLoss(nn.Module):
    def __init__(self, scale_factor=1.0, eps=1e-10, ridge_lambda=0.0):
        """
        Zero-Inflated Negative Binomial (ZINB) Loss
        Args:
            scale_factor (float): Scale factor applied to predictions.
            eps (float): Small value for numerical stability.
            ridge_lambda (float): Regularization weight for the zero-inflation probability (pi).
        """
        super(ZINBLoss, self).__init__()
        self.scale_factor = scale_factor
        self.eps = eps
        self.ridge_lambda = ridge_lambda

    def forward(self, y_true, y_pred, theta, pi):
        """
        Compute the ZINB loss.
        Args:
            y_true (torch.Tensor): Ground truth counts (non-negative integers).
            y_pred (torch.Tensor): Predicted mean values (mu).
            theta (torch.Tensor): Dispersion parameter (shape parameter).
            pi (torch.Tensor): Zero-inflation probability (between 0 and 1).
        Returns:
            torch.Tensor: ZINB negative log-likelihood.
        """
        eps = self.eps
        y_true = y_true.float()
        y_pred = y_pred.float() * self.scale_factor
        theta = theta.float()
        pi = torch.clamp(pi.float(), min=eps, max=1 - eps)  # Ensure pi is in (0, 1)

        # Clip theta to avoid numerical issues
        theta = torch.clamp(theta, max=1e6)

        # Negative binomial log-likelihood
        nb_case = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + eps)
            + (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps)))
            + y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )

        # Zero-inflation log-likelihood for y_true = 0
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        # Combine cases: zero or NB
        result = torch.where(y_true < eps, zero_case, nb_case)

        # Add ridge penalty for pi
        ridge = self.ridge_lambda * torch.square(pi)
        result += ridge

        return torch.mean(result)  # Return mean loss over the batch
