import torch
from torch import nn
import torch.nn.functional as F

class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial (ZINB) Loss.

    Note:
        Algorithmic implementation adapted from the `zinbautoencoder` module 
        within the Netmap package:
        https://github.com/bionetslab/netmap/blob/main/src/netmap/model/zinbautoencoder.py
    """
    def __init__(self, scale_factor=1.0, eps=1e-10, ridge_lambda=0.0, reduction='mean'):
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
        self.reduction = reduction

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

        # Inputs are ALREADY natively activated raw counts from the model container.
        y_pred = y_pred * self.scale_factor  # Ensure scale factor is applied to pre-activated mu

        # Clip pi 
        pi = torch.clamp(pi, min=eps, max=1.0 - eps)

        # Negative binomial log-likelihood
        nb_case = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + eps)
            + (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps)))
            + y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )

        nb_case = nb_case - torch.log(1.0 - pi)

        # Zero-inflation log-likelihood for y_true = 0
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        # Combine cases: zero or NB
        result = torch.where(y_true < eps, zero_case, nb_case)

        # Add ridge penalty for pi
        ridge = self.ridge_lambda * torch.square(pi)
        result += ridge

        if self.reduction == 'mean':
            return torch.mean(result)
        elif self.reduction == 'sum':
            return torch.sum(result)
        elif self.reduction == 'none':
            return result

class MSEWrapperLoss(nn.Module):
    """
    Wrapper that ignores theta and pi to train purely on MSE.
    """
    def __init__(self):
        super(MSEWrapperLoss, self).__init__()

    def forward(self, y_true, mu, theta, pi):
        # Log transform the ground truth
        y_true_log1p = torch.log1p(y_true.float())
        
        # Log transform the pre-activated raw prediction vector
        y_pred_log1p = torch.log1p(mu)
        
        # Calculate standard Mean Squared Error
        return F.mse_loss(y_pred_log1p, y_true_log1p)