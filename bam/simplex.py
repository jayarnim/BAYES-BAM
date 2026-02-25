import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjection(nn.Module):
    def __init__(
        self,
        beta: float,
    ):
        """
        Reference: Fan et al., "Bayesian Attention Modules", NeurIPS 2020.
        """
        super().__init__()

        self.beta = beta

    def forward(self, scores):
        # numerator
        numerator = F.relu(scores)
        # denominator
        numerator_sum = numerator.sum(dim=-1, keepdim=True) + 1e-8
        denominator = numerator_sum ** self.beta
        # attention weight
        weights = numerator / denominator
        # stabilize -inf, +inf
        valid = torch.isfinite(weights)
        weights = weights.masked_fill(~valid, 0.0)
        return numerator / denominator