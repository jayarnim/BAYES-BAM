import torch
import torch.nn as nn


class SoftmaxProjection(nn.Module):
    def __init__(
        self,
        beta: float,
    ):
        """
        Reference: He et al., "NAIS: Neural attentive item similarity model for recommendation", IEEE 2018.
        """
        super().__init__()

        self.beta = beta

    def forward(self, scores):
        # numerator
        numerator = torch.exp(scores)
        # denominator
        numerator_sum = numerator.sum(dim=-1, keepdim=True)
        denominator = (numerator_sum + 1e-8) ** self.beta
        # attention weight
        weights = numerator / denominator
        # stabilize -inf, +inf
        valid = torch.isfinite(weights)
        weights = weights.masked_fill(~valid, 0.0)
        return weights