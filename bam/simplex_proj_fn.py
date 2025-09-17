import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplexProjectionFunc(nn.Module):
    def __init__(
        self,
        tau: float=1.0, 
        beta: float=0.5,
    ):
        super().__init__()

        self.tau = tau
        self.beta = beta

    def forward(self, scores):
        log_s = torch.log(F.softplus(scores) + 1e-8)
        log_s_max = log_s.max(dim=-1, keepdim=True).values

        numerator = self.tau * (log_s - log_s_max)
        denominator = self.beta * torch.logsumexp(numerator, dim=-1, keepdim=True)
        reg = self.tau * (1-self.beta) * log_s_max
        log_w = numerator - denominator + reg

        weights = torch.exp(log_w)

        valid = torch.isfinite(weights)
        weights = weights.masked_fill(~valid, 0.0)
        return weights