import math
import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions import Weibull, Gamma


class WeibullSampler(nn.Module):
    def __init__(
        self,
        score_fn,
        dim: int,
        dropout: float,
        hyper_approx: float,
        hyper_prior: float,
    ):
        super().__init__()

        self.approx_exp_fn = score_fn
        self.dim = dim
        self.dropout = dropout
        self.k = hyper_approx
        self.beta = hyper_prior

        self._set_up_components()

    def forward(self, Q, K):
        # approx
        approx_exp_val = self.approx_exp_fn(Q, K)
        approx = self.build_approx_dist(approx_exp_val)
        # prior
        prior_exp_val = self.prior_exp_fn(K).squeeze(-1)
        prior = self.build_prior_dist(prior_exp_val)
        return torch.exp(approx_exp_val), approx, prior

    def estimate(self, Q, K):
        exp_val, approx, prior = self.forward(Q, K)
        samples = approx.rsample()
        kl = kl_divergence(approx, prior)
        return samples, kl

    @torch.no_grad()
    def predict(self, Q, K):
        exp_val, approx, prior = self.forward(Q, K)
        kl = kl_divergence(approx, prior)
        return exp_val, kl

    def build_approx_dist(self, exp_val):
        k = torch.full_like(exp_val, self.k)
        lambda_ = torch.exp(exp_val) / torch.exp(torch.lgamma(1 + 1.0 / k))
        dist = Weibull(lambda_, k)
        return dist

    def build_prior_dist(self, exp_val):
        beta = torch.full_like(exp_val, self.beta)
        alpha = torch.exp(beta) * beta
        dist = Gamma(alpha, beta)
        return dist

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        self.prior_exp_fn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.dim, 1),
            nn.Softmax(dim=1),
        )