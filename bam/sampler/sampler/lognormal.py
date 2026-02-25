import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions import LogNormal
from ..score.builder import score_fn_builder


class LogNormalSampler(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        hyper_approx: float,
        hyper_prior: float,
        score: str,
    ):
        super().__init__()

        self.dim = dim
        self.dropout = dropout
        self.std_approx = hyper_approx
        self.std_prior = hyper_prior
        self.score = score

        self._set_up_components()

    def forward(self, Q, K):
        # approx
        approx_exp = self.score_fn_approx(Q, K)
        approx = self.build_approx_dist(approx_exp)
        # prior
        prior_exp = self.score_fn_prior(K).squeeze(-1)
        prior = self.build_prior_dist(prior_exp)
        return torch.exp(approx_exp), approx, prior

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
        scale = torch.full_like(exp_val, self.std_approx)
        loc = exp_val - 0.5 * (scale ** 2)
        dist = LogNormal(loc, scale)
        return dist

    def build_prior_dist(self, exp_val):
        scale = torch.full_like(exp_val, self.std_prior)
        loc = exp_val - 0.5 * (scale ** 2)
        dist = LogNormal(loc, scale)
        return dist

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            name=self.score,
            dim=self.dim,
            dropout=self.dropout,
        )
        self.score_fn_approx = score_fn_builder(**kwargs)

        self.score_fn_prior = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),

            nn.Linear(self.dim, 1),
            nn.Softmax(dim=1),
        )