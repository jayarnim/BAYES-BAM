import torch
import torch.nn as nn
from torch.distributions import LogNormal, Weibull, Gamma
from .constants import SCORE_FN_TYPE
from .score_fn import AttentionScoreFunc


class LognormalSampler(nn.Module):
    def __init__(
        self,
        dim,
        score_fn_type: SCORE_FN_TYPE="hadamard",
        hyper_approx: float=0.1,
        hyper_prior: float=1.0,
        dropout: float=0.2,
    ):
        super().__init__()
    
        self.dim = dim
        self.score_fn_type = score_fn_type
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.dropout = dropout
        
        self._init_layers()

    def forward(self, Q, K, mask):
        approx = self._approx(Q, K)
        samples = approx.rsample()
        prior = self._prior(K)

        dist = dict(
            approx=approx,
            prior=prior,
            mask=self._match_dim(mask, samples),
        )

        return samples, dist

    def _approx(self, Q, K):
        psi = self.approx_exp_fn(Q, K)
        sigma = torch.full_like(psi, self.hyper_approx)
        mu = psi - 0.5 * (sigma ** 2)
        dist = LogNormal(mu, sigma)
        return dist

    def _prior(self, K):
        psi = self.prior_exp_fn(K).squeeze(-1)
        sigma = torch.full_like(psi, self.hyper_prior)
        mu = psi - 0.5 * (sigma ** 2)
        dist = LogNormal(mu, sigma)
        return dist

    def _match_dim(self, source, target):
        if source is not None:
            source = source.expand_as(target)
        return source

    def _init_layers(self):
        kwargs = dict(
            dim=self.dim, 
            score_fn_type=self.score_fn_type,
            dropout=self.dropout,
        )
        self.approx_exp_fn = AttentionScoreFunc(**kwargs)
        
        self.prior_exp_fn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, 1),
            nn.Softmax(dim=-1),
        )


class WeibullSampler(nn.Module):
    def __init__(
        self,
        dim,
        score_fn_type: SCORE_FN_TYPE="hadamard",
        hyper_approx: float=0.1,
        hyper_prior: float=1.0,
        dropout: float=0.2,
    ):
        super().__init__()
    
        self.dim = dim
        self.score_fn_type = score_fn_type
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.dropout = dropout
        
        self._init_layers()

    def forward(self, Q, K, mask):
        approx = self._approx(Q, K)
        samples = approx.rsample()
        prior = self._prior(K)

        dist = dict(
            approx=approx,
            prior=prior,
            mask=self._match_dim(mask, samples),
        )

        return samples, dist

    def _approx(self, Q, K):
        psi = self.approx_exp_fn(Q, K)
        k_ = torch.full_like(psi, self.hyper_approx)
        lambda_ = torch.exp(psi) / torch.exp(torch.lgamma(1 + 1.0 / k_))
        dist = Weibull(lambda_, k_)
        return dist

    def _prior(self, K):
        psi = self.prior_exp_fn(K).squeeze(-1)
        beta = torch.full_like(psi, self.hyper_prior)
        alpha = torch.exp(psi) * beta
        dist = Gamma(alpha, beta)
        return dist

    def _match_dim(self, source, target):
        if source is not None:
            source = source.expand_as(target)
        return source

    def _init_layers(self):
        kwargs = dict(
            dim=self.dim, 
            score_fn_type=self.score_fn_type,
            dropout=self.dropout,
        )
        self.approx_exp_fn = AttentionScoreFunc(**kwargs)
        
        self.prior_exp_fn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, 1),
            nn.Softmax(dim=-1),
        )