from typing import Optional
import torch
import torch.nn as nn
from .sampler import (
    LognormalSampler,
    WeibullSampler,
)
from .simplex_proj_fn import SimplexProjectionFunc as simplex_proj_fn
from .constants import (
    SAMPLER_TYPE,
    SCORE_FN_TYPE,
)


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        sampler_type: SAMPLER_TYPE="lognormal",
        score_fn_type: SCORE_FN_TYPE="hadamard",
        hyper_approx: float=0.1,
        hyper_prior: float=1.0,
        tau: float=1.0, 
        beta: float=0.5,
        dropout: float=0.2,
    ):
        super().__init__()

        self.dim = dim
        self.sampler_type = sampler_type
        self.score_fn_type = score_fn_type
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.tau = tau
        self.beta = beta
        self.dropout = dropout

        self._init_layers()

    def forward(
        self, 
        Q: torch.Tensor,    # (B,D)
        K: torch.Tensor,    # (B,H,D)
        V: torch.Tensor,    # (B,H,D)
        mask: Optional[torch.Tensor]=None,  # (B,H)
    ):
        B_len, H_len, D_len = K.shape

        # Q: (B,D) -> (B,1,D) -> (B,H,D)
        Q_exp = Q.unsqueeze(1).expand(B_len, H_len, D_len)

        # Sampling attn score: (B,H)
        samples, dist = self.sampler(Q_exp, K, mask)

        # Masking: (B,H) or (H,) -> (B,H)
        if mask is not None:
            kwargs = dict(
                input=samples, 
                mask=self._match_dim(mask, samples), 
                value=float('-inf'),
            )
            samples = torch.masked_fill(**kwargs)

        # Simplex projection: (B,H) -> (B,H)
        weights = self.simplex_proj_fn(samples)

        # Context vector: (B,H) x (B,H,D) -> (B,D)
        context = torch.sum(weights.unsqueeze(-1) * V, dim=1)

        return context, dist

    def _match_dim(self, source, target):
        if source is not None:
            source = source.expand_as(target)
        return source

    def _init_layers(self):
        kwargs = dict(
            dim=self.dim, 
            score_fn_type=self.score_fn_type, 
            hyper_approx=self.hyper_approx, 
            hyper_prior=self.hyper_prior, 
            dropout=self.dropout,
        )

        if self.sampler_type=='lognormal':
            self.sampler = LognormalSampler(**kwargs)
        elif self.sampler_type=='weibull':
            self.sampler = WeibullSampler(**kwargs)
        else:
            raise ValueError("Invalid Approx. Dist.")

        kwargs = dict(
            tau=self.tau, 
            beta=self.beta, 
        )
        self.simplex_proj_fn = simplex_proj_fn(**kwargs)