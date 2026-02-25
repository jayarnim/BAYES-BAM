import torch
import torch.nn as nn
from .sampler.builder import sampler_builder
from .score.builder import score_fn_builder
from .simplex.builder import simplex_fn_builder


class BayesianAttentionModules(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        dist: str,
        score: str,
        simplex: str,
        hyper_approx: float,
        hyper_prior: float,
        beta: float,
    ):
        """
        Bayesian Attention Modules (Fan et al., 2020)
        -----

        Args:
            dim (int):
                dimensionality of query, key, and value.
                (assuming that the query, key, and value dimensions are the same.)
            dropout (float):
                dropout rate applied to attention score expected value function.
            dist (str):
                approximate distribution for attention score.
                (e.g. `lognormal`, `weibull`)
            score (str):
                type of attention score expected value function.
                (e.g. `cat`, `prod`)
            simplex (str):
                type of simplex projection function.
                (e.g. `linear`, `softmax`)
            hyper_approx (float):
                hyper-parameter of approximate distribution.
                if approximate distribution is `lognormal`, hyper-parameter is `std`.
                if approximate distribution is `weibull`, hyper-parameter is `k`.
            hyper_prior (float):
                hyper-parameter of prior distribution.
                if approximate distribution is `lognormal`, prior distribution is `lognormal` and hyper-parameter is `std`.
                if approximate distribution is `weibull`, prior distribution is `gamma` and hyper-parameter is `beta`.            
            beta (float):
                smoothing factor for normalization @ simplex.
                (range: (0,1])
        """
        super().__init__()

        self.dim = dim
        self.dropout = dropout
        self.dist = dist
        self.score = score
        self.simplex = simplex
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.beta = beta

        self._set_up_components()

    def estimate(
        self, 
        Q: torch.Tensor,                    # (B,D)
        K: torch.Tensor,                    # (B,H,D)
        V: torch.Tensor,                    # (B,H,D)
        mask: torch.Tensor,                 # (B,H)
    ):
        """
        Training Method
        -----
        Attention scores are sampled from the approximate distribution.

        Args:
            Q (torch.Tensor):
                the number of queries is the batch size B, (B,D).
            K (torch.Tensor):
                assume each query has different reference information.
                the number of references is H per query, (B,H,D).
            V (torch.Tensor):
                even if the value of the reference information is the same as the key,
                it must be assigned individually.
            mask (torch.Tensor):
                mask for padding index or correct index, (B,H).

        Returns:
            context (torch.Tensor):
                dimensionality of context vector is (B,D).
            kl_mean (torch.Tensor):
                mini-batch average of the kl divergence of the attention score distribution,
                masked for padding indices or correct indices.
        """
        # Q: (B,D) -> (B,1,D) -> (B,H,D)
        Q_exp = Q.unsqueeze(1).expand_as(K)
        # attention scores: (B,H)
        scores, kl_entry = self.sampler.estimate(Q_exp, K)
        # masking: (B,H) -> (B,H)
        scores_masked = scores.masked_fill(~mask, float('-inf'))
        # simplex projection: (B,H) -> (B,H)
        weights = self.simplex_fn(scores_masked)
        # context vector: (B,H,1) x (B,H,D) -> (B,H,D) -> (B,D)
        context = torch.sum(weights.unsqueeze(-1) * V, dim=1)
        # kl mean: (B,H) -> scalar
        kl_mean = (
            (kl_entry * mask).sum(dim=1) 
            / mask.sum(dim=1).clamp_min(1)
        ).mean()
        return context, kl_mean

    @torch.no_grad()
    def predict(
        self, 
        Q: torch.Tensor,                    # (B,D)
        K: torch.Tensor,                    # (B,H,D)
        V: torch.Tensor,                    # (B,H,D)
        mask: torch.Tensor,                 # (B,H)
    ):
        """
        Evaluation Method
        -----
        Using the attention score as the expected value of the approximate distribution.

        Args:
            Q (torch.Tensor):
                the number of queries is the batch size B, (B,D).
            K (torch.Tensor):
                assume each query has different reference information.
                the number of references is H per query, (B,H,D).
            V (torch.Tensor):
                even if the value of the reference information is the same as the key,
                it must be assigned individually.
            mask (torch.Tensor):
                mask for padding index or correct index, (B,H).

        Returns:
            context (torch.Tensor):
                dimensionality of context vector is (B,D).
            kl_mean (torch.Tensor):
                mini-batch average of the kl divergence of the attention score distribution,
                masked for padding indices or correct indices.
        """
        # Q: (B,D) -> (B,1,D) -> (B,H,D)
        Q_exp = Q.unsqueeze(1).expand_as(K)
        # attention scores: (B,H)
        scores, kl_entry = self.sampler.predict(Q_exp, K)
        # masking: (B,H) -> (B,H)
        scores_masked = scores.masked_fill(~mask, float('-inf'))
        # simplex projection: (B,H) -> (B,H)
        weights = self.simplex_fn(scores_masked)
        # context vector: (B,H,1) x (B,H,D) -> (B,H,D) -> (B,D)
        context = torch.sum(weights.unsqueeze(-1) * V, dim=1)
        # kl mean: (B,H) -> scalar
        kl_mean = (
            (kl_entry * mask).sum(dim=1) 
            / mask.sum(dim=1).clamp_min(1)
        ).mean()
        return context, kl_mean

    def _set_up_components(self):
        self._create_components()
    
    def _create_components(self):
        kwargs = dict(
            dim=self.dim,
            dropout=self.dropout,
            score=self.score,
        )
        score_fn = score_fn_builder(**kwargs)

        kwargs = dict(
            score_fn=score_fn,
            dim=self.dim,
            dropout=self.dropout,
            dist=self.dist,
            hyper_approx=self.hyper_approx,
            hyper_prior=self.hyper_prior,
        )
        self.sampler = sampler_builder(**kwargs)

        kwargs = dict(
            simplex=self.simplex,
            beta=self.beta,
        )
        self.simplex_fn = simplex_fn_builder(**kwargs)