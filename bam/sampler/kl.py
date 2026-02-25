import torch
from torch.distributions import Weibull, Gamma, register_kl


@register_kl(Weibull, Gamma)
def weibull_gamma(
    approx: Weibull, 
    prior: Gamma,
):
    """
    Analytic KL divergence: KL(Weibull(k, λ) || Gamma(α, β))
    Reference: Fan et al., "Bayesian Attention Modules", NeurIPS 2020.
    """
    k, lam = approx.concentration, approx.scale
    alpha, beta = prior.concentration, prior.rate

    const = -torch.special.digamma(torch.tensor(1.0, device=k.device))
    
    eps = 1e-8
    
    kl = (
        (const / k) - torch.log(lam * k + eps)
        + ((alpha - 1) / k) * (const + torch.log(lam + eps))
        + beta * lam * torch.exp(torch.lgamma(1.0 + 1.0 / k))
        - alpha
    )
    
    kl_clamped = torch.clamp(kl, min=0.0)
    
    return kl_clamped