from .sampler.registry import SAMPLER_REGISTRY


def sampler_builder(
    score_fn,
    dim: int,
    dropout: float,
    dist: str,
    hyper_approx: float,
    hyper_prior: float,
):
    kwargs = dict(
        score_fn=score_fn,
        dim=dim,
        dropout=dropout,
        hyper_approx=hyper_approx,
        hyper_prior=hyper_prior,
    )
    return SAMPLER_REGISTRY[dist](**kwargs)