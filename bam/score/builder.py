from .score.registry import SCORE_FN_REGISTRY


def score_fn_builder(
    dim: int,
    dropout: float,
    score: str,
):
    kwargs = dict(
        dim=dim,
        dropout=dropout,
    )
    return SCORE_FN_REGISTRY[score](**kwargs)