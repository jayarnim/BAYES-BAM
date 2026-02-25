from .score.registry import SCORE_FN_REGISTRY


def score_fn_builder(name, **kwargs):
    cls = SCORE_FN_REGISTRY[name]
    return cls(**kwargs)