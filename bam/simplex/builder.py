from .simplex.registry import SIMPLEX_FN_REGISTRY


def simplex_fn_builder(
    simplex: str,
    beta: int,
):
    return SIMPLEX_FN_REGISTRY[simplex](beta)