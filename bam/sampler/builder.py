from .sampler.registry import SAMPLER_REGISTRY


def sampler_builder(name, **kwargs):
    cls = SAMPLER_REGISTRY[name]
    return cls(**kwargs)