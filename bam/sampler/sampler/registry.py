from .lognormal import LogNormalSampler
from .weibull import WeibullSampler


SAMPLER_REGISTRY = {
    "lognormal": LogNormalSampler,
    "weibull": WeibullSampler,
}