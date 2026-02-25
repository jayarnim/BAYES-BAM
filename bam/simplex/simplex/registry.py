from .softmax import SoftmaxProjection
from .linear import LinearProjection


SIMPLEX_FN_REGISTRY = {
    "softmax": SoftmaxProjection,
    "linear": LinearProjection,
}