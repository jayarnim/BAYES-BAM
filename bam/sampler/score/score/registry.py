from .cat import Concatenation
from .prod import Product


SCORE_FN_REGISTRY = {
    "cat": Concatenation,
    "prod": Product,
}