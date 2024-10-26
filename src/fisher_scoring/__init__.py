from .fisher_scoring_logistic import (
    FisherScoringLogisticRegression,
)
from .fisher_scoring_multinomial import (
    FisherScoringMultinomialRegression,
)
from .fisher_scoring_focal import (
    FisherScoringFocalRegression,
)

__all__ = [
    "FisherScoringLogisticRegression",
    "FisherScoringMultinomialRegression",
    "FisherScoringFocalRegression",
]

# Add dynamic version retrieval
try:
    from importlib.metadata import version
    __version__ = version("fisher-scoring")
except ImportError:
    __version__ = "unknown"