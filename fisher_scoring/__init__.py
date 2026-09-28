import logging
from typing import Any

from .fisher_scoring_bradley_terry import BradleyTerry, pairs_from_counts
from .fisher_scoring_divergence import DivergenceClassifier
from .fisher_scoring_focal import FocalLossRegression
from .fisher_scoring_logistic import LogisticRegression
from .fisher_scoring_multinomial import MultinomialLogisticRegression
from .fisher_scoring_poisson import NegativeBinomialRegression, PoissonRegression
from .fisher_scoring_robust import RobustLogisticRegression

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# Dummy classes for backward compatibility
class FisherScoringLogisticRegression(LogisticRegression):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "FisherScoringLogisticRegression is deprecated, use LogisticRegression instead."
        )
        super().__init__(*args, **kwargs)


class FisherScoringMultinomialRegression(MultinomialLogisticRegression):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "FisherScoringMultinomialRegression is deprecated, use MultinomialLogisticRegression instead."
        )
        super().__init__(*args, **kwargs)


class FisherScoringFocalRegression(FocalLossRegression):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "FisherScoringFocalRegression is deprecated, use FocalLossRegression instead."
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "BradleyTerry",
    "DivergenceClassifier",
    "FocalLossRegression",
    "LogisticRegression",
    "MultinomialLogisticRegression",
    "NegativeBinomialRegression",
    "PoissonRegression",
    "RobustLogisticRegression",
    "pairs_from_counts",
]

__version__ = "2.0.7"
