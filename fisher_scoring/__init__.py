import logging

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
    def __init__(self, *args, **kwargs):
        logger.warning(
            "FisherScoringLogisticRegression is deprecated, use LogisticRegression instead."
        )
        super().__init__(*args, **kwargs)


class FisherScoringMultinomialRegression(MultinomialLogisticRegression):
    def __init__(self, *args, **kwargs):
        logger.warning(
            "FisherScoringMultinomialRegression is deprecated, use MultinomialLogisticRegression instead."
        )
        super().__init__(*args, **kwargs)


class FisherScoringFocalRegression(FocalLossRegression):
    def __init__(self, *args, **kwargs):
        logger.warning(
            "FisherScoringFocalRegression is deprecated, use FocalLossRegression instead."
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "DivergenceClassifier",
    "LogisticRegression",
    "MultinomialLogisticRegression",
    "FocalLossRegression",
    "PoissonRegression",
    "NegativeBinomialRegression",
    "RobustLogisticRegression",
]

__version__ = "2.0.6"
