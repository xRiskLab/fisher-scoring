"""
Author: xRiskLab (deburky)
GitHub: github.com/xRiskLab
Beta Version: 0.1
2024 MIT License

Fisher Scoring Multinomial Regression
-------------------------------------

This is a Python implementation of the Fisher Scoring algorithm for multinomial 
logistic regression. The Fisher Scoring algorithm is an iterative optimization
algorithm that is used to estimate the parameters of a multinomial logistic 
regression model. 

The algorithm is based on the Newton-Raphson method and uses the expected or
observed Fisher information matrix to update the model parameters.

References:

Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing. https://web.stanford.edu/~jurafsky/slp3/
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class FisherScoringMultinomialRegression(
    BaseEstimator, ClassifierMixin
):
    """
    Fisher Scoring Multinomial Logistic Regression class.
    """

    def __init__(
        self,
        epsilon: float = 1e-10,
        max_iter: int = 100,
        information: str = "expected",
        use_bias: bool = True,
        verbose: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.information = information
        self.use_bias = use_bias
        self.verbose = verbose
        self.beta: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        self.beta_history: List[np.ndarray] = []
        self.information_matrix: Dict[
            str, List[np.ndarray]
        ] = {
            "iteration": [],
            "information": [],
        }
        self.is_fitted_: bool = False
        self.feature_names: Optional[List[str]] = None

    @staticmethod
    def softmax_function(z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax function for the input array z.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def compute_loss(y: np.ndarray, p: np.ndarray) -> float:
        """
        Compute the log likelihood loss for multinomial logistic regression.
        """
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.sum(xlogy(y, p))

    @staticmethod
    def invert_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Attempt to invert a matrix, falling back to the pseudo-inverse
        if the matrix is singular.
        """
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            print(
                "WARNING: Singular matrix. Using pseudo-inverse."
            )
            return np.linalg.pinv(matrix)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "FisherScoringMultinomialRegression":
        """
        Fit the multinomial logistic regression model using Fisher scoring.
        """

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        # Initialize bias term if use_bias is True
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            n_features += 1

        # Initialize weights (beta) to zero
        self.beta = np.zeros((n_features, n_classes))

        for iteration in range(self.max_iter):
            p = self.softmax_function(X @ self.beta)

            score_matrix = (y_one_hot - p).T @ X
            score = score_matrix.T

            expected_I = np.zeros((n_features, n_features))
            W = p * (1 - p)
            for i in range(n_classes):
                Wi = np.diag(W[:, i])
                expected_I += X.T @ Wi @ X

            observed_I = np.zeros((n_features, n_features))
            for i in range(n_samples):
                score_i = (y_one_hot[i] - p[i]).reshape(
                    -1, 1
                )
                Xi = X[i].reshape(-1, 1)
                observed_I += (
                    Xi @ score_i.T @ score_i @ Xi.T
                )

            if self.information == "expected":
                information_matrix = expected_I
            elif self.information == "observed":
                information_matrix = observed_I
            else:
                raise ValueError(
                    "Information must be 'expected' or 'observed'"
                )

            self.information_matrix["iteration"].append(iteration)  # type: ignore
            if self.information == "expected":
                self.information_matrix[
                    "information"
                ].append(expected_I)
            elif self.information == "observed":
                self.information_matrix[
                    "information"
                ].append(observed_I)
            else:
                raise ValueError(
                    "Information must be 'expected' or 'observed'"
                )

            loss = (
                self.compute_loss(y_one_hot, p) / n_samples
            )
            self.loss_history.append(loss)

            loss = self.compute_loss(y_one_hot, p)
            log_loss = -loss / X.shape[0]

            if self.verbose:
                if iteration == 0:
                    print(
                        "Starting Fisher Scoring Iterations..."
                    )
                print(
                    f"Iteration: {iteration + 1}, Log Loss: {log_loss:.4f}"
                )

            new_beta = (
                self.invert_matrix(information_matrix)
                @ score
            )
            self.beta += new_beta

            if np.linalg.norm(new_beta) < self.epsilon:
                print(
                    f"Convergence reached after {iteration + 1} iterations."
                )
                break

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class probabilities for the input data X.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                "This Classifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments "
                "before using this estimator."
            )
        X = np.array(X)
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.softmax_function(X @ self.beta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target labels for the input data X.
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def get_params(
        self, deep: bool = True
    ) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
            "information": self.information,
            "use_bias": self.use_bias,
            "verbose": self.verbose,
        }

    def set_params(
        self, **params: Union[float, int, str, bool]
    ) -> "FisherScoringMultinomialRegression":
        for key, value in params.items():
            setattr(self, key, value)
        return self
