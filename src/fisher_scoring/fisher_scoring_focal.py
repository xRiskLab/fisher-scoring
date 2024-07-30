"""
Author: xRiskLab (deburky)
GitHub: github.com/xRiskLab
Beta Version: 0.1
2024 MIT License

Fisher Scoring Focal Loss Regression
------------------------------------

This is a Python implementation of the Fisher Scoring algorithm for 
logistic regression with focal loss. It is designed for imbalanced
classification problems where the positive class is rare.

The implementation is purely experimental, use with caution.

References:

Tsung-Yi Lin et al. Focal Loss for Dense Object Detection. ICCV 2017.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class FisherScoringFocalRegression(
    BaseEstimator, ClassifierMixin
):
    """
    Fisher Scoring Focal Loss Regression class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        epsilon: float = 1e-10,
        max_iter: int = 100,
        information: str = "expected",
        use_bias: bool = True,
        verbose: bool = False,
    ) -> None:
        self.gamma = gamma
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
    def logistic_function(z: np.ndarray) -> np.ndarray:
        """
        Compute the logistic function for the input array z.
        """
        p = 1 / (1 + np.exp(-z))
        return np.clip(p, 1e-10, 1 - 1e-10)

    @staticmethod
    def safe_log(
        x: np.ndarray, minval: float = 1e-5
    ) -> np.ndarray:
        """
        Compute the safe log to prevent log(0).
        """
        return np.log(np.clip(x, minval, 1 - minval))

    @staticmethod
    def generate_focal_parameter(
        y: np.ndarray, p: np.ndarray, gamma: float
    ) -> np.ndarray:
        """
        Generate the focal parameter for the focal loss.
        """
        mask_pos = y == 1
        mask_neg = y == 0

        pt = np.where(mask_pos, p, 1 - p)
        pt = np.clip(pt, 1e-10, 1 - 1e-10)
        pt[mask_pos] = (1 - pt[mask_pos]) ** gamma
        pt[mask_neg] = pt[mask_neg] ** gamma
        return pt

    def compute_loss(
        self, y: np.ndarray, p: np.ndarray
    ) -> float:
        """
        Compute the focal loss for logistic regression.
        """
        pt = np.where(y == 1, p, 1 - p)
        pt = np.clip(pt, 1e-10, 1 - 1e-10)
        focal_weight = (1 - pt) ** self.gamma
        return np.sum(focal_weight * self.safe_log(pt))

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
        self, X: np.ndarray, y: np.ndarray
    ) -> "FisherScoringFocalRegression":
        """
        Fit the focal logistic regression model using Fisher scoring.
        """

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_features = X.shape[1]

        # Initialize bias term if use_bias is True
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            n_features += 1

        # Initialize weights (beta) to zero
        self.beta = np.zeros((n_features, 1))

        for iteration in range(self.max_iter):
            z = X @ self.beta
            p = self.logistic_function(z)
            pt = self.generate_focal_parameter(
                y, p, self.gamma
            )
            focal_p = p * pt

            score = X.T @ ((y - p) * pt)
            score_vector = (y - focal_p) * X
            W = np.diag((focal_p * (1 - focal_p)).ravel())

            expected_I = X.T @ W @ X
            observed_I = score_vector.T @ score_vector

            loss = self.compute_loss(y, p)
            focal_loss = -loss / X.shape[0]

            self.loss_history.append(loss)

            if self.verbose:
                loss = self.compute_loss(y, p) / X.shape[0]
                if iteration == 0:
                    print(
                        "Starting Fisher Scoring Iterations..."
                    )
                print(
                    f"Iteration: {iteration + 1}, Focal Loss: {focal_loss:.4f}"
                )

            if self.information == "expected":
                beta_new = (
                    self.beta
                    + self.invert_matrix(expected_I) @ score
                )
            elif self.information == "observed":
                beta_new = (
                    self.beta
                    + self.invert_matrix(observed_I) @ score
                )
            else:
                raise ValueError(
                    "Information must be 'expected' or 'observed'"
                )
            if (
                np.linalg.norm(beta_new - self.beta)
                < self.epsilon
            ):
                print(
                    f"Convergence reached after {iteration + 1} iterations."
                )
                self.beta = beta_new
                break

            self.beta = beta_new
            self.beta_history.append(self.beta.copy())
            if iteration == self.max_iter - 1:
                print(
                    "Maximum iterations reached without convergence."
                )
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
        proba_class_1 = self.logistic_function(
            X @ self.beta
        )
        return np.hstack([1 - proba_class_1, proba_class_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target labels for the input data X.
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

    def get_params(
        self, deep: bool = True
    ) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
            "information": self.information,
            "use_bias": self.use_bias,
            "verbose": self.verbose,
        }

    def set_params(
        self, **params: Union[float, int, str, bool]
    ) -> "FisherScoringFocalRegression":
        for key, value in params.items():
            setattr(self, key, value)
        return self
