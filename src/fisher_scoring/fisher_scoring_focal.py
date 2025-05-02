"""
fisher_scoring_focal.py.

Focal Loss Regression
------------------------------------

Author: xRiskLab (deburky)
GitHub: https://github.com/xRiskLab
License: MIT

Description:
This module provides a Python implementation of the Fisher Scoring algorithm
for logistic regression, incorporating focal loss to address challenges
in imbalanced classification problems. It is particularly suited for datasets
where the positive class is rare and traditional logistic regression may underperform.

Key Features:
- Fisher Scoring optimization for robust parameter estimation.
- Focal loss integration to prioritize hard-to-classify examples.
- Designed for research and experimental use (not production-ready).

Usage:
This implementation is experimental and should be used with caution.
Extensive testing and validation are recommended for specific applications.

References:
- Tsung-Yi Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.special import xlogy
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class FocalLossRegression(BaseEstimator, ClassifierMixin):
    """
    Fisher Scoring Focal Loss Regression class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        epsilon: float = 1e-10,
        max_iter: int = 100,
        information: str = "expected",
        significance: float = 0.05,
        use_bias: bool = True,
        verbose: bool = False,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.information = information
        self.use_bias = use_bias
        self.significance = significance
        self.verbose = verbose
        self.beta: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        self.beta_history: List[np.ndarray] = []
        self.information_matrix: Dict[str, List[np.ndarray]] = {
            "iteration": [],
            "information": [],
        }
        self.standard_errors: Optional[np.ndarray] = None
        self.wald_statistic: Optional[np.ndarray] = None
        self.p_values: Optional[np.ndarray] = None
        self.lower_bound: Optional[np.ndarray] = None
        self.upper_bound: Optional[np.ndarray] = None
        self.is_fitted_: bool = False
        self.feature_names: Optional[List[str]] = None
        self.score_vectors = pd.DataFrame()

    @staticmethod
    def logistic_function(z: np.ndarray) -> np.ndarray:
        """
        Compute the logistic function for the input array z.
        """
        p = 1 / (1 + np.exp(-z))
        return np.clip(p, 1e-10, 1 - 1e-10)

    @staticmethod
    def safe_log(x: np.ndarray, minval: float = 1e-5) -> np.ndarray:
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

    def compute_loss(self, y: np.ndarray, p: np.ndarray) -> float:
        """
        Compute the focal loss for logistic regression.
        """
        pt = np.where(y == 1, p, 1 - p)
        pt = np.clip(pt, 1e-10, 1 - 1e-10)
        focal_weight = (1 - pt) ** self.gamma
        return np.sum((xlogy(y, p) + xlogy(1 - y, 1 - p)) * focal_weight)

    @staticmethod
    def invert_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Attempt to invert a matrix, falling back to the pseudo-inverse
        if the matrix is singular.
        """
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            print("WARNING: Singular matrix. Using pseudo-inverse.")
            return np.linalg.pinv(matrix)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FisherScoringFocalRegression":
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
            p = self.logistic_function(X @ self.beta)
            pt = self.generate_focal_parameter(y, p, self.gamma)

            # Adjust weights so that their sum equals the sample size
            pt /= len(y) / np.sum(pt)

            score_vector = (y - p) * X * pt
            score = np.sum(score_vector, axis=0).reshape(-1, 1)

            # Select information matrix based on expected or empirical Fisher information
            if self.information == "expected":
                # Expected Fisher Information matrix
                W_diag = (p * (1 - p) * pt).ravel()
                information_matrix = (X.T * W_diag) @ X
            else:
                # Empirical Fisher Information matrix
                score_vector = (y - p).reshape(X.shape[0], 1, 1)
                X_vector = X.reshape(X.shape[0], -1, 1)
                information_matrix = np.sum(
                    X_vector
                    @ score_vector.transpose(0, 2, 1)
                    @ score_vector
                    @ X_vector.transpose(0, 2, 1)
                    * pt.reshape(-1, 1, 1),
                    axis=0,
                )
            self.information_matrix["iteration"].append(iteration)
            self.information_matrix["information"].append(information_matrix)

            loss = self.compute_loss(y, p)
            focal_loss = -loss / X.shape[0]

            self.loss_history.append(loss)

            if self.verbose:
                if iteration == 0:
                    print("Starting Fisher Scoring Iterations...")
                print(f"Iteration: {iteration + 1}, Focal Loss: {focal_loss:.4f}")

            # Update beta using the Fisher Scoring update rule
            beta_new = self.beta + self.invert_matrix(information_matrix) @ score

            # Check for convergence
            if np.linalg.norm(beta_new - self.beta) < self.epsilon:
                if self.verbose:
                    print(f"Convergence reached after {iteration + 1} iterations.")
                self.beta = beta_new
                self.max_iter = iteration + 1
                break

            self.beta = beta_new
            self.beta_history.append(self.beta.copy())
            if iteration == self.max_iter - 1:
                print("Maximum iterations reached without convergence.")
                self.max_iter = iteration + 1

        self.compute_statistics()
        self.is_fitted_ = True
        return self

    def compute_statistics(self) -> None:
        """
        Compute the standard errors, Wald statistic, p-values, and confidence intervals.
        """
        information_matrix = self.information_matrix["information"][
            -1
        ]  # Information at the MLE

        information_matrix_inv = self.invert_matrix(information_matrix)

        self.standard_errors = np.sqrt(np.diagonal(information_matrix_inv))
        betas = self.beta.flatten()

        # Handle division by zero
        if self.standard_errors.any() == 0:
            # Raise warning
            print("WARNING: Standard errors are zero. Setting to 1.")
        self.wald_statistic = betas / self.standard_errors
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.wald_statistic)))

        critical_value = norm.ppf(1 - self.significance / 2)
        self.lower_bound = betas - critical_value * self.standard_errors
        self.upper_bound = betas + critical_value * self.standard_errors

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
        proba_class_1 = self.logistic_function(X @ self.beta)
        return np.hstack([1 - proba_class_1, proba_class_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target labels for the input data X.
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

    def predict_ci(self, X, method="logit"):
        """
        Compute confidence intervals for predicted probabilities or logits.

        Parameters:
            X (numpy.ndarray): Input data matrix.
            method (str): Confidence interval method, "logit" (default) or "proba".

        Returns:
            np.ndarray: Array with lower and upper confidence intervals for predictions.
        """
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        logit = (X @ self.beta).flatten()
        proba = self.logistic_function(logit)
        information_matrix = self.information_matrix["information"][-1]
        cov_matrix = self.invert_matrix(information_matrix)
        z_crit = norm.ppf(1 - self.significance / 2)  # Critical value for CI

        if method == "logit":
            # Gradient for linear logit confidence intervals
            std_errors = np.array(
                [np.sqrt(np.dot(np.dot(g, cov_matrix), g)) for g in X]
            )
            lower_proba = self.logistic_function(logit - z_crit * std_errors)
            upper_proba = self.logistic_function(logit + z_crit * std_errors)
        elif method == "proba":  # Probability confidence intervals
            gradients = (proba * (1 - proba))[:, None] * X  # Element-wise gradient
            std_errors = np.sqrt(np.sum(gradients @ cov_matrix * gradients, axis=1))
            lower_proba = np.clip(proba - z_crit * std_errors, 0, 1)
            upper_proba = np.clip(proba + z_crit * std_errors, 0, 1)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'logit' or 'proba'.")

        return np.vstack((lower_proba, upper_proba)).T

    def get_params(self, deep: bool = True) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
            "information": self.information,
            "use_bias": self.use_bias,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Union[float, int, str, bool]) -> FocalLossRegression:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def summary(self) -> Dict[str, np.ndarray]:
        return {
            "betas": self.beta.flatten(),
            "standard_errors": self.standard_errors,
            "wald_statistic": self.wald_statistic,
            "p_values": self.p_values,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }  # type: ignore

    def display_summary(self, style="default") -> None:
        """
        Display a summary for IPython notebooks or console output.
        Args:
            style (str): The style for the summary output.
        """
        console = Console()
        summary_dict = self.summary()

        total_iterations = len(self.information_matrix["iteration"])
        table = Table(title="Fisher Scoring Focal Logistic Regression Summary")

        table.add_column(
            "Parameter",
            justify="right",
            style=style,
            no_wrap=True,
        )
        table.add_column("Estimate", style=style)
        table.add_column("Std. Error", style=style)
        table.add_column("Wald Statistic", style=style)
        table.add_column("P-value", style=style)
        table.add_column("Lower CI", style=style)
        table.add_column("Upper CI", style=style)

        if self.feature_names:
            param_names = (
                ["intercept (bias)"] + self.feature_names
                if self.use_bias
                else self.feature_names
            )
        else:
            param_names = [f"Beta {i}" for i in range(len(summary_dict["betas"]))]
        for i, param in enumerate(param_names):
            table.add_row(
                f"{param}",
                f"{summary_dict['betas'][i]:.4f}",
                f"{summary_dict['standard_errors'][i]:.4f}",
                f"{summary_dict['wald_statistic'][i]:.4f}",
                f"{summary_dict['p_values'][i]:.4f}",
                f"{summary_dict['lower_bound'][i]:.4f}",
                f"{summary_dict['upper_bound'][i]:.4f}",
            )

        summary_stats = f"""
        Total Fisher Scoring Iterations: [{style}]{total_iterations}[/{style}]
        Focal Log Likelihood: [{style}]{self.loss_history[-1]:.4f}[/{style}]
        Beta 0 = intercept (bias): [{style}]{self.use_bias}[/{style}]
        """

        console.print(
            Panel.fit(
                summary_stats,
                title="Fisher Scoring Focal Logistic Regression Fit",
                safe_box=True,
            )
        )
        console.print(table)
