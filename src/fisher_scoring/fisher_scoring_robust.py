"""
fisher_scoring_robust.py.

Robust Logistic Regression
----------------------------------

Author: xRiskLab (deburky)
GitHub: https://github.com/xRiskLab)
License: MIT

Description:
This module contains the `RobustLogisticRegression` class, which is a custom
implementation of robust logistic regression using the Fisher scoring algorithm
with epsilon-contamination for outlier resistance. The Fisher scoring algorithm
is an iterative optimization algorithm that uses weighted information matrices
to update the model parameters robustly. The class provides methods for fitting
the model, making predictions, and computing model statistics.

The robust approach uses an epsilon-contamination model where observations are
assumed to come from a mixture: (1-ε)F + εG, where F is the true model and G
is a contamination distribution. This provides resistance to outliers by
down-weighting observations that are unlikely under the main model.

References:

Kevin P. Murphy. Probabilistic Machine Learning: An Introduction. MIT Press, 2022.
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


class RobustLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Robust Fisher Scoring Logistic Regression class using epsilon-contamination.

    This class implements a robust version of logistic regression that is resistant
    to outliers through the use of an epsilon-contamination model. The method
    down-weights observations that are unlikely under the main model, providing
    robustness against data contamination.
    """

    def __init__(
        self,
        epsilon_contamination: float = 0.05,
        contamination_prob: float = 0.5,
        tol: float = 1e-6,
        max_iter: int = 100,
        information: str = "expected",
        use_bias: bool = True,
        significance: float = 0.05,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the Robust Logistic Regression model.

        Parameters:
            epsilon_contamination (float): Contamination level (0 ≤ ε ≤ 1).
                Higher values provide more robustness but may reduce efficiency.
            contamination_prob (float): Probability for contamination distribution.
                Default is 0.5 (uniform contamination).
            tol (float): Convergence tolerance for parameter updates.
            max_iter (int): Maximum number of Fisher scoring iterations.
            information (str): Type of information matrix ('expected' or 'empirical').
            use_bias (bool): Whether to include intercept term.
            significance (float): Significance level for confidence intervals.
            verbose (bool): Whether to print iteration details.
        """
        self.epsilon_contamination = epsilon_contamination
        self.contamination_prob = contamination_prob
        self.tol = tol
        self.max_iter = max_iter
        self.information = information
        self.use_bias = use_bias
        self.significance = significance
        self.verbose = verbose

        # Model parameters
        self.beta: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None  # Robust weights

        # Tracking and statistics
        self.information_matrix: Dict[str, List] = {
            "iteration": [],
            "information": [],
        }
        self.loss_history: List[float] = []
        self.beta_history: List[np.ndarray] = []
        self.standard_errors: Optional[np.ndarray] = None
        self.wald_statistic: Optional[np.ndarray] = None
        self.p_values: Optional[np.ndarray] = None
        self.lower_bound: Optional[np.ndarray] = None
        self.upper_bound: Optional[np.ndarray] = None
        self.is_fitted_: bool = False
        self.feature_names: Optional[List[str]] = None
        self.fitted_X: Optional[np.ndarray] = None

    @staticmethod
    def logistic_function(z: np.ndarray) -> np.ndarray:
        """
        Compute the logistic function for the input array z.
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        p = 1 / (1 + np.exp(-z))
        return np.clip(p, 1e-10, 1 - 1e-10)

    @staticmethod
    def compute_loss(
        y: np.ndarray, p: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute the weighted log-likelihood loss for robust logistic regression.
        """
        p = np.clip(p, 1e-10, 1 - 1e-10)
        log_likelihood = xlogy(y, p) + xlogy(1 - y, 1 - p)

        if weights is not None:
            log_likelihood = weights * log_likelihood

        return np.sum(log_likelihood)

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

    def _compute_robust_weights(
        self, X: np.ndarray, y: np.ndarray, p: np.ndarray
    ) -> np.ndarray:
        """
        Compute robust weights using epsilon-contamination model.

        The weights down-weight observations that are unlikely under the main model,
        providing robustness against outliers.
        """
        # Model probabilities: P(y|model)
        p_model = np.where(y == 1, p, 1 - p)

        # Numerator: (1 - ε) * P(y|main model)
        numerator = (1 - self.epsilon_contamination) * p_model

        # Denominator: numerator + ε * P(y|contamination)
        contamination_term = self.epsilon_contamination * self.contamination_prob
        denominator = numerator + contamination_term

        return numerator / denominator

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> RobustLogisticRegression:
        """Fit the robust logistic regression model using weighted Fisher scoring."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        X = np.array(X)
        y = np.array(y).reshape(-1)

        # Store for statistics computation
        self.fitted_X = X.copy()

        self.classes_ = np.unique(y)

        # Initialize bias term if use_bias is True
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Initialize weights (beta) to zero
        self.beta = np.zeros(X.shape[1])

        for iteration in range(self.max_iter):
            # Compute predictions
            z = X @ self.beta
            p = self.logistic_function(z)

            # Compute robust weights
            robust_weights = self._compute_robust_weights(X, y, p)

            # Store current weights for later use
            self.weights = robust_weights

            # Weighted diagonal matrix for Fisher information
            W_diag = robust_weights * p * (1 - p)

            # Score vector (gradient)
            score = X.T @ (robust_weights * (y - p))

            if self.information == "expected":
                # Expected Fisher Information matrix (weighted)
                information_matrix = X.T @ (W_diag[:, np.newaxis] * X)
            elif self.information == "empirical":
                # Empirical Fisher Information matrix (weighted)
                weighted_residuals = robust_weights * (y - p)
                weighted_score_vectors = weighted_residuals[:, np.newaxis] * X
                information_matrix = weighted_score_vectors.T @ weighted_score_vectors
            else:
                raise ValueError(
                    f"Unknown Fisher Information type: {self.information}. Use 'expected' or 'empirical'."
                )

            # Track iteration information
            self.information_matrix["iteration"].append(iteration)
            self.information_matrix["information"].append(information_matrix)

            # Compute weighted log-likelihood
            loss = self.compute_loss(y, p, robust_weights)
            self.loss_history.append(loss)

            if self.verbose:
                avg_weight = np.mean(robust_weights)
                if iteration == 0:
                    print("Starting Robust Fisher Scoring Iterations...")
                print(
                    f"Iteration: {iteration + 1}, Weighted Log-Likelihood: {loss:.4f}, Avg Weight: {avg_weight:.3f}"
                )

                # Update beta using the robust Fisher scoring algorithm
            beta_new = self.beta + self.invert_matrix(information_matrix) @ score

            # Check for convergence
            if np.linalg.norm(beta_new - self.beta) < self.tol:
                if self.verbose:
                    print(f"Convergence reached after {iteration + 1} iterations.")
                self.beta = beta_new
                break

            self.beta = beta_new
            self.beta_history.append(self.beta.copy())

            if iteration == self.max_iter - 1:
                print("Maximum iterations reached without convergence.")

        self.compute_statistics()
        self.is_fitted_ = True
        return self

    def compute_statistics(self) -> None:
        """Compute the standard errors, Wald statistic, p-values, and confidence intervals."""
        if not self.information_matrix["information"]:
            return

        information_matrix = self.information_matrix["information"][-1]
        information_matrix_inv = self.invert_matrix(information_matrix)

        self.standard_errors = np.sqrt(np.diagonal(information_matrix_inv))

        # Handle division by zero
        std_errors_safe = np.where(
            self.standard_errors == 0, 1e-10, self.standard_errors
        )

        self.wald_statistic = self.beta / std_errors_safe
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.wald_statistic)))

        critical_value = norm.ppf(1 - self.significance / 2)
        self.lower_bound = self.beta - critical_value * self.standard_errors
        self.upper_bound = self.beta + critical_value * self.standard_errors

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
        proba_class_0 = 1 - proba_class_1
        return np.hstack((proba_class_0.reshape(-1, 1), proba_class_1.reshape(-1, 1)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target labels for the input data X.
        """
        predicted_proba = self.predict_proba(X)[:, 1]
        return (predicted_proba > 0.5).astype(int)

    def predict_ci(self, X: np.ndarray, method: str = "logit") -> np.ndarray:
        """
        Compute confidence intervals for predicted probabilities or logits.

        Parameters:
            X (numpy.ndarray): Input data matrix.
            method (str): Confidence interval method, "logit" (default) or "proba".

        Returns:
            np.ndarray: Array with lower and upper confidence intervals for predictions.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                "Model must be fitted before computing confidence intervals."
            )

        X = np.array(X)
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        logit = X @ self.beta
        proba = self.logistic_function(logit)
        information_matrix = self.information_matrix["information"][-1]
        cov_matrix = self.invert_matrix(information_matrix)
        z_crit = norm.ppf(1 - self.significance / 2)

        if method == "logit":
            # Linear logit confidence intervals
            std_errors = np.array(
                [np.sqrt(np.dot(np.dot(x, cov_matrix), x)) for x in X]
            )
            lower_proba = self.logistic_function(logit - z_crit * std_errors)
            upper_proba = self.logistic_function(logit + z_crit * std_errors)
        elif method == "proba":
            # Probability confidence intervals using delta method
            gradients = (proba * (1 - proba))[:, None] * X
            std_errors = np.sqrt(np.sum(gradients @ cov_matrix * gradients, axis=1))
            lower_proba = np.clip(proba - z_crit * std_errors, 0, 1)
            upper_proba = np.clip(proba + z_crit * std_errors, 0, 1)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'logit' or 'proba'.")

        return np.vstack((lower_proba, upper_proba)).T

    def get_params(self, deep: bool = True) -> Dict[str, Union[float, int, str, bool]]:
        """Get parameters for this estimator."""
        return {
            "epsilon_contamination": self.epsilon_contamination,
            "contamination_prob": self.contamination_prob,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "information": self.information,
            "significance": self.significance,
            "use_bias": self.use_bias,
            "verbose": self.verbose,
        }

    def set_params(
        self, **params: Union[float, int, str, bool]
    ) -> RobustLogisticRegression:
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def summary(self) -> Dict[str, np.ndarray]:
        """Get a summary of the model parameters, standard errors, p-values, and confidence intervals."""
        if not self.is_fitted_:
            raise NotFittedError("Model must be fitted before getting summary.")

        return {
            "betas": self.beta,
            "standard_errors": self.standard_errors,
            "wald_statistic": self.wald_statistic,
            "p_values": self.p_values,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "robust_weights": self.weights,
        }

    def display_summary(self, style: str = "default") -> None:
        """
        Display a summary for IPython notebooks or console output.

        Args:
            style (str): The style for the summary output.
        """
        if not self.is_fitted_:
            raise NotFittedError("Model must be fitted before displaying summary.")

        console = Console()
        summary_dict = self.summary()

        total_iterations = len(self.information_matrix["iteration"])
        table = Table(title="Robust Fisher Scoring Logistic Regression Summary")

        table.add_column("Parameter", justify="right", style=style, no_wrap=True)
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

        # Robust-specific information
        avg_weight = (
            np.mean(summary_dict["robust_weights"])
            if summary_dict["robust_weights"] is not None
            else 0
        )
        min_weight = (
            np.min(summary_dict["robust_weights"])
            if summary_dict["robust_weights"] is not None
            else 0
        )

        summary_stats = f"""
        Total Fisher Scoring Iterations: [{style}]{total_iterations}[/{style}]
        Weighted Log Likelihood: [{style}]{self.loss_history[-1]:.4f}[/{style}]
        Beta 0 = intercept (bias): [{style}]{self.use_bias}[/{style}]
        Epsilon Contamination: [{style}]{self.epsilon_contamination:.3f}[/{style}]
        Average Robust Weight: [{style}]{avg_weight:.3f}[/{style}]
        Minimum Robust Weight: [{style}]{min_weight:.3f}[/{style}]
        """

        console.print(
            Panel.fit(
                summary_stats,
                title="Robust Fisher Scoring Logistic Regression Fit",
                safe_box=True,
            )
        )
        console.print(table)
