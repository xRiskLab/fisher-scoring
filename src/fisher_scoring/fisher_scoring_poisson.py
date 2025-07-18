"""
fisher_scoring_poisson.py.

Fisher Scoring Poisson Regression
----------------------------------

Author: xRiskLab (deburky)
GitHub: https://github.com/xRiskLab)
License: MIT

This module implements Poisson and Negative Binomial regression using the Fisher scoring method.
The implementation details are from J. Hilbe. Modeling Count Data. Cambridge University Press, 2014.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.stats import norm


# pylint: disable=invalid-name
class PoissonRegression:
    """Poisson regression using Fisher scoring method."""

    def __init__(
        self,
        max_iter=100,
        epsilon=1e-5,
        use_bias=True,
        offset=None,
        significance=0.05,
        information="expected",
    ):
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.weights = None
        self.offset = offset
        self.significance = significance
        self.information = information
        self.standard_errors: Optional[np.ndarray] = None
        self.wald_statistic: Optional[np.ndarray] = None
        self.p_values: Optional[np.ndarray] = None
        self.lower_bound: Optional[np.ndarray] = None
        self.upper_bound: Optional[np.ndarray] = None
        self.fitted_X: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.information_matrix: Dict[str, List] = {
            "iteration": [],
            "information": [],
        }
        self.loss_history: List[float] = []

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

    def fit(self, x, y) -> PoissonRegression:
        """
        Fit the Poisson regression model using Fisher scoring.
        """
        # Extract feature names if x is a pandas DataFrame
        if hasattr(x, "columns"):
            self.feature_names = x.columns.tolist()

        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Add a column of ones to X for the intercept
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        if self.offset is None:
            self.offset = np.zeros(len(y))  # Default offset
        self.weights = np.zeros(X.shape[1])  # Initialize weights (p+1,)

        # Store for statistics computation
        self.fitted_X = X

        for iteration in range(self.max_iter):
            # Linear predictor
            eta = X @ self.weights + self.offset
            # Clip eta to prevent overflow in exp(eta)
            eta = np.clip(eta, -500, 500)
            mu = np.exp(eta)  # Mean prediction (inverse link)

            # Poisson score vector
            score = X.T @ (y - mu)

            # Poisson information matrix
            if self.information == "expected":
                # Expected Fisher Information matrix
                W = np.diag(mu)  # Diagonal weight matrix
                information = X.T @ W @ X
            elif self.information == "empirical":
                # Empirical Fisher Information matrix
                residuals = y - mu
                score_vectors = residuals[:, np.newaxis] * X
                information = score_vectors.T @ score_vectors
            else:
                raise ValueError(
                    f"Unknown Fisher Information type: {self.information}. Use 'expected' or 'empirical'."
                )

            # Compute Poisson log-likelihood (excluding factorial term)
            # The -log(y!) term is constant w.r.t. parameters and can be omitted for optimization
            log_likelihood = np.sum(y * eta - mu)
            self.loss_history.append(log_likelihood)
            self.information_matrix["iteration"].append(iteration)
            self.information_matrix["information"].append(information)

            # Update weights using Fisher scoring method
            beta_new = self.weights + self.invert_matrix(information) @ score

            # Check for convergence
            if np.linalg.norm(beta_new - self.weights) < self.epsilon:
                self.weights = beta_new
                print(f"Converged in {iteration + 1} iterations.")
                break

            self.weights = beta_new
        else:
            print("Did not converge within the maximum number of iterations.")

        # Compute statistics for summary output
        self.compute_statistics()
        return self

    def calculate_st_errors(self, x):
        """
        Calculate standard errors for the coefficients.
        """
        # Add a column of ones to X for the intercept
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        eta = X @ self.weights + self.offset
        mu = np.exp(eta)
        W = np.diag(mu)
        information = X.T @ W @ X
        return np.sqrt(np.diag(self.invert_matrix(information)))

    def predict(self, x, offset=None):
        """
        Predict mean values for the Poisson model.

        Parameters:
        - x: Input features
        - offset: Optional offset for prediction. If None, uses the training offset
                 (expanded to match prediction data size if needed)
        """
        # Predict mean values
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x

        if offset is None:
            # Use training offset, handling size mismatch
            if self.offset is None:
                offset = np.zeros(X.shape[0])

            elif len(self.offset) == 1:
                offset = np.full(X.shape[0], self.offset[0])
            elif len(self.offset) == X.shape[0]:
                offset = self.offset
            else:
                # Default to zeros if training offset size doesn't match
                offset = np.zeros(X.shape[0])
        eta = X @ self.weights + offset
        return np.exp(eta)  # Return mean predictions (inverse link)

    def compute_statistics(self) -> None:
        """Compute the standard errors, Wald statistic, p-values, and confidence intervals."""
        if self.fitted_X is None:
            raise ValueError("Model must be fitted before computing statistics.")

        # Compute information matrix
        eta = self.fitted_X @ self.weights + self.offset
        mu = np.exp(eta)
        W = np.diag(mu)
        information_matrix = self.fitted_X.T @ W @ self.fitted_X

        # Compute standard errors
        information_matrix_inv = self.invert_matrix(information_matrix)
        self.standard_errors = np.sqrt(np.diagonal(information_matrix_inv))

        # Compute Wald statistics and p-values
        if np.any(self.standard_errors == 0):
            print("WARNING: Some standard errors are zero.")

        self.wald_statistic = self.weights / self.standard_errors
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.wald_statistic)))

        # Compute confidence intervals
        critical_value = norm.ppf(1 - self.significance / 2)
        self.lower_bound = self.weights - critical_value * self.standard_errors
        self.upper_bound = self.weights + critical_value * self.standard_errors

    def summary(self) -> Dict[str, np.ndarray]:
        """Get a summary of the model parameters, standard errors, p-values, and confidence intervals."""
        if self.standard_errors is None:
            self.compute_statistics()

        return {
            "coefficients": self.weights,
            "standard_errors": self.standard_errors,
            "wald_statistic": self.wald_statistic,
            "p_values": self.p_values,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    def display_summary(self, style: str = "default") -> None:
        """
        Display a summary for IPython notebooks or console output.

        Args:
            style (str): The style for the summary output.
        """
        console = Console()
        summary_dict = self.summary()

        table = Table(title="Fisher Scoring Poisson Regression Summary")

        table.add_column("Parameter", justify="right", style=style, no_wrap=True)
        table.add_column("Estimate", style=style)
        table.add_column("Std. Error", style=style)
        table.add_column("Wald Statistic", style=style)
        table.add_column("P-value", style=style)
        table.add_column("Lower CI", style=style)
        table.add_column("Upper CI", style=style)

        # Parameter names
        if self.feature_names:
            param_names = (
                ["intercept (bias)"] + self.feature_names
                if self.use_bias
                else self.feature_names
            )
        else:
            param_names = [
                f"Beta {i}" for i in range(len(summary_dict["coefficients"]))
            ]

        for i, param in enumerate(param_names):
            table.add_row(
                f"{param}",
                f"{summary_dict['coefficients'][i]:.4f}",
                f"{summary_dict['standard_errors'][i]:.4f}",
                f"{summary_dict['wald_statistic'][i]:.4f}",
                f"{summary_dict['p_values'][i]:.4f}",
                f"{summary_dict['lower_bound'][i]:.4f}",
                f"{summary_dict['upper_bound'][i]:.4f}",
            )

        total_iterations = len(self.information_matrix["iteration"])

        # Build summary stats like LogisticRegression
        offset_line = ""
        if self.offset is not None and np.any(self.offset != 0):
            offset_line = f"\n        Offset: [{style}]Yes[/{style}]"

        summary_stats = f"""
        Total Fisher Scoring Iterations: [{style}]{total_iterations}[/{style}]
        Log Likelihood: [{style}]{self.loss_history[-1]:.4f}[/{style}]{offset_line}
        Beta 0 = intercept (bias): [{style}]{self.use_bias}[/{style}]
        """

        console.print(
            Panel.fit(
                summary_stats,
                title="Fisher Scoring Poisson Regression Fit",
                safe_box=True,
            )
        )
        console.print(table)


class NegativeBinomialRegression:
    """Fisher scoring method for Negative Binomial regression."""

    def __init__(
        self,
        max_iter=100,
        epsilon=1e-5,
        use_bias=False,
        alpha=1.0,
        phi=1.0,
        offset=None,
        significance=0.05,
        information="expected",
    ):
        """
        Poisson regression with a fixed alpha for dispersion adjustment.

        Parameters:
        - max_iter: Maximum number of iterations for optimization.
        - epsilon: Convergence tolerance.
        - use_bias: Whether to include an intercept term.
        - alpha: Fixed dispersion parameter (overdispersion adjustment for Negative Binomial).
        - phi: Constant scale parameter.
        - offset: Offset term for the linear predictor.
        - significance: Significance level for confidence intervals.
        """
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.weights = None
        self.alpha = alpha  # Fixed overdispersion parameter
        self.phi = phi  # Scale parameter
        self.offset = offset
        self.significance = significance
        self.information = information
        self.standard_errors: Optional[np.ndarray] = None
        self.wald_statistic: Optional[np.ndarray] = None
        self.p_values: Optional[np.ndarray] = None
        self.lower_bound: Optional[np.ndarray] = None
        self.upper_bound: Optional[np.ndarray] = None
        self.fitted_X: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.information_matrix: Dict[str, List] = {
            "iteration": [],
            "information": [],
        }
        self.loss_history: List[float] = []

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

    def fit(self, x, y) -> NegativeBinomialRegression:
        """
        Fit the Negative Binomial regression model using Fisher scoring.
        """
        # Extract feature names if x is a pandas DataFrame
        if hasattr(x, "columns"):
            self.feature_names = x.columns.tolist()

        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Add intercept if necessary
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        if self.offset is None:
            self.offset = np.zeros(len(y))  # Default offset

        # Initialize weights (beta) to zero
        self.weights = np.zeros(X.shape[1])

        # Store for statistics computation
        self.fitted_X = X

        for iteration in range(self.max_iter):
            # Linear predictor
            eta = X @ self.weights + self.offset
            # Clip eta to prevent overflow in exp(eta)
            eta = np.clip(eta, -500, 500)
            mu = np.exp(eta)  # Mean prediction (inverse link)

            # Negative Binomial score vector
            score = X.T @ (y - mu)

            # Negative Binomial information matrix
            if self.information == "expected":
                # Expected Fisher Information matrix
                # For NB: Var(Y) = μ + α*μ² = μ(1 + α*μ)
                variance = mu * (1 + self.alpha * mu)
                # Ensure variance is not zero or inf
                variance = np.clip(variance, 1e-10, 1e10)
                W = np.diag(mu / variance)  # Weight matrix
                information = X.T @ W @ X
            elif self.information == "empirical":
                # Empirical Fisher Information matrix
                residuals = y - mu
                score_vectors = residuals[:, np.newaxis] * X
                information = score_vectors.T @ score_vectors
            else:
                raise ValueError(
                    f"Unknown Fisher Information type: {self.information}. Use 'expected' or 'empirical'."
                )

            # Compute Negative Binomial log-likelihood (simplified)
            log_likelihood = np.sum(y * eta - mu)
            self.loss_history.append(log_likelihood)
            self.information_matrix["iteration"].append(iteration)
            self.information_matrix["information"].append(information)

            # Update weights using Fisher scoring method
            beta_new = self.weights + self.invert_matrix(information) @ score

            # Check for convergence
            if np.linalg.norm(beta_new - self.weights) < self.epsilon:
                self.weights = beta_new
                print(f"Converged in {iteration + 1} iterations.")
                break

            self.weights = beta_new

        else:
            print("Did not converge within the maximum number of iterations.")

        # Compute statistics for summary output
        self.compute_statistics()
        return self

    def calculate_st_errors(self, x):
        """
        Calculate standard errors for the coefficients.
        """
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        eta = X @ self.weights + self.offset
        mu = np.exp(eta)
        variance = mu * (1 + self.alpha * mu)
        W_diag = 1 / (self.phi * variance * (1 / mu) ** 2)
        W = np.diag(W_diag)
        XtWX = X.T @ W @ X
        return np.sqrt(np.diag(self.invert_matrix(XtWX)))

    def predict(self, x, offset=None):
        """
        Predict mean values for the Negative Binomial model.

        Parameters:
        - x: Input features
        - offset: Optional offset for prediction. If None, uses the training offset
                 (expanded to match prediction data size if needed)
        """
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x

        if offset is None:
            # Use training offset, handling size mismatch
            if self.offset is None:
                offset = np.zeros(X.shape[0])

            elif len(self.offset) == 1:
                offset = np.full(X.shape[0], self.offset[0])
            elif len(self.offset) == X.shape[0]:
                offset = self.offset
            else:
                # Default to zeros if training offset size doesn't match
                offset = np.zeros(X.shape[0])
        eta = X @ self.weights + offset
        return np.exp(eta)

    def compute_statistics(self) -> None:
        """Compute the standard errors, Wald statistic, p-values, and confidence intervals."""
        if self.fitted_X is None:
            raise ValueError("Model must be fitted before computing statistics.")

        # Compute information matrix using negative binomial variance structure
        eta = self.fitted_X @ self.weights + self.offset
        mu = np.exp(eta)
        variance = mu * (1 + self.alpha * mu)
        W_diag = 1 / (self.phi * variance * (1 / mu) ** 2)
        W = np.diag(W_diag)
        information_matrix = self.fitted_X.T @ W @ self.fitted_X

        # Compute standard errors
        information_matrix_inv = self.invert_matrix(information_matrix)
        self.standard_errors = np.sqrt(np.diagonal(information_matrix_inv))

        # Compute Wald statistics and p-values
        if np.any(self.standard_errors == 0):
            print("WARNING: Some standard errors are zero.")

        self.wald_statistic = self.weights / self.standard_errors
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.wald_statistic)))

        # Compute confidence intervals
        critical_value = norm.ppf(1 - self.significance / 2)
        self.lower_bound = self.weights - critical_value * self.standard_errors
        self.upper_bound = self.weights + critical_value * self.standard_errors

    def summary(self) -> Dict[str, np.ndarray]:
        """Get a summary of the model parameters, standard errors, p-values, and confidence intervals."""
        if self.standard_errors is None:
            self.compute_statistics()

        return {
            "coefficients": self.weights,
            "standard_errors": self.standard_errors,
            "wald_statistic": self.wald_statistic,
            "p_values": self.p_values,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    def display_summary(self, style: str = "default") -> None:
        """
        Display a summary for IPython notebooks or console output.

        Args:
            style (str): The style for the summary output.
        """
        console = Console()
        summary_dict = self.summary()

        table = Table(title="Fisher Scoring Negative Binomial Regression Summary")

        table.add_column("Parameter", justify="right", style=style, no_wrap=True)
        table.add_column("Estimate", style=style)
        table.add_column("Std. Error", style=style)
        table.add_column("Wald Statistic", style=style)
        table.add_column("P-value", style=style)
        table.add_column("Lower CI", style=style)
        table.add_column("Upper CI", style=style)

        # Parameter names
        if self.feature_names:
            param_names = (
                ["intercept (bias)"] + self.feature_names
                if self.use_bias
                else self.feature_names
            )
        else:
            param_names = [
                f"Beta {i}" for i in range(len(summary_dict["coefficients"]))
            ]

        for i, param in enumerate(param_names):
            table.add_row(
                f"{param}",
                f"{summary_dict['coefficients'][i]:.4f}",
                f"{summary_dict['standard_errors'][i]:.4f}",
                f"{summary_dict['wald_statistic'][i]:.4f}",
                f"{summary_dict['p_values'][i]:.4f}",
                f"{summary_dict['lower_bound'][i]:.4f}",
                f"{summary_dict['upper_bound'][i]:.4f}",
            )

        total_iterations = len(self.information_matrix["iteration"])

        # Build summary stats like LogisticRegression
        offset_line = ""
        if self.offset is not None and np.any(self.offset != 0):
            offset_line = f"\n        Offset: [{style}]Yes[/{style}]"

        summary_stats = f"""
        Total Fisher Scoring Iterations: [{style}]{total_iterations}[/{style}]
        Log Likelihood: [{style}]{self.loss_history[-1]:.4f}[/{style}]{offset_line}
        Beta 0 = intercept (bias): [{style}]{self.use_bias}[/{style}]
        """

        console.print(
            Panel.fit(
                summary_stats,
                title="Fisher Scoring Negative Binomial Regression Fit",
                safe_box=True,
            )
        )
        console.print(table)
