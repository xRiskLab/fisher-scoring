"""
fisher_scoring_logistic.py.

Logistic Regression
----------------------------------

Author: xRiskLab (deburky)
GitHub: https://github.com/xRiskLab)
License: MIT

Description:
This module contains the `LogisticRegression` class, which is a custom
implementation of logistic regression using the Fisher scoring algorithm. The
Fisher scoring algorithm is an iterative optimization algorithm that uses the
empirical or expected information matrix to update the model parameters. The class provides
methods for fitting the model, making predictions, and computing model statistics.

We provide two types of information matrices: 'expected' and 'empirical'. The 'expected'
information matrix is computed using the Hessian matrix, while the 'empirical' information
matrix is computed using the outer product of the score vectors (variance of the score).

If `use_bias` is set to True, the model will include a bias term in the logistic regression
equation. If `use_bias` is set to True, we add the bias term before the input features in
the design matrix. In the summary output, the bias term will be included as the first element.

References:

Erich L. Lehmann and George Casella. Theory of Point Estimation (2nd ed.). Springer, 1998.

Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. An Introduction to Statistical
Learning: with Applications in Python. Springer, 2023.

Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning:
Data Mining, Inference, and Prediction (2nd ed.). Springer, 2009.

Yudi Pawitan. In All Likelihood: Statistical Modelling and Inference Using Likelihood. Oxford University
Press, 2001.
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


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Fisher Scoring Logistic Regression class.
    """

    def __init__(
        self,
        epsilon: float = 1e-10,
        max_iter: int = 100,
        information: str = "expected",
        use_bias: bool = True,
        significance: float = 0.05,
        verbose: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.information = information
        self.use_bias = use_bias
        self.significance = significance
        self.bias: Optional[float] = None
        self.beta: Optional[np.ndarray] = None
        self.information_matrix: Dict[str, List[np.ndarray]] = {
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
        self.verbose = verbose
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
    def compute_loss(y: np.ndarray, p: np.ndarray) -> float:
        """
        Compute the log-likelihood loss for logistic regression.
        """
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.sum(xlogy(y, p) + xlogy(1 - y, 1 - p))

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> LogisticRegression:
        """Fit the logistic regression model using Fisher scoring."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        self.classes_ = np.unique(y)

        # Initialize bias term if use_bias is True
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Initialize weights (beta) to zero
        self.beta = np.zeros((X.shape[1], 1))

        for iteration in range(self.max_iter):
            p = self.logistic_function(X @ self.beta)
            score_vector = (y - p) * X
            score = np.sum(score_vector, axis=0).reshape(-1, 1)

            if self.information == "expected":
                # Expected Fisher Information matrix
                W_diag = (p * (1 - p)).ravel()
                information_matrix = (X.T * W_diag) @ X
            else:
                # Empirical Fisher Information matrix
                score_vector = (y - p).reshape(X.shape[0], 1, 1)
                X_vector = X.reshape(X.shape[0], -1, 1)
                information_matrix = np.sum(
                    X_vector
                    @ score_vector.transpose(0, 2, 1)
                    @ score_vector
                    @ X_vector.transpose(0, 2, 1),
                    axis=0,
                )

            self.information_matrix["iteration"].append(iteration)
            self.information_matrix["information"].append(information_matrix)

            loss = self.compute_loss(y, p)
            log_loss = -loss / X.shape[0]

            self.loss_history.append(loss)

            if self.verbose:
                loss = self.compute_loss(y, p) / X.shape[0]
                if iteration == 0:
                    print("Starting Fisher Scoring Iterations...")
                print(f"Iteration: {iteration + 1}, Log Loss: {log_loss:.4f}")

            # Update beta using the Fisher scoring algorithm
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
        """Compute the standard errors, Wald statistic, p-values, and confidence intervals."""
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
        proba_class_0 = 1 - proba_class_1
        return np.hstack((proba_class_0, proba_class_1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target labels for the input data X.
        """
        predicted_proba = self.predict_proba(X)[:, 1]
        return (predicted_proba > 0.5).astype(int)

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
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
            "information": self.information,
            "significance": self.significance,
            "use_bias": self.use_bias,
        }

    def set_params(self, **params: Union[float, int, str, bool]) -> LogisticRegression:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def summary(self) -> Dict[str, np.ndarray]:
        """Get a summary of the model parameters, standard errors, p-values, and confidence intervals."""
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
        table = Table(title="Fisher Scoring Logistic Regression Summary")

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
        Log Likelihood: [{style}]{self.loss_history[-1]:.4f}[/{style}]
        Beta 0 = intercept (bias): [{style}]{self.use_bias}[/{style}]
        """

        console.print(
            Panel.fit(
                summary_stats,
                title="Fisher Scoring Logistic Regression Fit",
                safe_box=True,
            )
        )
        console.print(table)
