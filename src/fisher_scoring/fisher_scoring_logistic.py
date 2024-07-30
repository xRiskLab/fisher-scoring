"""
Author: xRiskLab (deburky)
GitHub: github.com/xRiskLab
Beta Version: 0.1
2024 MIT License

Fisher Scoring Logistic Regression
----------------------------------

This module contains the `FisherScoringLogisticRegression` class, which is a custom
implementation of logistic regression using the Fisher scoring algorithm. The
Fisher scoring algorithm is an iterative optimization algorithm that uses the
observed or expected information matrix to update the model parameters. The class provides
methods for fitting the model, making predictions, and computing model statistics.

We provide two types of information matrices: 'expected' and 'observed'. The 'expected'
information matrix is computed using the Hessian matrix, while the 'observed' information 
matrix is computed using the score. This allows to understand the difference between the
two types of information matrices and their impact on the model parameters and statistics.

Additionally, the class provides methods for computing the standard errors, Wald statistic,
p-values, and confidence intervals for the model parameters, which are not available in
sci-kit learn's logistic regression implementation. The implementation mirrors the
functionality of the statsmodels library in terms of parameter estimation and inference.

The `FisherScoringLogisticRegression` class is implemented as a scikit-learn compatible
estimator, allowing it to be used in scikit-learn pipelines and workflows. The class
inherits from the BaseEstimator and ClassifierMixin classes, providing common methods
such as get_params and set_params.

The `FisherScoringLogisticRegression` class has the following parameters:

- epsilon: A float value specifying the convergence threshold for the algorithm.
- max_iter: An integer specifying the maximum number of iterations for the algorithm.
- information: A string specifying the type of information matrix to use ('expected' or 'observed').
- use_bias: A boolean indicating whether to include a bias term in the model.
- significance: A float value specifying the significance level for computing confidence intervals.

The class provides the following methods:

- fit(X, y): Fit the logistic regression model to the input data X and target labels y.
- predict(X): Predict the target labels for the input data X.
- predict_proba(X): Predict the class probabilities for the input data X.
- get_params(): Get the parameters of the logistic regression model.
- set_params(**params): Set the parameters of the logistic regression model.
- summary(): Get a summary of the model parameters, standard errors, p-values, and confidence intervals.

If `use_bias` is set to True, the model will include a bias term in the logistic regression
equation. If `use_bias` is set to True, we add the bias term before the input features in 
the design matrix. In the summary output, the bias term will be included as the first element.

References:

Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning: 
Data Mining, Inference, and Prediction (2nd ed.). Springer, 2009.

Erich L. Lehmann and George Casella. Theory of Point Estimation (2nd ed.). Springer, 1998.

Yudi Pawitan. In All Likelihood: Statistical Modelling and Inference Using Likelihood. Oxford University
Press, 2001.
"""

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


class FisherScoringLogisticRegression(
    BaseEstimator, ClassifierMixin
):
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
        self.information_matrix: Dict[
            str, List[np.ndarray]
        ] = {
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
        self.feature_names = None

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
            print(
                "WARNING: Singular matrix. Using pseudo-inverse."
            )
            return np.linalg.pinv(matrix)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "FisherScoringLogisticRegression":
        """
        Fit the logistic regression model using Fisher scoring.
        """

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        # Initialize bias term if use_bias is True
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Initialize weights (beta) to zero
        self.beta = np.zeros((X.shape[1], 1))

        for iteration in range(self.max_iter):
            p = self.logistic_function(X @ self.beta)
            W = np.diag((p * (1 - p)).ravel())

            # Compute Score and Information
            score_vector = (y - p) * X
            score = np.sum(score_vector, axis=0).reshape(
                -1, 1
            )
            expected_I = X.T @ W @ X
            observed_I = score_vector.T @ score_vector

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

            loss = self.compute_loss(y, p)
            log_loss = -loss / X.shape[0]

            self.loss_history.append(loss)

            if self.verbose:
                loss = self.compute_loss(y, p) / X.shape[0]
                if iteration == 0:
                    print(
                        "Starting Fisher Scoring Iterations..."
                    )
                print(
                    f"Iteration: {iteration + 1}, Log Loss: {log_loss:.4f}"
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

        self.compute_statistics()
        self.is_fitted_ = True
        return self

    def compute_statistics(self) -> None:
        """
        Compute the standard errors, Wald statistic, p-values, and confidence intervals.
        """
        information_matrix = self.information_matrix[
            "information"
        ][
            -1
        ]  # Information at the MLE

        if self.information == "expected":
            information_matrix_inv = np.linalg.inv(
                information_matrix
            )
        elif self.information == "observed":
            information_matrix_inv = np.linalg.pinv(
                information_matrix
            )

        self.standard_errors = np.sqrt(
            np.diagonal(information_matrix_inv)
        )
        betas = self.beta.flatten()

        # Handle division by zero
        if self.standard_errors.any() == 0:
            # Raise warning
            print(
                "WARNING: Standard errors are zero. Setting to 1."
            )
        self.wald_statistic = betas / self.standard_errors
        self.p_values = 2 * (
            1 - norm.cdf(np.abs(self.wald_statistic))
        )

        critical_value = norm.ppf(1 - self.significance / 2)
        self.lower_bound = (
            betas - critical_value * self.standard_errors
        )
        self.upper_bound = (
            betas + critical_value * self.standard_errors
        )

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
        proba_class_0 = 1 - proba_class_1
        return np.hstack((proba_class_0, proba_class_1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target labels for the input data X.
        """
        predicted_proba = self.predict_proba(X)[:, 1]
        return (predicted_proba > 0.5).astype(int)

    def get_params(
        self, deep: bool = True
    ) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
            "information": self.information,
            "significance": self.significance,
            "use_bias": self.use_bias,
        }

    def set_params(
        self, **params: Union[float, int, str, bool]
    ) -> "FisherScoringLogisticRegression":
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

        total_iterations = len(
            self.information_matrix["iteration"]
        )
        table = Table(
            title="Fisher Scoring Logistic Regression Summary"
        )

        # style = "blue1"
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
            param_names = [
                f"Beta {i}"
                for i in range(len(summary_dict["betas"]))
            ]
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
