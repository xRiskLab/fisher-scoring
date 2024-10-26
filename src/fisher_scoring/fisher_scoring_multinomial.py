"""
Author: xRiskLab (deburky)
GitHub: github.com/xRiskLab
Version: 2.0.2
2024 MIT License

Fisher Scoring Multinomial Regression
-------------------------------------

This is a Python implementation of the Fisher Scoring algorithm for multinomial 
logistic regression. The Fisher Scoring algorithm is an iterative optimization
algorithm that is used to estimate the parameters of a multinomial logistic 
regression model. 

The algorithm is based on the Newton-Raphson method and uses the expected or
observed Fisher information matrix to update the model parameters.

Additionally we provide a method to compute the standard errors, Wald statistic,
p-values, and confidence intervals for each class.

References:

Christopher M. Bishop. Pattern Recognition and Machine Learning. Springer, 2006.

Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning: 
Data Mining, Inference, and Prediction (2nd ed.). Springer, 2009.

Dan Jurafsky and James H. Martin. Speech and Language Processing, 2024.
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


class FisherScoringMultinomialRegression(BaseEstimator, ClassifierMixin):
    """
    Fisher Scoring Multinomial Logistic Regression class.
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
        self.verbose = verbose
        self.beta: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        self.beta_history: List[np.ndarray] = []
        self.information_matrix: Dict[str, List[np.ndarray]] = {
            "iteration": [],
            "information": [],
        }
        self.is_fitted_: bool = False
        self.feature_names: Optional[List[str]] = None
        self.statistics: Dict[str, Dict[str, np.ndarray]] = {}

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
            print("WARNING: Singular matrix. Using pseudo-inverse.")
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
        self.classes_ = np.unique(y)

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
            score = X.T @ (y_one_hot - p)

            if self.information == "expected":
                # Expected Fisher Information matrix
                W_diag = (p * (1 - p)).sum(axis=1)
                expected_I = (X.T * W_diag) @ X
            else:
                # Observed Fisher Information matrix
                score_vector = (y_one_hot - p).reshape(X.shape[0], -1, 1)
                X_vector = X.reshape(X.shape[0], -1, 1)
                observed_I = np.sum(
                    X_vector @ score_vector.transpose(0, 2, 1) @ score_vector @ X_vector.transpose(0, 2, 1),
                    axis=0
                )
            
            # Select information matrix based on expected or observed
            information_matrix = expected_I if self.information == "expected" else observed_I
            self.information_matrix["iteration"].append(iteration)
            self.information_matrix["information"].append(information_matrix)
            
            # Calculate and log the loss
            loss = self.compute_loss(y_one_hot, p)
            self.loss_history.append(loss)
            log_loss = -loss / X.shape[0]

            if self.verbose:
                if iteration == 0:
                    print("Starting Fisher Scoring Iterations...")
                print(f"Iteration: {iteration + 1}, Log Loss: {log_loss:.4f}")

            # Update beta with optimized matrix inversion and step update
            new_beta = self.invert_matrix(information_matrix) @ score
            self.beta += new_beta

            # Check for convergence
            if np.linalg.norm(new_beta) < self.epsilon:
                print(f"Convergence reached after {iteration + 1} iterations.")
                break

        self.compute_statistics()
        self.is_fitted_ = True
        return self

    def compute_statistics(self) -> None:
        """
        Compute the standard errors, Wald statistic, p-values, and confidence intervals for each class.
        """
        n_classes = self.beta.shape[1]  # Number of classes

        self.statistics = {}  # Initialize the statistics dictionary

        for k in range(n_classes):
            # Use the correct information matrix for the k-th class
            information_matrix = self.information_matrix["information"][
                -1
            ]  # Last information matrix (MLE)

            # Invert the information matrix
            information_matrix_inv = np.linalg.pinv(information_matrix)

            # Extract standard errors for the k-th class
            standard_errors = np.sqrt(np.diagonal(information_matrix_inv))
            betas = self.beta[:, k]  # Coefficients for the k-th class

            # Wald statistics
            wald_statistic = betas / standard_errors

            # p-values
            p_values = 2 * (1 - norm.cdf(np.abs(wald_statistic)))

            # Confidence intervals
            critical_value = norm.ppf(1 - self.significance / 2)
            lower_bound = betas - critical_value * standard_errors
            upper_bound = betas + critical_value * standard_errors

            # Store computed statistics in the dictionary for the k-th class
            self.statistics[f"Class_{k}"] = {
                "betas": betas,
                "standard_errors": standard_errors,
                "wald_statistic": wald_statistic,
                "p_values": p_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

    def summary(self, class_idx: int) -> Dict[str, np.ndarray]:
        """
        Get a summary of the model parameters, standard errors, Wald statistics, p-values, and confidence intervals.
        """
        return self.statistics.get(f"Class_{class_idx}", {})

    def display_summary(self, class_idx: int, style="default") -> None:
        """
        Display a summary for IPython notebooks or console output for a given class index.
        Args:
            class_idx (int): The index of the class for which to display the summary.
            style (str): The style for the summary output.
        """
        console = Console()
        summary_dict = self.summary(class_idx)

        total_iterations = len(self.information_matrix["iteration"])
        table = Table(
            title=f"Fisher Scoring Multinomial Regression Summary for Class {class_idx}"
        )

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

        summary_stats = f"""
        Total Fisher Scoring Iterations: [{style}]{total_iterations}[/{style}]
        Log Likelihood: [{style}]{self.loss_history[-1]:.4f}[/{style}]
        Beta 0 = intercept (bias): [{style}]{self.use_bias}[/{style}]
        """

        console.print(
            Panel.fit(
                summary_stats,
                title=f"Fisher Scoring Multinomial Regression Fit for Class {class_idx}",
                safe_box=True,
            )
        )
        console.print(table)

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

    def get_params(self, deep: bool = True) -> Dict[str, Union[float, int, str, bool]]:
        return {
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
            "information": self.information,
            "significance": self.significance,
            "use_bias": self.use_bias,
            "verbose": self.verbose,
        }

    def set_params(
        self, **params: Union[float, int, str, bool]
    ) -> "FisherScoringMultinomialRegression":
        for key, value in params.items():
            setattr(self, key, value)
        return self