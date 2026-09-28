"""
fisher_scoring_multinomial.py.

Multinomial Logistic Regression
-------------------------------------

Author: xRiskLab (deburky)
GitHub: github.com/xRiskLab
License: MIT

This is a Python implementation of the Fisher Scoring algorithm for multinomial
logistic regression. The Fisher Scoring algorithm is an iterative optimization
algorithm that is used to estimate the parameters of a multinomial logistic
regression model.

Additionally we provide a method to compute the standard errors, Wald statistic,
p-values, and confidence intervals for each class.

The model keeps one coefficient vector per class (K classes, as in scikit-learn)
rather than K - 1 contrasts against a reference class. Because the softmax is
unchanged when the same vector is added to every class, the coefficients are
identified by a sum-to-zero constraint across classes. The Fisher information is
the full (p K) x (p K) matrix

    I = sum_i (diag(p_i) - p_i p_i') kron x_i x_i'

with parameters stacked class by class. It is singular along the sum-to-zero
directions, so the step and the covariance use its pseudo-inverse, obtained by
adding a rank-p term on those directions (the score is orthogonal to them). The
covariance of any class contrast beta_k - beta_j then matches a reference-class
fit such as statsmodels MNLogit.

References:

Christopher M. Bishop. Pattern Recognition and Machine Learning. Springer, 2006.

Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning:
Data Mining, Inference, and Prediction (2nd ed.). Springer, 2009.

Dan Jurafsky and James H. Martin. Speech and Language Processing, 2024.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.special import xlogy
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from ._typing import MatrixLike, VectorLike


class MultinomialLogisticRegression(ClassifierMixin, BaseEstimator):
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
        self.information_matrix: Dict[str, List[Union[int, np.ndarray]]] = {
            "iteration": [],
            "information": [],
        }
        self.is_fitted_: bool = False
        self.feature_names: Optional[List[str]] = None
        self.statistics: Dict[str, Dict[str, np.ndarray]] = {}
        self.covariance_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.log_likelihood_: Optional[float] = None
        self._fit_data: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @staticmethod
    def softmax_function(z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax function for the input array z.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        result: np.ndarray = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return result

    @staticmethod
    def compute_loss(y: np.ndarray, p: np.ndarray) -> float:
        """
        Compute the log likelihood loss for multinomial logistic regression.
        """
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return float(np.sum(xlogy(y, p)))

    @staticmethod
    def invert_matrix(matrix: np.ndarray, cond_threshold: float = 1e12) -> np.ndarray:
        """
        Invert a matrix, falling back to the pseudo-inverse
        if the matrix is singular or near-singular.

        Uses the condition number to detect near-singularity,
        since np.linalg.inv silently returns garbage for
        ill-conditioned matrices without raising an error.
        """
        if not np.all(np.isfinite(matrix)):
            cond = np.inf
        else:
            cond = np.linalg.cond(matrix)
        if cond > cond_threshold:
            warnings.warn(
                f"Near-singular information matrix (condition number: {cond:.2e}). "
                "Using pseudo-inverse. Results may be unreliable due to "
                "multicollinearity or quasi-complete separation.",
                stacklevel=2,
            )
            try:
                return np.linalg.pinv(matrix)
            except np.linalg.LinAlgError:
                return np.zeros_like(matrix)
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    @staticmethod
    def compute_information_matrix(
        X: np.ndarray,
        y_one_hot: np.ndarray,
        p: np.ndarray,
        information: str = "expected",
    ) -> np.ndarray:
        """
        Full (p K) x (p K) Fisher information, parameters stacked class by class.

        Expected: sum_i (diag(p_i) - p_i p_i') kron x_i x_i'. Empirical: the sum
        of outer products of the per-observation scores (y_i - p_i) kron x_i.
        Both are built with a single matrix product over an (n, K p) array.
        """
        n, n_features = X.shape
        n_classes = p.shape[1]
        if information == "expected":
            # (n, K, p) -> (n, K p): row i holds p_ik x_i for every class k
            weighted = (p[:, :, None] * X[:, None, :]).reshape(n, -1)
            info = -(weighted.T @ weighted)
            for k in range(n_classes):
                block = slice(k * n_features, (k + 1) * n_features)
                info[block, block] += (X.T * p[:, k]) @ X
            return np.asarray(info)
        if information == "empirical":
            scores = ((y_one_hot - p)[:, :, None] * X[:, None, :]).reshape(n, -1)
            return np.asarray(scores.T @ scores)
        raise ValueError(
            f"Unknown Fisher Information type: {information}. "
            "Use 'expected' or 'empirical'."
        )

    @staticmethod
    def _centering_projector(n_features: int, n_classes: int) -> np.ndarray:
        """Projector onto the sum-to-zero directions (sum over classes = 0)."""
        mean_over_classes = np.kron(
            np.full((n_classes, n_classes), 1.0 / n_classes), np.eye(n_features)
        )
        return np.asarray(np.eye(n_features * n_classes) - mean_over_classes)

    def _constrained_inverse(self, information_matrix: np.ndarray) -> np.ndarray:
        """
        Pseudo-inverse of the information on the sum-to-zero subspace.

        The information vanishes on the directions that add the same vector to
        every class. Adding c * (11' kron I) on those directions makes it
        invertible without changing it on the sum-to-zero subspace; projecting
        the inverse back onto that subspace gives the Moore-Penrose inverse.
        """
        n_params = information_matrix.shape[0]
        assert self.beta is not None
        n_features, n_classes = self.beta.shape
        c = np.trace(information_matrix) / n_params
        if not np.isfinite(c) or c <= 0:
            c = 1.0
        augmented = information_matrix + c * np.kron(
            np.ones((n_classes, n_classes)), np.eye(n_features)
        )
        projector = self._centering_projector(n_features, n_classes)
        return np.asarray(projector @ self.invert_matrix(augmented) @ projector)

    def fit(
        self,
        X: MatrixLike,
        y: VectorLike,
    ) -> MultinomialLogisticRegression:
        """
        Fit the multinomial logistic regression model using Fisher scoring.
        """
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None

        X = np.asarray(X, dtype=np.float64)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y_idx] = 1

        # Initialize bias term if use_bias is True
        if self.use_bias:
            X = np.hstack([np.ones((n_samples, 1)), X])
        n_features = X.shape[1]

        # Reset fitted state so that refitting does not accumulate history
        self.beta = np.zeros((n_features, n_classes))
        self.loss_history = []
        self.beta_history = []
        self.information_matrix = {"iteration": [], "information": []}

        for iteration in range(self.max_iter):
            p = self.softmax_function(X @ self.beta)
            # Score stacked class by class, matching the information layout
            score = (X.T @ (y_one_hot - p)).T.ravel()
            information_matrix = self.compute_information_matrix(
                X, y_one_hot, p, self.information
            )
            self.information_matrix["iteration"].append(iteration)
            self.information_matrix["information"].append(information_matrix)

            # Calculate and log the loss
            loss = self.compute_loss(y_one_hot, p)
            self.loss_history.append(loss)
            if self.verbose:
                if iteration == 0:
                    print("Starting Fisher Scoring Iterations...")
                print(f"Iteration: {iteration + 1}, Log Loss: {-loss / n_samples:.4f}")

            step = self._constrained_inverse(information_matrix) @ score
            self.beta = self.beta + step.reshape(n_classes, n_features).T
            self.beta_history.append(self.beta.copy())
            self.n_iter_ = iteration + 1

            if np.linalg.norm(step) < self.epsilon:
                if self.verbose:
                    print(f"Convergence reached after {iteration + 1} iterations.")
                break
        else:
            warnings.warn(
                "Maximum iterations reached without convergence.", stacklevel=2
            )

        self._fit_data = (X, y_one_hot)
        self.compute_statistics()
        self.is_fitted_ = True
        return self

    @property
    def coef_(self) -> np.ndarray:
        """Coefficients per class, shape (n_classes, n_features), as in sklearn."""
        assert self.beta is not None, "Model has not been fitted yet."
        coefficients = self.beta[1:] if self.use_bias else self.beta
        return np.asarray(coefficients.T)

    @property
    def intercept_(self) -> np.ndarray:
        """Intercept per class, shape (n_classes,); zeros when use_bias=False."""
        assert self.beta is not None, "Model has not been fitted yet."
        if self.use_bias:
            return np.asarray(self.beta[0])
        return np.zeros(self.beta.shape[1])

    def compute_statistics(self) -> None:
        """
        Compute the standard errors, Wald statistic, p-values, and confidence intervals for each class.
        """
        assert self.beta is not None and self._fit_data is not None
        n_features, n_classes = self.beta.shape
        # Evaluate at the returned coefficients: loss_history and information_matrix
        # hold the values at the start of each iteration, before its update.
        X, y_one_hot = self._fit_data
        p = self.softmax_function(X @ self.beta)
        self.log_likelihood_ = self.compute_loss(y_one_hot, p)
        info = self.compute_information_matrix(X, y_one_hot, p, self.information)
        self.covariance_ = self._constrained_inverse(info)
        variances = np.clip(np.diagonal(self.covariance_), 0.0, None)
        critical_value = norm.ppf(1 - self.significance / 2)

        self.statistics = {}
        for k in range(n_classes):
            betas = self.beta[:, k]
            standard_errors = np.sqrt(variances[k * n_features : (k + 1) * n_features])
            with np.errstate(divide="ignore", invalid="ignore"):
                wald_statistic = betas / standard_errors
            self.statistics[f"Class_{k}"] = {
                "betas": betas,
                "standard_errors": standard_errors,
                "wald_statistic": wald_statistic,
                "p_values": 2 * (1 - norm.cdf(np.abs(wald_statistic))),
                "lower_bound": betas - critical_value * standard_errors,
                "upper_bound": betas + critical_value * standard_errors,
            }

    def summary(self, class_idx: int) -> Dict[str, np.ndarray]:
        """
        Get a summary of the model parameters, standard errors, Wald statistics, p-values, and confidence intervals.
        """
        return self.statistics.get(f"Class_{class_idx}", {})

    def display_summary(self, class_idx: int, style: str = "default") -> None:
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
        Log Likelihood: [{style}]{self.log_likelihood_:.4f}[/{style}]
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

    def predict_proba(self, X: MatrixLike) -> np.ndarray:
        """
        Predict the class probabilities for the input data X.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                "This Classifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments "
                "before using this estimator."
            )
        X = np.asarray(X, dtype=np.float64)
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        assert self.beta is not None
        return self.softmax_function(X @ self.beta)

    def predict(self, X: MatrixLike) -> np.ndarray:
        """
        Predict the target labels (values from `classes_`) for the input data X.
        """
        probas = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(probas, axis=1)])

    def predict_ci(self, X: MatrixLike, method: str = "logit") -> Dict[int, np.ndarray]:
        """
        Compute confidence intervals for predicted probabilities or logits for each class.

        Parameters:
            X (numpy.ndarray): Input data matrix.
            method (str): Confidence interval method, "logit" (default) or "proba".

        Returns:
            Dict[int, np.ndarray]: A dictionary where the key is the class index and the value is a
            2D array with lower and upper confidence intervals for predictions of that class.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                "This Classifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments "
                "before using this estimator."
            )

        X = np.asarray(X, dtype=np.float64)
        if self.use_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        assert self.beta is not None and self.covariance_ is not None
        n_features, n_classes = self.beta.shape
        logits = X @ self.beta
        probabilities = self.softmax_function(logits)
        z_crit = norm.ppf(1 - self.significance / 2)  # Critical value for CI

        ci_results = {}
        for class_idx in range(n_classes):
            block = slice(class_idx * n_features, (class_idx + 1) * n_features)
            probabilities_k = probabilities[:, class_idx]

            if method == "logit":
                # Interval for the class logit x' beta_k, mapped through the softmax
                # with the other logits held at their estimates.
                cov_k = self.covariance_[block, block]
                std_errors = np.sqrt(
                    np.clip(np.einsum("ij,jk,ik->i", X, cov_k, X), 0.0, None)
                )
                lower_logits = logits.copy()
                upper_logits = logits.copy()
                lower_logits[:, class_idx] -= z_crit * std_errors
                upper_logits[:, class_idx] += z_crit * std_errors
                lower_ci = self.softmax_function(lower_logits)[:, class_idx]
                upper_ci = self.softmax_function(upper_logits)[:, class_idx]
            elif method == "proba":
                # Delta method: d p_k / d beta_l = p_k (1[k = l] - p_l) x
                weights = -probabilities_k[:, None] * probabilities
                weights[:, class_idx] += probabilities_k
                gradients = (weights[:, :, None] * X[:, None, :]).reshape(
                    X.shape[0], -1
                )
                std_errors = np.sqrt(
                    np.clip(
                        np.einsum(
                            "ij,jk,ik->i", gradients, self.covariance_, gradients
                        ),
                        0.0,
                        None,
                    )
                )
                lower_ci = np.clip(probabilities_k - z_crit * std_errors, 0, 1)
                upper_ci = np.clip(probabilities_k + z_crit * std_errors, 0, 1)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'logit' or 'proba'.")

            ci_results[class_idx] = np.vstack((lower_ci, upper_ci)).T

        return ci_results

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
    ) -> MultinomialLogisticRegression:
        for key, value in params.items():
            setattr(self, key, value)
        return self
