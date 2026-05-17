"""
fisher_scoring_divergence.py.

Divergence Classifier (FICO Divergence Maximization)
-----------------------------------------------------

Author: xRiskLab (deburky)
GitHub: https://github.com/xRiskLab
License: MIT

Description:
This module contains the `DivergenceClassifier` class, which finds the linear
scoring function that maximizes FICO divergence — a standard metric in credit
scoring and scorecard development.

The FICO divergence of a score is defined as:

    Div[Score] = (μ_G - μ_B)² / σ²

where μ_G and μ_B are the mean scores for the Good and Bad classes, and
σ² = (σ²_G + σ²_B) / 2 is the average within-class variance.

This is equivalent to twice the Fisher ratio (J = (μ₁ − μ₀)² / (σ₁² + σ₀²))
because the Fisher ratio uses the sum of variances while FICO divergence uses
the average.

In the unconstrained case, the maximum-divergence weights are the classical
Fisher LDA solution: S = C⁻¹d, where C = (Cov_G + Cov_B)/2 is the average
within-class covariance and d = mean_B − mean_G.

When linear constraints are present (centering, pattern, bound constraints as
in score engineering), the problem is solved via quadratic programming:

    Minimize   S' C S
    Subject to:  d' S = δ
                 A_eq S = b_eq     (equality constraints)
                 A_ub S ≤ b_ub     (inequality constraints)
                 lb ≤ S ≤ ub       (bound constraints)

The solution T is then rescaled to the weight of evidence scale:
    W = β T,  where  β = d'T / (T' C T)

On this scale, the score variance equals the difference in means (σ² = μ_G - μ_B),
which is the standard FICO convention.

References:

Bruce Hoadley. A Quadratic Programming Solution to the FICO Credit Scoring
Problem. Fair, Isaac Technical Paper, July 14, 2000.

Bruce Hoadley. Score Engineered Logistic Regression. Fair, Isaac Technical
Paper, September 13, 2000.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class DivergenceClassifier(ClassifierMixin, BaseEstimator):
    """
    FICO Divergence-based binary classifier.

    Finds score weights S that maximize the FICO divergence:

        Div = (d'S)² / (S'CS)

    where d = mean(X|B) - mean(X|G) and C = (Cov(X|G) + Cov(X|B)) / 2.

    Parameters
    ----------
    delta : float, default=1.0
        Scale parameter for the linear constraint d'S = δ in the QP
        formulation. Controls the scale of the score weights before
        rescaling to weight of evidence.
    use_bias : bool, default=True
        Whether to include an intercept term. When True, the intercept
        is set so that the score at the overall mean of X equals the
        log population odds.
    significance : float, default=0.05
        Significance level for confidence intervals on the weights.
    A_eq : array-like of shape (n_eq, n_features), optional
        Left-hand side of equality constraints A_eq @ S = b_eq.
    b_eq : array-like of shape (n_eq,), optional
        Right-hand side of equality constraints.
    A_ub : array-like of shape (n_ub, n_features), optional
        Left-hand side of inequality constraints A_ub @ S <= b_ub.
    b_ub : array-like of shape (n_ub,), optional
        Right-hand side of inequality constraints.
    lb : array-like of shape (n_features,), optional
        Lower bounds on S. Defaults to no lower bounds.
    ub : array-like of shape (n_features,), optional
        Upper bounds on S. Defaults to no upper bounds.
    penalty : float, default=0.0
        Ridge penalty λ. When > 0, the QP objective becomes
        S'(C + λI)S, shrinking weights toward zero.
    woe_rescale : bool, default=True
        Whether to rescale the QP solution to weight of evidence scale
        (Theorem 1 in Hoadley 2000).
    verbose : bool, default=False
        Whether to print optimization progress.
    """

    def __init__(
        self,
        delta: float = 1.0,
        use_bias: bool = True,
        significance: float = 0.05,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        penalty: float = 0.0,
        woe_rescale: bool = True,
        verbose: bool = False,
    ) -> None:
        self.delta = delta
        self.use_bias = use_bias
        self.significance = significance
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.lb = lb
        self.ub = ub
        self.penalty = penalty
        self.woe_rescale = woe_rescale
        self.verbose = verbose

        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.beta_: Optional[np.ndarray] = None
        self.divergence_: Optional[float] = None
        self.C_: Optional[np.ndarray] = None
        self.d_: Optional[np.ndarray] = None
        self.mean_good_: Optional[np.ndarray] = None
        self.mean_bad_: Optional[np.ndarray] = None
        self.n_good_: int = 0
        self.n_bad_: int = 0
        self.is_fitted_: bool = False
        self.feature_names: Optional[List[str]] = None

    @staticmethod
    def _compute_moments(X: np.ndarray, y: np.ndarray) -> tuple:
        """Compute class means and average within-class covariance."""
        mask_good = y == 0
        mask_bad = y == 1

        X_good = X[mask_good]
        X_bad = X[mask_bad]

        mean_good = X_good.mean(axis=0)
        mean_bad = X_bad.mean(axis=0)

        cov_good = np.cov(X_good, rowvar=False, ddof=1)
        cov_bad = np.cov(X_bad, rowvar=False, ddof=1)

        # Average within-class covariance (FICO convention)
        C = (cov_good + cov_bad) / 2

        # Difference in means (Bad - Good direction, matches logistic regression)
        d = mean_bad - mean_good

        return C, d, mean_good, mean_bad, len(X_good), len(X_bad)

    @staticmethod
    def compute_divergence(scores: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the FICO divergence of a score vector.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Score values for each observation.
        y : array-like of shape (n_samples,)
            Binary labels (0 = Good, 1 = Bad).

        Returns
        -------
        float
            FICO divergence D = (μ_G - μ_B)² / ((σ²_G + σ²_B) / 2).
        """
        scores = np.asarray(scores)
        y = np.asarray(y)
        s_good = scores[y == 0]
        s_bad = scores[y == 1]
        mean_diff = s_good.mean() - s_bad.mean()
        avg_var = (s_good.var(ddof=1) + s_bad.var(ddof=1)) / 2
        return 0.0 if avg_var < 1e-15 else mean_diff**2 / avg_var

    def _solve_unconstrained(self, C: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Analytical Fisher LDA solution: S = C⁻¹d, scaled so d'S = δ."""
        C_reg = C + self.penalty * np.eye(C.shape[0]) if self.penalty > 0 else C
        try:
            S = np.linalg.solve(C_reg, d)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("WARNING: Singular covariance. Using pseudo-inverse.")
            S = np.linalg.pinv(C_reg) @ d

        # Scale so that d'S = δ
        scale = self.delta / (d @ S)
        S = S * scale
        return np.asarray(S)

    def _solve_constrained(
        self, C: np.ndarray, d: np.ndarray, n_features: int
    ) -> np.ndarray:
        """Solve the classic QP via scipy.optimize.minimize (SLSQP)."""
        C_obj = C + self.penalty * np.eye(n_features) if self.penalty > 0 else C

        def objective(S):
            return S @ C_obj @ S

        def gradient(S):
            return 2 * C_obj @ S

        # Mandatory constraint: d'S = δ
        constraints = [{"type": "eq", "fun": lambda S: d @ S - self.delta}]

        # User-supplied equality constraints: A_eq @ S = b_eq
        if self.A_eq is not None and self.b_eq is not None:
            A_eq = np.atleast_2d(self.A_eq)
            b_eq = np.atleast_1d(self.b_eq)
            for i in range(A_eq.shape[0]):
                row = A_eq[i]
                rhs = b_eq[i]
                constraints.append(
                    {"type": "eq", "fun": lambda S, r=row, b=rhs: r @ S - b}
                )

        # User-supplied inequality constraints: A_ub @ S <= b_ub
        if self.A_ub is not None and self.b_ub is not None:
            A_ub = np.atleast_2d(self.A_ub)
            b_ub = np.atleast_1d(self.b_ub)
            for i in range(A_ub.shape[0]):
                row = A_ub[i]
                rhs = b_ub[i]
                # SLSQP uses >= 0 convention, so we need rhs - row @ S >= 0
                constraints.append(
                    {"type": "ineq", "fun": lambda S, r=row, b=rhs: b - r @ S}
                )

        # Bounds
        if self.lb is not None or self.ub is not None:
            lb = self.lb if self.lb is not None else np.full(n_features, -np.inf)
            ub = self.ub if self.ub is not None else np.full(n_features, np.inf)
            bounds = list(zip(lb, ub))
        else:
            bounds = None

        # Initial guess: unconstrained solution (may not satisfy constraints)
        S0 = self._solve_unconstrained(C, d)

        result = minimize(
            objective,
            S0,
            jac=gradient,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": self.verbose},
        )

        if not result.success and self.verbose:
            print(f"WARNING: QP optimization did not converge: {result.message}")

        return np.asarray(result.x)

    def _has_constraints(self) -> bool:
        """Check whether any score engineering constraints are provided."""
        return any(
            x is not None
            for x in [self.A_eq, self.b_eq, self.A_ub, self.b_ub, self.lb, self.ub]
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> DivergenceClassifier:
        """
        Fit the divergence classifier.

        Finds score weights that maximize FICO divergence. In the
        unconstrained case, this is the Fisher LDA solution. With
        constraints, solves the classic quadratic program from
        Hoadley (2000).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Binary target (0 = Good, 1 = Bad).

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.classes_ = np.unique(y)

        # Compute score moments
        C, d, mean_good, mean_bad, n_good, n_bad = self._compute_moments(X, y)
        self.C_ = C
        self.d_ = d
        self.mean_good_ = mean_good
        self.mean_bad_ = mean_bad
        self.n_good_ = n_good
        self.n_bad_ = n_bad

        n_features = X.shape[1]

        # Solve for optimal weights
        if self._has_constraints():
            T = self._solve_constrained(C, d, n_features)
        else:
            T = self._solve_unconstrained(C, d)

        # Rescale to weight of evidence scale (Theorem 1, Hoadley 2000)
        if self.woe_rescale:
            TcT = T @ C @ T
            if TcT > 1e-15:
                beta_scale = (d @ T) / TcT
                W = beta_scale * T
            else:
                W = T
        else:
            W = T

        self.weights_ = W

        # Full beta vector (with optional bias)
        if self.use_bias:
            # Set intercept so that score at overall mean = log pop odds
            overall_mean = (n_good * mean_good + n_bad * mean_bad) / (n_good + n_bad)
            log_pop_odds = np.log(n_good / n_bad) if n_bad > 0 else 0.0
            self.bias_ = log_pop_odds - overall_mean @ W
            self.beta_ = np.concatenate([[self.bias_], W])
        else:
            self.bias_ = 0.0
            self.beta_ = W.copy()

        # Compute achieved divergence
        scores = X @ W
        self.divergence_ = self.compute_divergence(scores, y)

        self.is_fitted_ = True

        if self.verbose:
            print(f"FICO Divergence: {self.divergence_:.4f}")
            print(f"Fisher Ratio J: {self.divergence_ / 2:.4f}")
            d_S = d @ W
            S_C_S = W @ C @ W
            print(f"d'W = {d_S:.4f}, W'CW = {S_C_S:.4f}")

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw scores for each observation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The linear score S'X + bias. Higher scores indicate
            higher likelihood of being Bad (class 1).
        """
        if not self.is_fitted_:
            raise NotFittedError("This classifier is not fitted yet. Call 'fit' first.")
        X = np.asarray(X, dtype=np.float64)
        scores = X @ self.weights_
        if self.use_bias:
            scores = scores + self.bias_
        return np.asarray(scores)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the logistic function
        applied to the divergence score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0 = P(Good), Column 1 = P(Bad).
        """
        scores = self.score_samples(X)
        # Higher score → more likely Bad (class 1), matching logistic regression
        # P(Bad) = sigmoid(score), P(Good) = 1 - P(Bad)
        # Numerically stable sigmoid to avoid overflow in exp()
        p_bad = np.where(
            scores >= 0,
            1 / (1 + np.exp(-scores)),
            np.exp(scores) / (1 + np.exp(scores)),
        )
        p_bad = np.clip(p_bad, 1e-10, 1 - 1e-10)
        return np.column_stack((1 - p_bad, p_bad))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted labels (0 = Good, 1 = Bad).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "delta": self.delta,
            "use_bias": self.use_bias,
            "significance": self.significance,
            "A_eq": self.A_eq,
            "b_eq": self.b_eq,
            "A_ub": self.A_ub,
            "b_ub": self.b_ub,
            "lb": self.lb,
            "ub": self.ub,
            "penalty": self.penalty,
            "woe_rescale": self.woe_rescale,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> DivergenceClassifier:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def summary(self) -> Dict[str, np.ndarray]:
        """Get a summary of model weights."""
        assert self.beta_ is not None, "Model not fitted: beta_ is None"
        return {
            "betas": self.beta_,
        }

    def display_summary(self, style="default") -> None:
        """Display a summary table with divergence statistics."""
        console = Console()
        summary_dict = self.summary()

        table = Table(title="Divergence Classifier Summary")

        table.add_column("Parameter", justify="right", style=style, no_wrap=True)
        table.add_column("Weight", style=style)

        if self.feature_names:
            param_names = (
                ["intercept (bias)"] + self.feature_names
                if self.use_bias
                else self.feature_names
            )
        else:
            n_params = len(summary_dict["betas"])
            param_names = [f"Beta {i}" for i in range(n_params)]

        for i, param in enumerate(param_names):
            table.add_row(
                f"{param}",
                f"{summary_dict['betas'][i]:.4f}",
            )

        div = self.divergence_ if self.divergence_ is not None else 0.0
        summary_stats = f"""
        FICO Divergence: [{style}]{div:.4f}[/{style}]
        Fisher Ratio J: [{style}]{div / 2:.4f}[/{style}]
        N Good: [{style}]{self.n_good_}[/{style}]  N Bad: [{style}]{self.n_bad_}[/{style}]
        Weight of Evidence Rescaling: [{style}]{self.woe_rescale}[/{style}]
        Bias: [{style}]{self.use_bias}[/{style}]
        """

        console.print(
            Panel.fit(
                summary_stats,
                title="Divergence Classifier Fit",
                safe_box=True,
            )
        )
        console.print(table)
