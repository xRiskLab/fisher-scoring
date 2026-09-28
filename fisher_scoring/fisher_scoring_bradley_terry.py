"""fisher_scoring_bradley_terry.py.

Bradley-Terry Model
----------------------------------

Author: xRiskLab (deburky)
GitHub: https://github.com/xRiskLab
License: MIT

Description:
This module contains the `BradleyTerry` class, a paired-comparison model fitted by
maximum likelihood with the Fisher scoring algorithm, following the conventions of
the other estimators in this package (`LogisticRegression`, `PoissonRegression`).

The basic model says that for a comparison between item i and item j

    logit P(i beats j) = lambda_i - lambda_j

where lambda are latent "abilities" (log-worths). The class fits this core model and
three optional extensions, which can be combined freely:

1. Order effect / home advantage (`use_bias=True`)
       logit P(i beats j) = lambda_i - lambda_j + bias
   where `bias` is the advantage of the first-listed item of every comparison.

2. Comparison-level covariates (`X` in `fit`)
       logit P(i beats j) = lambda_i - lambda_j + x_ij' beta
   e.g. a rest-days difference, a surface indicator, or a DPO-style log-ratio.

3. Item-level covariates (`item_features` in `fit`)
       lambda_i = z_i' gamma
   Abilities are explained by item attributes. Free per-item abilities are then
   dropped, because they would be perfectly collinear with the item features.

Identification: the likelihood only depends on ability differences, so the abilities
are estimated under the sum-to-zero constraint. Internally this is done by adding a
rank-one term c * 11' to the ability block of the information matrix; because the
score is orthogonal to the constant vector, the Fisher scoring step is then exactly the
Moore-Penrose (constrained) step and the iterates remain mean-centered. Estimates can
be reported relative to a `reference` item instead.

Standard errors, Wald statistics, p-values and confidence intervals are obtained from
the inverse information matrix at the MLE, using either the expected or the empirical
Fisher information. Ties can be entered as y = 0.5 (half a win to each side).

References:
Ralph A. Bradley and Milton E. Terry. Rank Analysis of Incomplete Block Designs: I.
The Method of Paired Comparisons. Biometrika, 1952.

Heather Turner and David Firth. Bradley-Terry Models in R: The BradleyTerry2 Package.
Journal of Statistical Software, 2012.

David R. Hunter. MM Algorithms for Generalized Bradley-Terry Models. Annals of
Statistics, 2004.
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
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, list]


class BradleyTerry(BaseEstimator):
    """Fisher Scoring Bradley-Terry model for paired comparisons.

    Parameters
    ----------
    epsilon : float
        Convergence threshold on the parameter update norm.
    max_iter : int
        Maximum number of Fisher scoring iterations.
    information : {"expected", "empirical"}
        Information matrix used for the updates and the standard errors.
    use_bias : bool
        Add an order-effect / home-advantage intercept for the first-listed item.
    l2 : float
        Optional ridge penalty on all parameters. Keeps the MLE finite when a item
        is unbeaten or winless (complete separation). Standard errors are then the
        usual penalized-information approximation.
    reference : str, optional
        Item whose ability is fixed to zero for reporting. If None (default), the
        abilities are reported mean-centered, i.e. relative to the average item.
    significance : float
        Significance level for the confidence intervals.
    verbose : bool
        Print the log loss at each iteration.
    max_halvings : int
        Maximum number of step halvings per iteration when a full Fisher scoring
        step would lower the penalised log-likelihood.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        max_iter: int = 100,
        information: str = "expected",
        use_bias: bool = False,
        l2: float = 0.0,
        reference: Optional[str] = None,
        significance: float = 0.05,
        verbose: bool = False,
        max_halvings: int = 20,
    ) -> None:
        """Initialize the model; see the class docstring for parameters."""
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.information = information
        self.use_bias = use_bias
        self.l2 = l2
        self.reference = reference
        self.significance = significance
        self.verbose = verbose
        self.max_halvings = max_halvings

        # Fitted state
        self.beta: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.fisher_information_: Optional[pd.DataFrame] = None
        self._fit_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._fit_offset: Optional[np.ndarray] = None
        self.items_: List[str] = []
        self.feature_names: Optional[List[str]] = None
        self.item_feature_names: Optional[List[str]] = None
        self.param_names_: List[str] = []
        self.abilities_: Optional[pd.Series] = None
        self.abilities_se_: Optional[pd.Series] = None
        self.coef_: Optional[pd.Series] = None
        self.information_matrix: Dict[str, List[Union[int, np.ndarray]]] = {
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
        self.n_iter_: int = 0
        self.n_comparisons_: int = 0
        self.is_fitted_: bool = False
        self._item_features: Optional[pd.DataFrame] = None
        self._free_abilities: bool = True

    @staticmethod
    def logistic_function(z: np.ndarray) -> np.ndarray:
        """Numerically stable logistic function."""
        z = np.asarray(z, dtype=np.float64)
        p = np.where(
            z >= 0,
            1 / (1 + np.exp(-np.abs(z))),
            np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))),
        )
        return np.clip(p, 1e-10, 1 - 1e-10)

    @staticmethod
    def compute_loss(
        y: np.ndarray, p: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> float:
        """Weighted Bernoulli log-likelihood (higher is better)."""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = xlogy(y, p) + xlogy(1 - y, 1 - p)
        if weights is not None:
            ll = weights * ll
        return float(np.sum(ll))

    @staticmethod
    def compute_information_matrix(
        D: np.ndarray,
        p: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        information: str = "expected",
    ) -> np.ndarray:
        """Fisher information of the Bradley-Terry likelihood at probabilities `p`.

        With design rows d_ij = e_i - e_j (plus covariate columns), the expected
        information is

            I(theta) = sum_ij w_ij p_ij (1 - p_ij) d_ij d_ij'

        i.e. a weighted graph Laplacian over the comparison graph: each game adds
        p(1-p) to the diagonal of both items and subtracts it from their off-
        diagonal entry. Because the logit link is canonical, the observed
        information (negative Hessian) coincides with the expected one. The
        empirical information replaces p(1-p) by the squared residual (y - p)^2
        (the outer product of the per-observation scores).
        """
        if information == "expected":
            W_diag = weights * p * (1 - p)
        elif information == "empirical":
            W_diag = weights * (y - p) ** 2
        else:
            raise ValueError(
                f"Unknown Fisher Information type: {information}. "
                "Use 'expected' or 'empirical'."
            )
        return np.asarray((D.T * W_diag) @ D)

    @staticmethod
    def invert_matrix(matrix: np.ndarray, cond_threshold: float = 1e12) -> np.ndarray:
        """Invert a matrix, falling back to the pseudo-inverse if near-singular.

        Near-singularity is detected through the condition number.
        """
        if not np.all(np.isfinite(matrix)):
            cond = np.inf
        else:
            cond = np.linalg.cond(matrix)
        if cond > cond_threshold:
            warnings.warn(
                f"Near-singular information matrix (condition number: {cond:.2e}). "
                "Using pseudo-inverse. Results may be unreliable due to "
                "disconnected comparison graphs or complete separation.",
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

    def _check_items(
        self, item1: ArrayLike, item2: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        p1 = np.asarray(pd.Series(item1).astype(object)).reshape(-1)
        p2 = np.asarray(pd.Series(item2).astype(object)).reshape(-1)
        if p1.shape != p2.shape:
            raise ValueError("item1 and item2 must have the same length.")
        # Comparing an item with itself is allowed: the ability contrast is zero,
        # so such rows inform only the order effect and the covariates.
        return p1, p2

    def _check_X(
        self, X: Optional[ArrayLike], n: int, fitting: bool
    ) -> Optional[np.ndarray]:
        """Validate comparison-level covariates and record their names when fitting."""
        if X is None:
            if fitting:
                self.feature_names = None
            elif self.feature_names:
                raise ValueError(
                    "The model was fitted with covariates X; pass X to predict."
                )
            return None
        names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        X_arr = np.asarray(X, dtype=np.float64).reshape(n, -1)
        if fitting:
            self.feature_names = names or [f"x{i}" for i in range(X_arr.shape[1])]
        elif self.feature_names is None:
            raise ValueError("The model was fitted without covariates X.")
        elif X_arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X_arr.shape[1]} columns; the model was fitted with "
                f"{len(self.feature_names)}."
            )
        return X_arr

    def _build_design(
        self,
        item1: np.ndarray,
        item2: np.ndarray,
        X: Optional[np.ndarray],
    ) -> np.ndarray:
        """Stack [ability contrasts | item-feature contrasts | bias | X]."""
        n = len(item1)
        index = {name: k for k, name in enumerate(self.items_)}
        try:
            i = np.fromiter((index[p] for p in item1), dtype=int, count=n)
            j = np.fromiter((index[p] for p in item2), dtype=int, count=n)
        except KeyError as err:
            raise ValueError(f"Unknown item {err.args[0]!r}.") from None

        blocks: List[np.ndarray] = []
        if self._free_abilities:
            P = np.zeros((n, len(self.items_)))
            P[np.arange(n), i] = 1.0
            P[np.arange(n), j] -= 1.0
            blocks.append(P)
        if self._item_features is not None:
            Z = self._item_features.to_numpy(dtype=np.float64)
            blocks.append(Z[i] - Z[j])
        if self.use_bias:
            blocks.append(np.ones((n, 1)))
        if X is not None:
            blocks.append(X)
        return np.hstack(blocks)

    @property
    def _n_ability_params(self) -> int:
        return len(self.items_) if self._free_abilities else 0

    def fit(
        self,
        item1: ArrayLike,
        item2: ArrayLike,
        y: Optional[ArrayLike] = None,
        X: Optional[ArrayLike] = None,
        weights: Optional[ArrayLike] = None,
        item_features: Optional[pd.DataFrame] = None,
        offset: Optional[ArrayLike] = None,
    ) -> BradleyTerry:
        """Fit the Bradley-Terry model using Fisher scoring.

        Parameters
        ----------
        item1, item2 : array-like of shape (n,)
            Identifiers of the two items in each comparison. With `use_bias=True`,
            `item1` is the side that receives the order/home advantage.
        y : array-like of shape (n,), optional
            1 if item1 won, 0 if item2 won, 0.5 for a tie. Defaults to all ones,
            i.e. `item1` is the winner of every row.
        X : array-like of shape (n, k), optional
            Comparison-level covariates entering the logit additively.
        weights : array-like of shape (n,), optional
            Frequency weights (number of times the row was observed). Use these to fit
            aggregated win/loss counts without expanding the data.
        item_features : DataFrame indexed by item, optional
            Item-level covariates. Abilities become lambda_i = z_i' gamma and the
            free per-item abilities are dropped.
        offset : array-like of shape (n,), optional
            Known term added to each comparison's logit and not estimated, e.g. the
            logit difference of an existing model that the fit adjusts. With an
            `l2` penalty this gives a reference-anchored fit: the parameters are the
            smallest adjustment to the offset model that explains the comparisons.
        """
        p1, p2 = self._check_items(item1, item2)
        n = len(p1)
        self.n_comparisons_ = n

        y_arr = self._check_vector(y, n, "y", fill=1.0)
        if np.any((y_arr < 0) | (y_arr > 1)):
            raise ValueError("y must lie in [0, 1] (use 0.5 for ties).")
        w = self._check_vector(weights, n, "weights", fill=1.0)
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        off = self._check_vector(offset, n, "offset", fill=0.0)

        self._set_items(p1, p2, item_features)
        X_arr = self._check_X(X, n, fitting=True)
        D = self._build_design(p1, p2, X_arr)
        self.param_names_ = self._make_param_names()
        n_ab = self._n_ability_params

        self.information_matrix = {"iteration": [], "information": []}
        self.loss_history, self.beta_history = [], []
        self.beta = np.zeros(D.shape[1])

        for iteration in range(self.max_iter):
            p = self.logistic_function(D @ self.beta + off)
            score = D.T @ (w * (y_arr - p)) - self.l2 * self.beta
            information_matrix = self.compute_information_matrix(
                D, p, y_arr, w, self.information
            ) + self.l2 * np.eye(D.shape[1])
            if self.information == "expected":
                self._warn_if_separated(p, iteration)

            self.information_matrix["iteration"].append(iteration)
            self.information_matrix["information"].append(information_matrix)

            loss = self._penalized_loss(D, y_arr, w, off, self.beta)
            self.loss_history.append(loss)
            if self.verbose:
                if iteration == 0:
                    print("Starting Fisher Scoring Iterations...")
                print(
                    f"Iteration: {iteration + 1}, "
                    f"Log Loss: {-loss / max(w.sum(), 1):.4f}"
                )

            step = self.invert_matrix(self._augment(information_matrix, n_ab)) @ score
            beta_new = self._damped_update(D, y_arr, w, off, step, loss)

            if np.linalg.norm(beta_new - self.beta) < self.epsilon:
                if self.verbose:
                    print(f"Convergence reached after {iteration + 1} iterations.")
                self.beta = beta_new
                self.n_iter_ = iteration + 1
                break

            self.beta = beta_new
            self.beta_history.append(self.beta.copy())
            self.n_iter_ = iteration + 1
            if iteration == self.max_iter - 1:
                warnings.warn(
                    "Maximum iterations reached without convergence.", stacklevel=2
                )

        self._fit_data = (D, y_arr, w)
        self._fit_offset = off
        self.compute_statistics()
        self.fisher_information_ = self.fisher_information()
        self.is_fitted_ = True
        return self

    @staticmethod
    def _check_vector(
        values: Optional[ArrayLike], n: int, name: str, fill: float
    ) -> np.ndarray:
        """Return `values` as a float vector of length n, or a constant if None."""
        if values is None:
            return np.full(n, fill)
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"{name} must have the same length as item1/item2.")
        return arr

    def _set_items(
        self,
        item1: np.ndarray,
        item2: np.ndarray,
        item_features: Optional[pd.DataFrame],
    ) -> None:
        """Record the item set and whether abilities are free or feature-driven."""
        if item_features is None:
            self._item_features = None
            self.item_feature_names = None
            self.items_ = sorted(set(item1) | set(item2), key=str)
            self._free_abilities = True
        else:
            if not isinstance(item_features, pd.DataFrame):
                raise TypeError(
                    "item_features must be a pandas DataFrame indexed by item."
                )
            self._item_features = item_features.astype(np.float64)
            self.item_feature_names = item_features.columns.tolist()
            self.items_ = item_features.index.tolist()
            self._free_abilities = False
            missing = (set(item1) | set(item2)) - set(self.items_)
            if missing:
                raise ValueError(
                    f"Items missing from item_features: {sorted(map(str, missing))}"
                )
        if self.reference is not None and self.reference not in self.items_:
            raise ValueError(
                f"reference item {self.reference!r} not found in the data."
            )

    def _penalized_loss(
        self,
        D: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        offset: np.ndarray,
        beta: np.ndarray,
    ) -> float:
        """Weighted log-likelihood minus the ridge penalty at `beta`."""
        p = self.logistic_function(D @ beta + offset)
        return self.compute_loss(y, p, w) - 0.5 * self.l2 * float(beta @ beta)

    def _damped_update(
        self,
        D: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        offset: np.ndarray,
        step: np.ndarray,
        loss: float,
    ) -> np.ndarray:
        """Take the Fisher scoring step, halving it while the likelihood decreases.

        Fisher scoring can overshoot when many comparisons sit near p = 0 or 1
        (e.g. a large offset). Halve the step until the penalised log-likelihood
        does not decrease, at most `max_halvings` times.
        """
        assert self.beta is not None
        beta_new = self.beta + step
        target = loss - 1e-12 * abs(loss)
        for _ in range(self.max_halvings):
            if self._penalized_loss(D, y, w, offset, beta_new) >= target:
                break
            step = step / 2
            beta_new = self.beta + step
        return np.asarray(beta_new)

    @staticmethod
    def _warn_if_separated(p: np.ndarray, iteration: int) -> None:
        """Warn when most fitted probabilities are numerically 0 or 1."""
        near_zero_frac = float(np.mean((p * (1 - p)) < 1e-6))
        if near_zero_frac > 0.9:
            warnings.warn(
                "Possible complete or quasi-complete separation detected: "
                f"{near_zero_frac:.0%} of comparisons have p*(1-p) < 1e-6 "
                f"at iteration {iteration + 1}. The MLE may not exist; "
                "consider a small l2 penalty.",
                stacklevel=3,
            )

    @staticmethod
    def _augment(information_matrix: np.ndarray, n_ab: int) -> np.ndarray:
        """Add c * 11' to the ability block to make the information invertible.

        The information matrix is rank-deficient in the ability block. The
        constant vector is an eigenvector of the result and the score is
        orthogonal to it, so the Fisher step equals the pseudo-inverse step and
        the abilities stay mean-centered.
        """
        if n_ab == 0:
            return information_matrix
        aug = information_matrix.copy()
        c = np.trace(information_matrix[:n_ab, :n_ab]) / n_ab
        if not np.isfinite(c) or c <= 0:
            c = 1.0
        aug[:n_ab, :n_ab] += c
        return aug

    def _reporting_transform(self) -> np.ndarray:
        """Linear map from internal (sum-to-zero) parameters to reported ones."""
        assert self.beta is not None
        k = len(self.beta)
        n_ab = self._n_ability_params
        T = np.eye(k)
        if n_ab == 0:
            return T
        if self.reference is None:
            T[:n_ab, :n_ab] -= 1.0 / n_ab
        else:
            r = self.items_.index(self.reference)
            T[:n_ab, r] -= 1.0
        return T

    def compute_statistics(self) -> None:
        """Compute standard errors, Wald statistics, p-values and CIs."""
        assert self.beta is not None
        info = self.information_matrix["information"][-1]
        assert isinstance(info, np.ndarray)
        n_ab = self._n_ability_params
        cov_internal = self.invert_matrix(self._augment(info, n_ab))

        T = self._reporting_transform()
        self.beta = T @ self.beta
        self.covariance_ = T @ cov_internal @ T.T

        variances = np.clip(np.diagonal(self.covariance_), 0.0, None)
        self.standard_errors = np.sqrt(variances)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.wald_statistic = np.where(
                self.standard_errors > 0, self.beta / self.standard_errors, np.nan
            )
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.wald_statistic)))
        critical_value = norm.ppf(1 - self.significance / 2)
        self.lower_bound = self.beta - critical_value * self.standard_errors
        self.upper_bound = self.beta + critical_value * self.standard_errors

        # Convenience views
        if n_ab:
            self.abilities_ = pd.Series(
                self.beta[:n_ab], index=self.items_, name="ability"
            )
            self.abilities_se_ = pd.Series(
                self.standard_errors[:n_ab], index=self.items_, name="se"
            )
        else:
            assert self._item_features is not None
            Z = self._item_features.to_numpy(dtype=np.float64)
            m = Z.shape[1]
            gamma = self.beta[:m]
            cov_g = self.covariance_[:m, :m]
            lam = Z @ gamma
            lam_se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", Z, cov_g, Z), 0.0, None))
            self.abilities_ = pd.Series(lam, index=self.items_, name="ability")
            self.abilities_se_ = pd.Series(lam_se, index=self.items_, name="se")
        self.coef_ = pd.Series(
            self.beta[n_ab:], index=self.param_names_[n_ab:], name="coef"
        )

    def _make_param_names(self) -> List[str]:
        names: List[str] = []
        if self._free_abilities:
            names += [str(p) for p in self.items_]
        if self.item_feature_names:
            names += [str(c) for c in self.item_feature_names]
        if self.use_bias:
            names.append("bias (order effect)")
        if self.feature_names:
            names += [str(c) for c in self.feature_names]
        return names

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise NotFittedError(
                "This BradleyTerry instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def _linear_predictor(
        self,
        item1: ArrayLike,
        item2: ArrayLike,
        X: Optional[ArrayLike],
        offset: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        p1, p2 = self._check_items(item1, item2)
        X_arr = self._check_X(X, len(p1), fitting=False)
        D = self._build_design(p1, p2, X_arr)
        off = (
            0.0 if offset is None else np.asarray(offset, dtype=np.float64).reshape(-1)
        )
        # Reported parameters differ from the internal ones by a constant shift of the
        # ability block, which cancels in every contrast, so D @ beta is unchanged.
        return D, D @ self.beta + off

    def predict_proba(
        self,
        item1: ArrayLike,
        item2: ArrayLike,
        X: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Predict [P(item2 wins), P(item1 wins)] for each comparison.

        Follows the sklearn layout: column 1 is the probability of y = 1, i.e.
        that item1 wins.
        """
        _, eta = self._linear_predictor(item1, item2, X, offset)
        proba_1 = self.logistic_function(eta)
        return np.column_stack((1 - proba_1, proba_1))

    def predict(
        self,
        item1: ArrayLike,
        item2: ArrayLike,
        X: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Predict 1 if item1 is favoured to beat item2, else 0."""
        return (self.predict_proba(item1, item2, X, offset)[:, 1] > 0.5).astype(int)

    def predict_ci(
        self,
        item1: ArrayLike,
        item2: ArrayLike,
        X: Optional[ArrayLike] = None,
        method: str = "logit",
        offset: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Confidence intervals for P(item1 wins).

        method="logit" builds the interval on the logit scale and maps it back;
        method="proba" uses the delta method directly on the probability.
        """
        D, eta = self._linear_predictor(item1, item2, X, offset)
        assert self.covariance_ is not None
        proba = self.logistic_function(eta)
        z_crit = norm.ppf(1 - self.significance / 2)
        # The reported covariance is expressed for the reported parameters; contrasts
        # D are invariant to the identification shift, so this is the right form.
        var_eta = np.einsum("ij,jk,ik->i", D, self.covariance_, D)
        std_errors = np.sqrt(np.clip(var_eta, 0.0, None))
        if method == "logit":
            lower = self.logistic_function(eta - z_crit * std_errors)
            upper = self.logistic_function(eta + z_crit * std_errors)
        elif method == "proba":
            se_p = proba * (1 - proba) * std_errors
            lower = np.clip(proba - z_crit * se_p, 0, 1)
            upper = np.clip(proba + z_crit * se_p, 0, 1)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'logit' or 'proba'.")
        return np.vstack((lower, upper)).T

    def fisher_information(
        self, information: Optional[str] = None, penalized: bool = False
    ) -> pd.DataFrame:
        """Fisher information matrix at the fitted parameters, labelled by parameter.

        Parameters
        ----------
        information : {"expected", "empirical"}, optional
            Defaults to the type used for fitting.
        penalized : bool
            Add the ridge term `l2 * I` (the curvature actually used for the
            standard errors when `l2 > 0`).

        Notes
        -----
        The ability block is returned in the full (one column per item)
        parameterization, so it is the weighted graph Laplacian of the comparison
        graph: rows and columns sum to zero and the matrix has rank n_items - 1.
        Its Moore-Penrose pseudo-inverse is the covariance of the mean-centered
        abilities; dropping a reference item's row and column and inverting
        gives the covariance relative to that item. The linear predictor
        d_ij' beta is invariant to that shift, so the same matrix serves either
        identification.
        """
        if self._fit_data is None or self.beta is None:
            raise NotFittedError(
                "This BradleyTerry instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        D, y, w = self._fit_data
        p = self.logistic_function(D @ self.beta + self._fit_offset)
        info = self.compute_information_matrix(
            D, p, y, w, information or self.information
        )
        if penalized:
            info = info + self.l2 * np.eye(D.shape[1])
        return pd.DataFrame(info, index=self.param_names_, columns=self.param_names_)

    def rank(self) -> pd.DataFrame:
        """Items sorted by estimated ability, with standard errors."""
        self._check_is_fitted()
        assert self.abilities_ is not None and self.abilities_se_ is not None
        out = pd.DataFrame({"ability": self.abilities_, "se": self.abilities_se_})
        return pd.DataFrame(out.sort_values("ability", ascending=False))

    def get_params(
        self, deep: bool = True
    ) -> Dict[str, Union[float, int, str, bool, None]]:
        """Get the model parameters."""
        return {
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
            "information": self.information,
            "use_bias": self.use_bias,
            "l2": self.l2,
            "reference": self.reference,
            "significance": self.significance,
            "verbose": self.verbose,
            "max_halvings": self.max_halvings,
        }

    def set_params(self, **params: Union[float, int, str, bool, None]) -> BradleyTerry:
        """Set the model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def summary(self) -> Dict[str, np.ndarray]:
        """Parameter estimates, standard errors, Wald statistics, p-values and CIs."""
        self._check_is_fitted()
        assert self.beta is not None and self.standard_errors is not None
        assert self.wald_statistic is not None and self.p_values is not None
        assert self.lower_bound is not None and self.upper_bound is not None
        return {
            "parameters": np.asarray(self.param_names_, dtype=object),
            "betas": self.beta,
            "standard_errors": self.standard_errors,
            "wald_statistic": self.wald_statistic,
            "p_values": self.p_values,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    def _is_reference(self, param: str) -> bool:
        """True for the item whose ability is fixed at zero for identification."""
        return self.reference is not None and param == str(self.reference)

    def summary_frame(self) -> pd.DataFrame:
        """The summary as a DataFrame indexed by parameter name.

        The reference item (if any) has its ability fixed at zero, so it has no
        standard error, Wald statistic or p-value; those are NaN and the row is
        flagged in the `reference` column.
        """
        s = self.summary()
        frame = pd.DataFrame(
            {
                "estimate": s["betas"],
                "std_error": s["standard_errors"],
                "wald": s["wald_statistic"],
                "p_value": s["p_values"],
                "lower_ci": s["lower_bound"],
                "upper_ci": s["upper_bound"],
            },
            index=pd.Index(self.param_names_, name="parameter"),
        )
        frame["reference"] = [self._is_reference(n) for n in self.param_names_]
        return frame

    def display_summary(self, style: str = "default") -> None:
        """Display a rich summary table in the console or a notebook."""
        console = Console()
        summary_dict = self.summary()

        table = Table(title="Fisher Scoring Bradley-Terry Summary")
        table.add_column("Parameter", justify="right", style=style, no_wrap=True)
        table.add_column("Estimate", style=style)
        table.add_column("Std. Error", style=style)
        table.add_column("Wald Statistic", style=style)
        table.add_column("P-value", style=style)
        table.add_column("Lower CI", style=style)
        table.add_column("Upper CI", style=style)

        n_ab = self._n_ability_params
        for i, param in enumerate(self.param_names_):
            if i == n_ab and n_ab:
                table.add_section()
            if self._is_reference(param):
                table.add_row(f"{param}", "0 (reference)", "-", "-", "-", "-", "-")
                continue
            table.add_row(
                f"{param}",
                f"{summary_dict['betas'][i]:.4f}",
                f"{summary_dict['standard_errors'][i]:.4f}",
                f"{summary_dict['wald_statistic'][i]:.4f}",
                f"{summary_dict['p_values'][i]:.4f}",
                f"{summary_dict['lower_bound'][i]:.4f}",
                f"{summary_dict['upper_bound'][i]:.4f}",
            )

        identification = (
            f"reference = {self.reference}"
            if self.reference is not None
            else "sum-to-zero (vs. average item)"
        )
        if not self._free_abilities:
            identification = "abilities = item_features @ gamma"
        n_items = len(self.items_)
        summary_stats = f"""Total Fisher Scoring Iterations: [{style}]{self.n_iter_}[/{style}]
        Log Likelihood: [{style}]{self.loss_history[-1]:.4f}[/{style}]
        Comparisons / Items: [{style}]{self.n_comparisons_} / {n_items}[/{style}]
        Identification: [{style}]{identification}[/{style}]
        Order effect (bias): [{style}]{self.use_bias}[/{style}]
        Information: [{style}]{self.information}[/{style}]
        """
        console.print(
            Panel.fit(
                summary_stats, title="Fisher Scoring Bradley-Terry Fit", safe_box=True
            )
        )
        console.print(table)


def pairs_from_counts(
    df: pd.DataFrame,
    item1: str,
    item2: str,
    wins1: str,
    wins2: str,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Turn an aggregated win-count table into weighted long format for `BradleyTerry.fit`.

    Each input row becomes two rows: (item1, item2, y=1, weight=wins1) and
    (item1, item2, y=0, weight=wins2). `item1` keeps its role in both, so
    `use_bias=True` estimates a `item1` (e.g. home) advantage.
    """
    rows_won = df[[item1, item2]].assign(y=1.0, weight=df[wins1].astype(float))
    rows_lost = df[[item1, item2]].assign(y=0.0, weight=df[wins2].astype(float))
    long = pd.concat([rows_won, rows_lost], ignore_index=True)
    long = long[long["weight"] > 0].reset_index(drop=True)
    return long[item1], long[item2], long["y"], long["weight"]
