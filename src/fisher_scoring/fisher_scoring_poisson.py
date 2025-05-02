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

import numpy as np


# pylint: disable=invalid-name
class PoissonRegression:
    """Poisson regression using Fisher scoring method."""

    def __init__(self, max_iter=100, epsilon=1e-5, use_bias=True):
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.weights = None

    def fit(self, x, y):
        """
        Fit the Poisson regression model using Fisher scoring.
        """
        # Add a column of ones to X for the intercept
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        self.weights = np.zeros(X.shape[1])  # Initialize weights (p+1,)

        for iteration in range(self.max_iter):
            # Linear predictor
            eta = X @ self.weights
            mu = np.exp(eta)  # Mean prediction (inverse link)

            # Poisson score vector
            score = X.T @ (y - mu)

            # Poisson information matrix
            W = np.diag(mu)  # Diagonal weight matrix
            information = X.T @ W @ X

            # Update weights using Newton-Raphson method
            delta = np.linalg.solve(information, score)
            beta_new = self.weights + delta

            # Check for convergence
            if np.linalg.norm(beta_new - self.weights) < self.epsilon:
                self.weights = beta_new
                print(f"Converged in {iteration + 1} iterations.")
                break

            self.weights = beta_new
        else:
            print("Did not converge within the maximum number of iterations.")

    def calculate_st_errors(self, x):
        """
        Calculate standard errors for the coefficients.
        """
        # Add a column of ones to X for the intercept
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        eta = X @ self.weights
        mu = np.exp(eta)
        W = np.diag(mu)
        information = X.T @ W @ X
        return np.sqrt(np.diag(np.linalg.inv(information)))

    def predict(self, x):
        """
        Predict mean values for the Poisson model.
        """
        # Predict mean values
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        eta = X @ self.weights
        return np.exp(eta)  # Return mean predictions (inverse link)


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
        """
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.weights = None
        self.alpha = alpha  # Fixed overdispersion parameter
        self.phi = phi  # Scale parameter
        self.offset = offset

    def fit(self, x, y):
        """
        Fit the Poisson regression model with dispersion-adjusted updates using Fisher scoring.
        """
        # Add intercept if necessary
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        if self.offset is None:
            self.offset = np.zeros(len(y))  # Default offset

        self.weights = np.zeros(X.shape[1])  # Initialize weights

        # Initialize mean (starting point for µ)
        mu = (y + np.mean(y)) / 2  # Non-binomial initialization
        eta = np.log(mu)  # Linear predictor

        for iteration in range(self.max_iter):
            # Variance: V(µ) = µ * (1 + alpha * µ)
            variance = mu * (1 + self.alpha * mu)

            # Link derivative: g'(µ) = 1 / µ (canonical log-link)
            g_prime = 1 / mu

            # Score function (gradient): z = η + (y - µ) / g'(µ)
            z = eta + (y - mu) * g_prime

            # Adjusted weights: W_e = p / (φ * V * g'^2)
            W_diag = 1 / (self.phi * variance * g_prime**2)
            W_e = np.diag(W_diag)  # Expected information matrix

            # Update coefficients: β = (X'WX)^(-1) X'Wz
            XtW = X.T @ W_e
            XtWX = XtW @ X
            XtWz = XtW @ z
            beta_new = np.linalg.solve(XtWX, XtWz)

            # Update linear predictor and mean
            eta = X @ beta_new + self.offset
            mu = np.exp(eta)

            # Convergence checks: |Δβ| and |ΔL|
            delta_beta = np.linalg.norm(beta_new - self.weights)
            if delta_beta < self.epsilon:
                self.weights = beta_new
                print(f"Converged in {iteration + 1} iterations.")
                break

            self.weights = beta_new

        else:
            print("Did not converge within the maximum number of iterations.")

    def calculate_st_errors(self, x):
        """
        Calculate standard errors for the coefficients.
        """
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        eta = X @ self.weights
        mu = np.exp(eta)
        variance = mu * (1 + self.alpha * mu)
        W_diag = 1 / (self.phi * variance * (1 / mu) ** 2)
        W = np.diag(W_diag)
        XtWX = X.T @ W @ X
        return np.sqrt(np.diag(np.linalg.inv(XtWX)))

    def predict(self, x):
        """
        Predict mean values for the Negative Binomial model.
        """
        X = np.hstack([np.ones((x.shape[0], 1)), x]) if self.use_bias else x
        eta = X @ self.weights
        return np.exp(eta)
