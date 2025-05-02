"""test_fisher_scoring_poisson.py."""

import unittest

import numpy as np
from fisher_scoring.fisher_scoring_poisson import (
    NegativeBinomialRegression,
    PoissonRegression,
)


class TestPoissonRegression(unittest.TestCase):
    """Unit tests for the Fisher Scoring Poisson Regression model."""

    def setUp(self):
        """Set up the test case."""
        self.model = PoissonRegression()

    def test_fit(self):
        """Test the fit method."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([1, 2, 3])
        self.model.fit(X, y)
        self.assertIsNotNone(self.model.weights)

    def test_predict(self):
        """Test the predict method."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([1, 2, 3])
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.assertIsNotNone(predictions)


class TestNegativeBinomialRegression(unittest.TestCase):
    """Unit tests for the Fisher Scoring Negative Binomial Regression model."""

    def setUp(self):
        """Set up the test case."""
        self.model = NegativeBinomialRegression()

    def test_fit(self):
        """Test the fit method."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([1, 2, 3])
        self.model.fit(X, y)
        self.assertIsNotNone(self.model.weights)

    def test_predict(self):
        """Test the predict method."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([1, 2, 3])
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.assertIsNotNone(predictions)


if __name__ == "__main__":
    unittest.main()
