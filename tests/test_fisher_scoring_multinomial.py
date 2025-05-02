"""test_fisher_scoring_multinomial.py."""

import unittest

import numpy as np
from fisher_scoring.fisher_scoring_multinomial import MultinomialLogisticRegression
from sklearn.exceptions import NotFittedError


class TestMultinomialLogisticRegression(unittest.TestCase):
    """Unit tests for the Fisher Scoring Multinomial Logistic Regression model."""

    def setUp(self):
        """Set up the test case."""
        self.model = MultinomialLogisticRegression()
        # Generate a synthetic dataset with 3 classes
        np.random.seed(0)
        self.X = np.random.rand(100, 5)
        self.y = np.random.randint(0, 3, 100)

    def test_fit_sets_is_fitted(self):
        """Test that the model is fitted after calling fit."""
        self.assertFalse(
            self.model.is_fitted_,
            "The model should not be fitted initially.",
        )
        self.model.fit(self.X, self.y)
        self.assertTrue(
            self.model.is_fitted_,
            "The model should be fitted after calling fit.",
        )

    def test_predict_raises_not_fitted_error(self):
        """Test that predict raises NotFittedError if the model is not fitted."""
        with self.assertRaises(NotFittedError):
            self.model.predict(self.X)

    def test_predict_proba_raises_not_fitted_error(self):
        """Test that predict_proba raises NotFittedError if the model is not fitted."""
        with self.assertRaises(NotFittedError):
            self.model.predict_proba(self.X)

    def test_fit_predict(self):
        """Test the fit and predict methods."""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(
            predictions.shape,
            self.y.shape,
            "Predictions shape should match the target shape.",
        )
        self.assertTrue(
            ((predictions >= 0) & (predictions < 3)).all(),
            "Predictions should be within the class range.",
        )

    def test_fit_predict_proba(self):
        """Test the fit and predict_proba methods."""
        self.model.fit(self.X, self.y)
        probabilities = self.model.predict_proba(self.X)
        self.assertEqual(
            probabilities.shape,
            (self.y.shape[0], 3),
            "Probabilities shape should match the number of samples and classes.",
        )
        self.assertTrue(
            (probabilities >= 0).all() and (probabilities <= 1).all(),
            "Probabilities should be between 0 and 1.",
        )

    def test_predict_ci(self):
        """Test the predict_ci method."""
        self.model.fit(self.X, self.y)
        # Test "logit" confidence intervals
        ci_logit = self.model.predict_ci(self.X, method="logit")
        self.assertEqual(
            len(ci_logit),
            3,
            "Logit confidence intervals should have entries for each class.",
        )
        # sourcery skip: no-loop-in-tests
        for class_idx, ci in ci_logit.items():
            self.assertEqual(
                ci.shape,
                (self.X.shape[0], 2),
                f"CI for class {class_idx} should have shape (n_samples, 2).",
            )
            self.assertTrue(
                (ci[:, 0] <= ci[:, 1]).all(),
                f"Lower CI should not exceed upper CI for class {class_idx}.",
            )

        # Test "proba" confidence intervals
        ci_proba = self.model.predict_ci(self.X, method="proba")
        self.assertEqual(
            len(ci_proba),
            3,
            "Probability confidence intervals should have entries for each class.",
        )
        # sourcery skip: no-loop-in-tests
        for class_idx, ci in ci_proba.items():
            self.assertEqual(
                ci.shape,
                (self.X.shape[0], 2),
                f"CI for class {class_idx} should have shape (n_samples, 2).",
            )
            self.assertTrue(
                (ci[:, 0] <= ci[:, 1]).all(),
                f"Lower CI should not exceed upper CI for class {class_idx}.",
            )
            self.assertTrue(
                (ci >= 0).all() and (ci <= 1).all(),
                f"Probability CIs for class {class_idx} should lie between 0 and 1.",
            )


if __name__ == "__main__":
    unittest.main()
