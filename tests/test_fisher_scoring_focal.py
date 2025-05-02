"""test_fisher_scoring_focal.py."""

import unittest

import numpy as np
from fisher_scoring.fisher_scoring_focal import FocalLossRegression
from fisher_scoring.fisher_scoring_logistic import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score


class TestFocalLossRegression(unittest.TestCase):
    """Test the FocalLossRegression class."""

    def setUp(self):
        """Set up the test case."""
        self.model = FocalLossRegression()
        # Generate a synthetic dataset with Bernoulli distribution
        np.random.seed(0)
        self.X = np.random.rand(100, 2)
        self.y = np.random.randint(0, 2, 100)

    def test_fit_sets_is_fitted(self):
        """Test that fit sets the is_fitted_ attribute."""
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
        """Test that fit and predict work correctly."""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(
            predictions.shape,
            self.y.shape,
            "Predictions shape should match the target shape.",
        )
        self.assertTrue(
            ((predictions == 0) | (predictions == 1)).all(),
            "Predictions should be binary.",
        )

    def test_fit_predict_proba(self):
        """Test that fit and predict_proba work correctly."""
        self.model.fit(self.X, self.y)
        probabilities = self.model.predict_proba(self.X)[:, 1]
        self.assertEqual(
            probabilities.shape,
            (self.y.shape[0],),
            "Probabilities shape should match the number of samples.",
        )
        self.assertTrue(
            (probabilities >= 0).all() and (probabilities <= 1).all(),
            "Probabilities should be between 0 and 1.",
        )

    def test_predict_ci(self):
        """Test that predict_ci works correctly."""
        self.model.fit(self.X, self.y)
        ci_logit = self._predict_ci(
            "logit",
            "CI for logits should have shape (n_samples, 2).",
            "Lower CI should not exceed upper CI for logits.",
        )
        ci_proba = self._predict_ci(
            "proba",
            "CI for probabilities should have shape (n_samples, 2).",
            "Lower CI should not exceed upper CI for probabilities.",
        )
        self.assertTrue(
            (ci_proba >= 0).all() and (ci_proba <= 1).all(),
            "Probability CIs should lie between 0 and 1.",
        )

    def _predict_ci(self, method, arg1, arg2):
        """Helper method to test predict_ci."""
        # Test the "logit" method
        result = self.model.predict_ci(self.X, method=method)
        self.assertEqual(result.shape, (self.X.shape[0], 2), arg1)
        self.assertTrue((result[:, 0] <= result[:, 1]).all(), arg2)

        return result

    def test_focal_vs_standard_logistic(self):
        """Test that FocalLossRegression and LogisticRegression give similar results."""
        model_standard = LogisticRegression()
        model_focal = FocalLossRegression(gamma=0.0)

        model_standard.fit(self.X, self.y)
        model_focal.fit(self.X, self.y)

        probas_standard = model_standard.predict_proba(self.X)[:, 1]
        probas_focal = model_focal.predict_proba(self.X)[:, 1]

        gini_standard = 2 * roc_auc_score(self.y, probas_standard) - 1
        gini_focal = 2 * roc_auc_score(self.y, probas_focal) - 1

        self.assertAlmostEqual(
            gini_standard,
            gini_focal,
            places=4,
            msg="Gini coefficients should be approximately equal for gamma=0.0",
        )


if __name__ == "__main__":
    unittest.main()
