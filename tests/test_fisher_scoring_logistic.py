"""test_fisher_scoring_logistic.py."""

import unittest

import numpy as np
from fisher_scoring_logistic import LogisticRegression
from sklearn.exceptions import NotFittedError


class TestLogisticRegression(unittest.TestCase):
    """Unit tests for the Fisher Scoring Logistic Regression model."""

    def setUp(self):
        self.model = LogisticRegression()
        # generate a synthetic dataset with Bernoulli distribution
        np.random.seed(0)
        self.X = np.random.rand(100, 2)
        self.y = np.random.randint(0, 2, 100)

    def test_fit_sets_is_fitted(self):
        """Test that the is_fitted_ attribute is set correctly after fitting."""
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
        """Test the predict method."""
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
        """Test the predict_proba method."""
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
        """Test the predict_ci method for both logit and proba."""
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
        # Test the "logit" method
        result = self.model.predict_ci(self.X, method=method)
        self.assertEqual(result.shape, (self.X.shape[0], 2), arg1)
        self.assertTrue((result[:, 0] <= result[:, 1]).all(), arg2)

        return result

    def test_fit_difficult_convergence(self):
        # Generate an imbalanced dataset with a very small number of positive samples
        np.random.seed(0)
        self.X = np.random.rand(100, 2)
        self.y = np.zeros(100)
        self.y[:1] = 1  # Only 1 positive sample
        print("Imbalanced dataset with only 1 positive sample.")

        model = LogisticRegression(
            max_iter=10
        )  # Limit the number of iterations to observe convergence issue
        model.fit(self.X, self.y)

        # Check if the model is fitted
        self.assertTrue(
            model.is_fitted_,
            "The model should be fitted after calling fit.",
        )

        # Check if the model struggled to converge by examining loss history
        self.assertGreater(
            len(model.loss_history),
            1,
            "There should be multiple iterations of loss recorded.",
        )
        print("Loss history:", model.loss_history)
        self.assertTrue(
            any(
                abs(model.loss_history[i] - model.loss_history[i - 1]) > 1e-4
                for i in range(1, len(model.loss_history))
            ),
            "There should be noticeable changes in loss indicating convergence issues.",
        )

    def test_matrix_inversion_handling(self):
        """Test the model's handling of singular matrices during fitting."""
        # Arrange
        X_singular = np.array([[1, 1], [1, 1], [1, 1]])
        y_singular = np.array([0, 1, 0])

        # Act & Assert for information='expected'
        model_expected = LogisticRegression(information="expected")
        try:
            model_expected.fit(X_singular, y_singular)
        except np.linalg.LinAlgError:
            pass  # Expected exception for singular matrix
        except Exception as e:
            self.fail(
                f"An unexpected exception occurred while fitting the model with information='expected': {e}"
            )

        # Act & Assert for information='empirical'
        model_empirical = LogisticRegression(information="empirical")
        try:
            model_empirical.fit(X_singular, y_singular)
        except np.linalg.LinAlgError:
            pass  # Expected exception for singular matrix
        except Exception as e:
            self.fail(
                f"An unexpected exception occurred while fitting the model with information='empirical': {e}"
            )


if __name__ == "__main__":
    unittest.main()
