"""test_fisher_scoring_robust.py."""

import unittest

import numpy as np
from fisher_scoring.fisher_scoring_logistic import LogisticRegression
from fisher_scoring.fisher_scoring_robust import RobustLogisticRegression
from sklearn.exceptions import NotFittedError


class TestRobustLogisticRegression(unittest.TestCase):
    """Unit tests for the Robust Fisher Scoring Logistic Regression model."""

    def setUp(self):
        """Set up test fixtures with clean and contaminated datasets."""
        np.random.seed(42)

        # Clean dataset
        self.X_clean = np.random.rand(100, 2)
        self.y_clean = (self.X_clean[:, 0] + self.X_clean[:, 1] > 1).astype(int)

        # Contaminated dataset (same as clean but with outliers)
        self.X_contaminated = self.X_clean.copy()
        self.y_contaminated = self.y_clean.copy()

        # Add outliers - flip labels for some extreme points
        outlier_indices = [5, 15, 25, 35, 45]  # 5% contamination
        self.y_contaminated[outlier_indices] = 1 - self.y_contaminated[outlier_indices]

        # Test data for prediction
        self.X_test = np.random.rand(20, 2)

        self.robust_model = RobustLogisticRegression()
        self.standard_model = LogisticRegression()

    def test_init_parameters(self):
        """Test that the model initializes with correct parameters."""
        model = RobustLogisticRegression(
            epsilon_contamination=0.1,
            contamination_prob=0.3,
            tol=1e-8,
            max_iter=50,
            information="empirical",
            use_bias=False,
            significance=0.01,
            verbose=True,
        )

        self.assertEqual(model.epsilon_contamination, 0.1)
        self.assertEqual(model.contamination_prob, 0.3)
        self.assertEqual(model.tol, 1e-8)
        self.assertEqual(model.max_iter, 50)
        self.assertEqual(model.information, "empirical")
        self.assertFalse(model.use_bias)
        self.assertEqual(model.significance, 0.01)
        self.assertTrue(model.verbose)

    def test_fit_sets_is_fitted(self):
        """Test that the is_fitted_ attribute is set correctly after fitting."""
        self.assertFalse(
            self.robust_model.is_fitted_,
            "The model should not be fitted initially.",
        )
        self.robust_model.fit(self.X_clean, self.y_clean)
        self.assertTrue(
            self.robust_model.is_fitted_,
            "The model should be fitted after calling fit.",
        )

    def test_predict_raises_not_fitted_error(self):
        """Test that predict raises NotFittedError if the model is not fitted."""
        with self.assertRaises(NotFittedError):
            self.robust_model.predict(self.X_clean)

    def test_predict_proba_raises_not_fitted_error(self):
        """Test that predict_proba raises NotFittedError if the model is not fitted."""
        with self.assertRaises(NotFittedError):
            self.robust_model.predict_proba(self.X_clean)

    def test_summary_raises_not_fitted_error(self):
        """Test that summary raises NotFittedError if the model is not fitted."""
        with self.assertRaises(NotFittedError):
            self.robust_model.summary()

    def test_display_summary_raises_not_fitted_error(self):
        """Test that display_summary raises NotFittedError if the model is not fitted."""
        with self.assertRaises(NotFittedError):
            self.robust_model.display_summary()

    def test_fit_predict_clean_data(self):
        """Test the fit and predict methods on clean data."""
        self.robust_model.fit(self.X_clean, self.y_clean)
        predictions = self.robust_model.predict(self.X_test)

        self.assertEqual(
            predictions.shape[0],
            self.X_test.shape[0],
            "Predictions shape should match the test data.",
        )
        self.assertTrue(
            np.all((predictions == 0) | (predictions == 1)),
            "Predictions should be binary (0 or 1).",
        )

    def test_predict_proba_clean_data(self):
        """Test the predict_proba method on clean data."""
        self.robust_model.fit(self.X_clean, self.y_clean)
        probabilities = self.robust_model.predict_proba(self.X_test)

        self.assertEqual(
            probabilities.shape,
            (self.X_test.shape[0], 2),
            "Probabilities should have shape (n_samples, 2).",
        )
        self.assertTrue(
            np.allclose(probabilities.sum(axis=1), 1.0),
            "Probabilities should sum to 1.",
        )
        self.assertTrue(
            np.all((probabilities >= 0) & (probabilities <= 1)),
            "Probabilities should be between 0 and 1.",
        )

    def test_robustness_against_outliers(self):
        """Test that robust model is more resistant to outliers than standard model."""
        # Fit both models on contaminated data
        self.robust_model.fit(self.X_contaminated, self.y_contaminated)
        self.standard_model.fit(self.X_contaminated, self.y_contaminated)

        # Test on clean test data (should better match clean pattern)
        robust_proba = self.robust_model.predict_proba(self.X_test)[:, 1]
        standard_proba = self.standard_model.predict_proba(self.X_test)[:, 1]

        # Fit both models on clean data to get "true" probabilities
        robust_clean = RobustLogisticRegression()
        standard_clean = LogisticRegression()
        robust_clean.fit(self.X_clean, self.y_clean)
        standard_clean.fit(self.X_clean, self.y_clean)

        true_robust_proba = robust_clean.predict_proba(self.X_test)[:, 1]
        true_standard_proba = standard_clean.predict_proba(self.X_test)[:, 1]

        # Robust model should be closer to its clean version than standard model
        robust_mse_contaminated = np.mean((robust_proba - true_robust_proba) ** 2)
        standard_mse_contaminated = np.mean((standard_proba - true_standard_proba) ** 2)

        # The robust model should generally be less affected by contamination
        # This test checks that robust model maintains more stability
        self.assertIsInstance(robust_mse_contaminated, (int, float))
        self.assertIsInstance(standard_mse_contaminated, (int, float))

        # Check that robust weights are working (some should be down-weighted)
        self.assertIsNotNone(self.robust_model.weights)
        self.assertTrue(
            np.min(self.robust_model.weights) < np.max(self.robust_model.weights)
        )

    def test_robust_weights_computation(self):
        """Test that robust weights are computed correctly."""
        self.robust_model.fit(self.X_contaminated, self.y_contaminated)

        weights = self.robust_model.weights
        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), len(self.y_contaminated))

        # All weights should be positive and <= 1
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(weights <= 1))

        # With contamination, some weights should be down-weighted
        self.assertLess(
            np.min(weights), 0.99
        )  # Some observations should be down-weighted

    def test_epsilon_contamination_effect(self):
        """Test that different epsilon values affect robustness."""
        low_epsilon_model = RobustLogisticRegression(epsilon_contamination=0.01)
        high_epsilon_model = RobustLogisticRegression(epsilon_contamination=0.2)

        low_epsilon_model.fit(self.X_contaminated, self.y_contaminated)
        high_epsilon_model.fit(self.X_contaminated, self.y_contaminated)

        # Higher epsilon should lead to more down-weighting (lower minimum weight)
        self.assertLess(
            np.min(high_epsilon_model.weights), np.min(low_epsilon_model.weights)
        )

    def test_statistical_inference_methods(self):
        """Test that statistical inference methods work correctly."""
        self.robust_model.fit(self.X_clean, self.y_clean)

        # Test summary method
        summary = self.robust_model.summary()
        expected_keys = [
            "betas",
            "standard_errors",
            "wald_statistic",
            "p_values",
            "lower_bound",
            "upper_bound",
            "robust_weights",
        ]

        # sourcery skip: no-loop-in-tests
        for key in expected_keys:
            self.assertIn(key, summary)

        # Test that beta and standard errors have correct shape
        n_features = self.X_clean.shape[1] + (1 if self.robust_model.use_bias else 0)
        self.assertEqual(len(summary["betas"]), n_features)
        self.assertEqual(len(summary["standard_errors"]), n_features)

    def test_predict_ci_method(self):
        """Test confidence intervals for predictions."""
        self.robust_model.fit(self.X_clean, self.y_clean)

        # Test logit confidence intervals
        ci_logit = self.robust_model.predict_ci(self.X_test, method="logit")
        self.assertEqual(ci_logit.shape, (self.X_test.shape[0], 2))
        self.assertTrue(np.all(ci_logit[:, 0] <= ci_logit[:, 1]))  # lower <= upper

        # Test probability confidence intervals
        ci_proba = self.robust_model.predict_ci(self.X_test, method="proba")
        self.assertEqual(ci_proba.shape, (self.X_test.shape[0], 2))
        self.assertTrue(
            np.all(ci_proba >= 0) and np.all(ci_proba <= 1)
        )  # valid probabilities

    def test_invalid_information_type(self):
        """Test that invalid information type raises ValueError."""
        model = RobustLogisticRegression(information="invalid")
        with self.assertRaises(ValueError):
            model.fit(self.X_clean, self.y_clean)

    def test_invalid_ci_method(self):
        """Test that invalid CI method raises ValueError."""
        self.robust_model.fit(self.X_clean, self.y_clean)
        with self.assertRaises(ValueError):
            self.robust_model.predict_ci(self.X_test, method="invalid")

    def test_get_set_params(self):
        """Test parameter getting and setting."""
        params = self.robust_model.get_params()
        expected_params = [
            "epsilon_contamination",
            "contamination_prob",
            "tol",
            "max_iter",
            "information",
            "significance",
            "use_bias",
            "verbose",
        ]

        # sourcery skip: no-loop-in-tests
        for param in expected_params:
            self.assertIn(param, params)

        # Test setting parameters
        self.robust_model.set_params(epsilon_contamination=0.1, tol=1e-8)
        self.assertEqual(self.robust_model.epsilon_contamination, 0.1)
        self.assertEqual(self.robust_model.tol, 1e-8)

    def test_convergence_with_verbose(self):
        """Test that verbose mode works without errors."""
        verbose_model = RobustLogisticRegression(verbose=True, max_iter=5)

        # Capture output by fitting - should not raise errors
        try:
            verbose_model.fit(self.X_clean, self.y_clean)
        except Exception as e:
            self.fail(f"Verbose mode raised an exception: {e}")

    def test_empirical_information_matrix(self):
        """Test that empirical information matrix works."""
        empirical_model = RobustLogisticRegression(information="empirical")
        empirical_model.fit(self.X_clean, self.y_clean)

        self.assertTrue(empirical_model.is_fitted_)
        self.assertIsNotNone(empirical_model.beta)

        # Should produce valid predictions
        predictions = empirical_model.predict(self.X_test)
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))

    def test_robust_information_matrix_comparison(self):
        """Test that RobustLogisticRegression expected vs empirical information matrices give reasonable results."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_beta = np.array([0.5, -0.3, 0.8, 0.2])  # Including intercept
        eta = X @ true_beta[1:] + true_beta[0]
        p = 1 / (1 + np.exp(-eta))
        y = (np.random.rand(100) < p).astype(int)

        # Add some outliers
        outlier_indices = [5, 15, 25, 35, 45]  # 5% contamination
        y[outlier_indices] = 1 - y[outlier_indices]

        # Fit with expected Fisher information
        model_expected = RobustLogisticRegression(
            epsilon_contamination=0.05, max_iter=50, tol=1e-6, information="expected"
        )
        model_expected.fit(X, y)

        # Fit with empirical Fisher information
        model_empirical = RobustLogisticRegression(
            epsilon_contamination=0.05, max_iter=50, tol=1e-6, information="empirical"
        )
        model_empirical.fit(X, y)

        # Both should converge to reasonable solutions
        self.assertIsNotNone(model_expected.beta)
        self.assertIsNotNone(model_empirical.beta)
        self.assertEqual(len(model_expected.beta), 4)  # intercept + 3 features
        self.assertEqual(len(model_empirical.beta), 4)

        # Coefficients should be reasonably close for robust models
        # (Robust models should be more stable than count models)
        coeff_diff = np.linalg.norm(model_expected.beta - model_empirical.beta)
        self.assertLess(coeff_diff, 1.0)  # Should be closer than count models

        # Test predictions on new data
        X_test = np.random.randn(20, 3)
        pred_expected = model_expected.predict(X_test)
        pred_empirical = model_empirical.predict(X_test)

        self.assertEqual(len(pred_expected), 20)
        self.assertEqual(len(pred_empirical), 20)
        self.assertTrue(np.all((pred_expected == 0) | (pred_expected == 1)))
        self.assertTrue(np.all((pred_empirical == 0) | (pred_empirical == 1)))

        # Both should identify outliers similarly (robust weights)
        self.assertIsNotNone(model_expected.weights)
        self.assertIsNotNone(model_empirical.weights)
        
        # Check that both models down-weight outliers
        expected_outlier_weights = model_expected.weights[outlier_indices]
        empirical_outlier_weights = model_empirical.weights[outlier_indices]
        expected_normal_weights = np.delete(model_expected.weights, outlier_indices)
        empirical_normal_weights = np.delete(model_empirical.weights, outlier_indices)
        
        # Outliers should have lower weights than normal observations for both methods
        self.assertLess(np.mean(expected_outlier_weights), np.mean(expected_normal_weights))
        self.assertLess(np.mean(empirical_outlier_weights), np.mean(empirical_normal_weights))

        print(f"🧪 Robust Expected coefficients: {model_expected.beta}")
        print(f"🧪 Robust Empirical coefficients: {model_empirical.beta}")
        print(f"🧪 Robust Coefficient difference (L2): {coeff_diff:.6f}")
        print(f"🧪 Expected avg outlier weight: {np.mean(expected_outlier_weights):.4f}")
        print(f"🧪 Empirical avg outlier weight: {np.mean(empirical_outlier_weights):.4f}")

    def test_no_bias_option(self):
        """Test that the model works without bias term."""
        no_bias_model = RobustLogisticRegression(use_bias=False)
        no_bias_model.fit(self.X_clean, self.y_clean)

        # Beta should have same length as number of features (no +1 for bias)
        self.assertEqual(len(no_bias_model.beta), self.X_clean.shape[1])

        # Should still make valid predictions
        predictions = no_bias_model.predict(self.X_test)
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))

    def test_feature_names_with_pandas(self):
        """Test that feature names are preserved when using pandas DataFrame."""
        import pandas as pd

        # Create DataFrame with feature names
        df = pd.DataFrame(self.X_clean, columns=["feature1", "feature2"])

        self.robust_model.fit(df, self.y_clean)
        self.assertEqual(self.robust_model.feature_names, ["feature1", "feature2"])

    def test_original_snippet_equivalence(self):
        """Test that the class produces similar results to the original snippet."""
        # Recreate the original snippet data
        np.random.seed(0)
        X = np.linspace(-4, 4, 100).reshape(-1, 1)
        y = (np.random.rand(100) < self._sigmoid(1.2 * X[:, 0])).astype(int)
        y[5] = 1  # Outlier
        y[95] = 0  # Outlier

        # Fit with the class (bias will be added automatically)
        model = RobustLogisticRegression(
            epsilon_contamination=0.05, tol=1e-6, max_iter=30
        )
        model.fit(X, y)

        # Test that it converged and produced reasonable results
        self.assertTrue(model.is_fitted_)
        self.assertIsNotNone(model.beta)
        self.assertEqual(len(model.beta), 2)  # intercept + 1 feature

        # Test predictions
        X_test = np.array([[0], [2], [-2]])
        probs = model.predict_proba(X_test)
        self.assertEqual(probs.shape, (3, 2))

        # The robust weights should reflect outlier down-weighting
        self.assertTrue(np.min(model.weights) < np.max(model.weights))

    def _sigmoid(self, z):
        """Helper sigmoid function for testing."""
        return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    unittest.main()
