"""test_fisher_scoring_divergence.py."""

import unittest

import numpy as np
from fisher_scoring.fisher_scoring_divergence import DivergenceClassifier
from sklearn.exceptions import NotFittedError


class TestDivergenceClassifier(unittest.TestCase):
    """Unit tests for the DivergenceClassifier."""

    def setUp(self):
        self.model = DivergenceClassifier()
        np.random.seed(42)
        # Two well-separated classes
        n = 200
        self.X = np.vstack([
            np.random.randn(n, 3) + [2, 0, 1],
            np.random.randn(n, 3) + [-1, 1, -1],
        ])
        self.y = np.array([0] * n + [1] * n)

    def test_fit_sets_is_fitted(self):
        """Test that is_fitted_ is set after fitting."""
        self.assertFalse(self.model.is_fitted_)
        self.model.fit(self.X, self.y)
        self.assertTrue(self.model.is_fitted_)

    def test_predict_raises_not_fitted_error(self):
        """Test that predict raises NotFittedError before fit."""
        with self.assertRaises(NotFittedError):
            self.model.predict(self.X)

    def test_predict_proba_raises_not_fitted_error(self):
        """Test that predict_proba raises NotFittedError before fit."""
        with self.assertRaises(NotFittedError):
            self.model.predict_proba(self.X)

    def test_score_samples_raises_not_fitted_error(self):
        """Test that score_samples raises NotFittedError before fit."""
        with self.assertRaises(NotFittedError):
            self.model.score_samples(self.X)

    def test_fit_predict(self):
        """Test that predictions have correct shape."""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)

    def test_predict_proba_shape(self):
        """Test that predict_proba returns (n_samples, 2)."""
        self.model.fit(self.X, self.y)
        proba = self.model.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.y), 2))

    def test_predict_proba_sums_to_one(self):
        """Test that probabilities sum to 1."""
        self.model.fit(self.X, self.y)
        proba = self.model.predict_proba(self.X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_predict_proba_range(self):
        """Test that probabilities are in [0, 1]."""
        self.model.fit(self.X, self.y)
        proba = self.model.predict_proba(self.X)
        self.assertTrue(np.all(proba >= 0))
        self.assertTrue(np.all(proba <= 1))

    def test_predict_binary(self):
        """Test that predictions are 0 or 1."""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_divergence_positive(self):
        """Test that divergence is positive for separable data."""
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.divergence_)
        assert self.model.divergence_ is not None
        self.assertGreater(self.model.divergence_, 0)

    def test_woe_rescale(self):
        """Test that WoE rescaling gives d'W = W'CW."""
        self.model.fit(self.X, self.y)
        assert self.model.weights_ is not None
        assert self.model.d_ is not None
        assert self.model.C_ is not None
        W = self.model.weights_
        d = self.model.d_
        C = self.model.C_
        dW = d @ W
        WCW = W @ C @ W
        np.testing.assert_allclose(dW, WCW, rtol=1e-6)

    def test_divergence_matches_manual(self):
        """Test that stored divergence matches manual computation."""
        self.model.fit(self.X, self.y)
        assert self.model.weights_ is not None
        scores = self.X @ self.model.weights_
        manual_div = DivergenceClassifier.compute_divergence(scores, self.y)
        assert self.model.divergence_ is not None
        np.testing.assert_allclose(self.model.divergence_, manual_div, rtol=1e-4)

    def test_score_samples_shape(self):
        """Test that score_samples returns correct shape."""
        self.model.fit(self.X, self.y)
        scores = self.model.score_samples(self.X)
        self.assertEqual(scores.shape, (len(self.y),))

    def test_no_bias(self):
        """Test fitting without bias."""
        model = DivergenceClassifier(use_bias=False)
        model.fit(self.X, self.y)
        self.assertEqual(model.bias_, 0.0)
        assert model.beta_ is not None
        self.assertEqual(len(model.beta_), self.X.shape[1])

    def test_with_bias(self):
        """Test fitting with bias adds one extra parameter."""
        model = DivergenceClassifier(use_bias=True)
        model.fit(self.X, self.y)
        assert model.beta_ is not None
        self.assertEqual(len(model.beta_), self.X.shape[1] + 1)

    def test_penalty(self):
        """Test that ridge penalty shrinks weights."""
        model_no_pen = DivergenceClassifier(penalty=0.0)
        model_pen = DivergenceClassifier(penalty=10.0)
        model_no_pen.fit(self.X, self.y)
        model_pen.fit(self.X, self.y)
        assert model_no_pen.weights_ is not None
        assert model_pen.weights_ is not None
        norm_no_pen = np.linalg.norm(model_no_pen.weights_)
        norm_pen = np.linalg.norm(model_pen.weights_)
        self.assertLess(norm_pen, norm_no_pen)

    def test_get_set_params(self):
        """Test sklearn get_params/set_params compatibility."""
        params = self.model.get_params()
        self.assertIn("delta", params)
        self.assertIn("use_bias", params)
        self.model.set_params(delta=2.0)
        self.assertEqual(self.model.delta, 2.0)

    def test_summary_after_fit(self):
        """Test that summary returns correct keys."""
        self.model.fit(self.X, self.y)
        s = self.model.summary()
        self.assertIn("betas", s)

    def test_constrained_solver(self):
        """Test that providing constraints triggers the QP solver."""
        A_eq = np.array([[1.0, -1.0, 0.0]])  # weight[0] == weight[1]
        b_eq = np.array([0.0])
        model = DivergenceClassifier(A_eq=A_eq, b_eq=b_eq)
        model.fit(self.X, self.y)
        assert model.weights_ is not None
        np.testing.assert_allclose(
            model.weights_[0], model.weights_[1], atol=1e-4
        )

    def test_compute_divergence_static(self):
        """Test the static compute_divergence method."""
        scores = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        y = np.array([0, 0, 0, 1, 1, 1])
        div = DivergenceClassifier.compute_divergence(scores, y)
        self.assertGreater(div, 0)

    def test_classes_attribute(self):
        """Test that classes_ is set after fit."""
        self.model.fit(self.X, self.y)
        np.testing.assert_array_equal(self.model.classes_, np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
