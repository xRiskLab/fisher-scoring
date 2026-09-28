"""test_fisher_scoring_multinomial.py."""

import io
import unittest
from contextlib import redirect_stdout
from typing import Tuple

import numpy as np
import statsmodels.api as sm
from fisher_scoring.fisher_scoring_multinomial import MultinomialLogisticRegression
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression as SkLogisticRegression


def _simulate(n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Three classes, two features, well-specified multinomial logit."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    B = np.array([[0.0, 0.5, -0.5], [0.0, 1.0, -0.5], [0.0, -0.8, 0.6]])
    eta = np.column_stack([np.ones(n), X]) @ B
    P = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y = (P.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    return X, y


def _contrast_matrix(n_features: int, n_classes: int) -> np.ndarray:
    """Map class-stacked coefficients to beta_k - beta_0 for k = 1..K-1."""
    C = np.zeros(((n_classes - 1) * n_features, n_classes * n_features))
    eye = np.eye(n_features)
    for k in range(1, n_classes):
        rows = slice((k - 1) * n_features, k * n_features)
        C[rows, k * n_features : (k + 1) * n_features] = eye
        C[rows, :n_features] = -eye
    return C


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

    def test_matches_statsmodels_mnlogit(self):
        """Test class contrasts and their standard errors against statsmodels."""
        X, y = _simulate(1500)
        model = MultinomialLogisticRegression().fit(X, y)
        reference = sm.MNLogit(y, sm.add_constant(X)).fit(disp=0)
        n_features, n_classes = model.beta.shape
        np.testing.assert_allclose(
            model.beta[:, 1:] - model.beta[:, [0]], reference.params, atol=1e-8
        )
        C = _contrast_matrix(n_features, n_classes)
        se = np.sqrt(np.diag(C @ model.covariance_ @ C.T))
        np.testing.assert_allclose(
            se.reshape(n_classes - 1, n_features).T, reference.bse, rtol=1e-6
        )

    def test_matches_sklearn_unpenalized(self):
        """Test probabilities and centred coef_/intercept_ against sklearn."""
        X, y = _simulate(1500)
        model = MultinomialLogisticRegression().fit(X, y)
        sk = SkLogisticRegression(penalty=None, max_iter=10_000, tol=1e-12).fit(X, y)
        # Older sklearn lbfgs stops ~1e-5 short of the optimum (1.3.0); the exact
        # comparison is the statsmodels test above.
        np.testing.assert_allclose(
            model.predict_proba(X), sk.predict_proba(X), atol=1e-4
        )
        np.testing.assert_allclose(
            model.coef_, sk.coef_ - sk.coef_.mean(axis=0), atol=1e-3
        )
        np.testing.assert_allclose(
            model.intercept_, sk.intercept_ - sk.intercept_.mean(), atol=1e-3
        )
        self.assertEqual(model.coef_.shape, (3, 2))
        self.assertEqual(model.intercept_.shape, (3,))

    def test_coefficients_sum_to_zero_across_classes(self):
        """Test the sum-to-zero identification of the K class coefficient vectors."""
        X, y = _simulate(500)
        model = MultinomialLogisticRegression().fit(X, y)
        np.testing.assert_allclose(model.beta.sum(axis=1), 0.0, atol=1e-10)

    def test_standard_errors_differ_by_class(self):
        """Test that each class gets its own standard errors."""
        X, y = _simulate(1500)
        model = MultinomialLogisticRegression().fit(X, y)
        se = [model.summary(k)["standard_errors"] for k in range(3)]
        self.assertFalse(np.allclose(se[0], se[1]))
        self.assertFalse(np.allclose(se[1], se[2]))

    def test_covariance_is_pseudo_inverse_of_information(self):
        """Test that the covariance is the pseudo-inverse of the information."""
        X, y = _simulate(500)
        model = MultinomialLogisticRegression().fit(X, y)
        info = model.information_matrix["information"][-1]
        np.testing.assert_allclose(model.covariance_, np.linalg.pinv(info), atol=1e-10)

    def test_information_matrices_match_definitions(self):
        """Test expected and empirical information against a per-row sum."""
        X, y = _simulate(200)
        Xd = np.column_stack([np.ones(len(X)), X])
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(3), size=len(X))
        y_one_hot = np.eye(3)[y]
        expected = sum(
            np.kron(np.diag(pi) - np.outer(pi, pi), np.outer(xi, xi))
            for pi, xi in zip(p, Xd)
        )
        empirical = sum(
            np.outer(np.kron(ri, xi), np.kron(ri, xi))
            for ri, xi in zip(y_one_hot - p, Xd)
        )
        compute = MultinomialLogisticRegression.compute_information_matrix
        np.testing.assert_allclose(
            compute(Xd, y_one_hot, p, "expected"), expected, atol=1e-10
        )
        np.testing.assert_allclose(
            compute(Xd, y_one_hot, p, "empirical"), empirical, atol=1e-10
        )
        with self.assertRaises(ValueError):
            compute(Xd, y_one_hot, p, "observed")

    def test_empirical_information_same_estimates(self):
        """Test that empirical information converges to the same MLE."""
        X, y = _simulate(1500)
        expected = MultinomialLogisticRegression().fit(X, y)
        empirical = MultinomialLogisticRegression(information="empirical").fit(X, y)
        np.testing.assert_allclose(empirical.beta, expected.beta, atol=1e-8)

    def test_converges_in_few_iterations(self):
        """Test that full Fisher scoring converges quadratically."""
        X, y = _simulate(1500)
        model = MultinomialLogisticRegression().fit(X, y)
        self.assertLess(model.n_iter_, 15)

    def test_arbitrary_class_labels(self):
        """Test that labels need not be 0..K-1 and predict returns classes_."""
        X, y = _simulate(500)
        for labels in (y + 1, np.array(["a", "b", "c"])[y]):
            model = MultinomialLogisticRegression().fit(X, labels)
            np.testing.assert_array_equal(model.classes_, np.unique(labels))
            self.assertTrue(np.isin(model.predict(X), model.classes_).all())
        integer = MultinomialLogisticRegression().fit(X, y)
        named = MultinomialLogisticRegression().fit(X, np.array(["a", "b", "c"])[y])
        np.testing.assert_allclose(integer.beta, named.beta)

    def test_refit_resets_state(self):
        """Test that refitting does not accumulate history."""
        X, y = _simulate(500)
        self.model.fit(X, y)
        first = self.model.n_iter_
        self.model.fit(X, y)
        self.assertEqual(len(self.model.loss_history), first)
        self.assertEqual(len(self.model.information_matrix["information"]), first)

    def test_quiet_unless_verbose(self):
        """Test that fit prints only with verbose=True."""
        X, y = _simulate(200)
        quiet, loud = io.StringIO(), io.StringIO()
        with redirect_stdout(quiet):
            MultinomialLogisticRegression().fit(X, y)
        with redirect_stdout(loud):
            MultinomialLogisticRegression(verbose=True).fit(X, y)
        self.assertEqual(quiet.getvalue(), "")
        self.assertIn("Convergence reached", loud.getvalue())

    def test_max_iter_warns_without_convergence(self):
        """Test that hitting max_iter emits a warning."""
        X, y = _simulate(200)
        with self.assertWarns(UserWarning):
            MultinomialLogisticRegression(max_iter=1).fit(X, y)

    def test_predict_ci_contains_point_estimate(self):
        """Test that both CI methods bracket the predicted probabilities."""
        X, y = _simulate(500)
        model = MultinomialLogisticRegression().fit(X, y)
        p = model.predict_proba(X)
        for method in ("logit", "proba"):
            ci = model.predict_ci(X, method=method)
            for k in range(3):
                self.assertTrue(np.all(ci[k][:, 0] <= p[:, k] + 1e-12))
                self.assertTrue(np.all(ci[k][:, 1] >= p[:, k] - 1e-12))
        with self.assertRaises(ValueError):
            model.predict_ci(X, method="nope")

    def test_sklearn_clone(self):
        """Test that sklearn.clone returns an unfitted copy with the same params."""
        X, y = _simulate(200)
        model = MultinomialLogisticRegression(information="empirical").fit(X, y)
        copy = clone(model)
        self.assertEqual(copy.get_params(), model.get_params())
        self.assertFalse(copy.is_fitted_)


if __name__ == "__main__":
    unittest.main()
