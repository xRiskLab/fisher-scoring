"""test_fisher_scoring_bradley_terry.py."""

# Tests inspect the fitted design matrix (_fit_data) to check the information
# matrices independently of the model code.
# pylint: disable=protected-access,too-many-public-methods

import io
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Tuple

import choix
import numpy as np
import pandas as pd
from scipy.special import gammaln, xlogy
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)

import fisher_scoring
from fisher_scoring.fisher_scoring_bradley_terry import BradleyTerry, pairs_from_counts

DATA_DIR = Path(__file__).resolve().parent / "data"

# 1987 American League East (Agresti, 2002; BradleyTerry2 `baseball` data).
BASEBALL = pd.DataFrame(
    {
        "home": ["Milwaukee"] * 6 + ["Detroit"] * 6 + ["Toronto"] * 6 + ["New York"] * 6
        + ["Boston"] * 6 + ["Cleveland"] * 6 + ["Baltimore"] * 6,
        "away": [
            "Detroit", "Toronto", "New York", "Boston", "Cleveland", "Baltimore",
            "Milwaukee", "Toronto", "New York", "Boston", "Cleveland", "Baltimore",
            "Milwaukee", "Detroit", "New York", "Boston", "Cleveland", "Baltimore",
            "Milwaukee", "Detroit", "Toronto", "Boston", "Cleveland", "Baltimore",
            "Milwaukee", "Detroit", "Toronto", "New York", "Cleveland", "Baltimore",
            "Milwaukee", "Detroit", "Toronto", "New York", "Boston", "Baltimore",
            "Milwaukee", "Detroit", "Toronto", "New York", "Boston", "Cleveland",
        ],
        "home_wins": [4, 4, 4, 6, 4, 6, 3, 4, 4, 6, 6, 4, 2, 4, 2, 4, 4, 6,
                      3, 5, 2, 4, 4, 6, 5, 2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 2,
                      2, 1, 1, 2, 1, 3],
        "away_wins": [3, 2, 3, 1, 2, 0, 3, 2, 3, 0, 1, 3, 5, 3, 4, 3, 2, 0,
                      3, 1, 5, 3, 2, 1, 1, 5, 3, 2, 2, 0, 5, 3, 4, 3, 2, 4,
                      5, 5, 6, 4, 6, 4],
    }
)  # fmt: skip


def _expand(
    X: pd.DataFrame, y: pd.Series, w: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand weighted rows into one row per game: (n_games, 2) items and outcomes."""
    reps = w.astype(int).to_numpy()
    return (
        np.repeat(X.to_numpy(dtype=object), reps, axis=0),
        np.repeat(y.to_numpy(), reps),
    )


class TestBradleyTerry(unittest.TestCase):
    """Unit tests for the Fisher Scoring Bradley-Terry model."""

    def setUp(self):
        self.X, self.y, self.w = pairs_from_counts(
            BASEBALL, "home", "away", "home_wins", "away_wins"
        )
        self.model = BradleyTerry()

    def test_fit_sets_is_fitted(self):
        """Test that is_fitted_ and n_iter_ are set after fitting."""
        self.assertFalse(self.model.is_fitted_)
        self.model.fit(self.X, self.y, sample_weight=self.w)
        self.assertTrue(self.model.is_fitted_)
        self.assertGreater(self.model.n_iter_, 0)
        self.assertLess(self.model.n_iter_, self.model.max_iter)

    def test_predict_raises_not_fitted_error(self):
        """Test that predict, predict_proba and summary raise NotFittedError before fit."""
        with self.assertRaises(NotFittedError):
            self.model.predict([("Boston", "Detroit")])
        with self.assertRaises(NotFittedError):
            self.model.predict_proba([("Boston", "Detroit")])
        with self.assertRaises(NotFittedError):
            self.model.summary()

    def test_abilities_sum_to_zero(self):
        """Test that abilities are mean-centered and indexed by sorted item."""
        self.model.fit(self.X, self.y, sample_weight=self.w)
        self.assertAlmostEqual(float(self.model.abilities_.sum()), 0.0, places=10)
        self.assertEqual(list(self.model.abilities_.index), sorted(set(self.X["home"])))

    def test_matches_unpenalized_logistic_regression(self):
        """The MLE should agree with sklearn on one-hot contrasts (C -> inf)."""
        self.model.fit(self.X, self.y, sample_weight=self.w)
        eX, ey = _expand(self.X, self.y, self.w)
        teams = self.model.items_
        idx = {t: i for i, t in enumerate(teams)}
        contrasts = np.zeros((len(eX), len(teams)))
        for r, (a, b) in enumerate(eX):
            contrasts[r, idx[a]], contrasts[r, idx[b]] = 1, -1
        sk = SkLogisticRegression(
            fit_intercept=False, C=1e8, max_iter=10_000, tol=1e-12
        )
        sk.fit(np.vstack([contrasts, -contrasts]), np.concatenate([ey, 1 - ey]))
        s = sk.coef_[0]
        np.testing.assert_allclose(
            self.model.abilities_.to_numpy(), s - s.mean(), atol=1e-4
        )

    def test_matches_choix_estimators(self):
        """Same MLE as choix: MM, iterative Luce spectral ranking and Newton.

        All three choix estimators are unpenalized (alpha=0).
        """
        self.model.fit(self.X, self.y, sample_weight=self.w)
        idx = {t: k for k, t in enumerate(self.model.items_)}
        eX, ey = _expand(self.X, self.y, self.w)
        winners = np.where(ey == 1, eX[:, 0], eX[:, 1])
        losers = np.where(ey == 1, eX[:, 1], eX[:, 0])
        data = [(idx[a], idx[b]) for a, b in zip(winners, losers)]
        n_items = len(idx)
        ours = self.model.abilities_.to_numpy()
        for est in (
            choix.mm_pairwise(n_items, data, alpha=0.0, max_iter=10_000, tol=1e-12),
            choix.ilsr_pairwise(n_items, data, alpha=0.0, max_iter=10_000, tol=1e-12),
            choix.opt_pairwise(n_items, data, alpha=0.0, tol=1e-12),
        ):
            np.testing.assert_allclose(est - est.mean(), ours, atol=1e-8)
        p_ours = self.model.predict_proba([("Milwaukee", "Baltimore")])[0, 1]
        p_choix = choix.probabilities([idx["Milwaukee"], idx["Baltimore"]], ours)[0]
        self.assertAlmostEqual(p_ours, p_choix, places=10)

    def test_l2_matches_choix_penalty(self):
        """Test that l2 = 2 * alpha reproduces choix.opt_pairwise with a penalty."""
        self.model.set_params(l2=0.2)
        self.model.fit(self.X, self.y, sample_weight=self.w)
        idx = {t: k for k, t in enumerate(self.model.items_)}
        eX, ey = _expand(self.X, self.y, self.w)
        winners = np.where(ey == 1, eX[:, 0], eX[:, 1])
        losers = np.where(ey == 1, eX[:, 1], eX[:, 0])
        data = [(idx[a], idx[b]) for a, b in zip(winners, losers)]
        est = choix.opt_pairwise(len(idx), data, alpha=0.1, tol=1e-12)
        np.testing.assert_allclose(
            est - est.mean(), self.model.abilities_.to_numpy(), atol=1e-8
        )

    def test_weights_equal_expanded_rows(self):
        """Test that frequency weights give the same fit as one row per game."""
        weighted = BradleyTerry().fit(self.X, self.y, sample_weight=self.w)
        eX, ey = _expand(self.X, self.y, self.w)
        expanded = BradleyTerry().fit(eX, ey)
        np.testing.assert_allclose(
            weighted.abilities_.to_numpy(), expanded.abilities_.to_numpy(), atol=1e-8
        )
        np.testing.assert_allclose(
            weighted.standard_errors, expanded.standard_errors, atol=1e-8
        )
        self.assertAlmostEqual(
            weighted.loss_history[-1], expanded.loss_history[-1], places=6
        )

    def test_matches_bradleyterry2_reference_fit(self):
        """Compare with R's BradleyTerry2::BTm on the same data (Baltimore reference).

        The reference tables in tests/data were produced by
        tests/data/baseball_btm_reference.R with R 4.6.1 / BradleyTerry2 1.1.3.
        """
        coefs = pd.read_csv(DATA_DIR / "baseball_btm_coefficients.csv")
        fits = pd.read_csv(DATA_DIR / "baseball_btm_fit.csv").set_index("model")
        n = (BASEBALL["home_wins"] + BASEBALL["away_wins"]).to_numpy(dtype=float)
        k = BASEBALL["home_wins"].to_numpy(dtype=float)
        for name, use_bias in (("plain", False), ("home_advantage", True)):
            model = BradleyTerry(use_bias=use_bias, reference="Baltimore")
            model.fit(self.X, self.y, sample_weight=self.w)
            frame = model.summary_frame()
            ref = coefs[coefs["model"] == name].set_index("term")
            ref.index = ref.index.str.replace("at.home", "bias (order effect)")
            got = frame.loc[ref.index]
            np.testing.assert_allclose(got["estimate"], ref["estimate"], atol=1e-8)
            np.testing.assert_allclose(got["std_error"], ref["std_error"], atol=1e-5)
            np.testing.assert_allclose(got["wald"], ref["z_value"], atol=1e-4)
            np.testing.assert_allclose(got["p_value"], ref["p_value"], atol=1e-6)
            self.assertAlmostEqual(frame.loc["Baltimore", "estimate"], 0.0)
            self.assertTrue(np.isnan(frame.loc["Baltimore", "p_value"]))

            # Fit statistics: binomial deviance on the 42 aggregated cells and the
            # binomial-level log-likelihood / AIC that R's glm reports.
            p_home = model.predict_proba(BASEBALL[["home", "away"]])[:, 1]
            deviance = 2 * np.sum(
                xlogy(k, k / (n * p_home)) + xlogy(n - k, (n - k) / (n * (1 - p_home)))
            )
            loglik = model.loss_history[-1] + np.sum(
                gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            )
            n_params = len(model.beta) - 1  # one ability is fixed by identification
            self.assertEqual(n_params, fits.loc[name, "n_params"])
            self.assertAlmostEqual(deviance, fits.loc[name, "deviance"], places=6)
            self.assertAlmostEqual(loglik, fits.loc[name, "loglik"], places=6)
            self.assertAlmostEqual(
                -2 * loglik + 2 * n_params, fits.loc[name, "aic"], places=6
            )

    def test_reference_reporting_preserves_differences(self):
        """Test that a reference item shifts abilities but leaves predictions unchanged."""
        centered = BradleyTerry().fit(self.X, self.y, sample_weight=self.w)
        ref = BradleyTerry(reference="Boston").fit(self.X, self.y, sample_weight=self.w)
        diff_c = centered.abilities_ - centered.abilities_["Boston"]
        np.testing.assert_allclose(ref.abilities_.to_numpy(), diff_c.to_numpy())
        np.testing.assert_allclose(
            centered.predict_proba(self.X), ref.predict_proba(self.X)
        )

    def test_reference_standard_errors_match_reduced_information(self):
        """SEs relative to a reference equal inv(info) with the reference dropped."""
        ref = "Baltimore"
        model = BradleyTerry(reference=ref).fit(self.X, self.y, sample_weight=self.w)
        info = model.information_matrix["information"][-1]
        keep = [i for i, t in enumerate(model.items_) if t != ref]
        cov_reduced = np.linalg.inv(info[np.ix_(keep, keep)])
        np.testing.assert_allclose(
            model.standard_errors[keep], np.sqrt(np.diag(cov_reduced)), rtol=1e-8
        )

    def test_predict_proba_symmetry(self):
        """Test that P(a beats b) + P(b beats a) = 1 and predict favours the stronger item."""
        self.model.fit(self.X, self.y, sample_weight=self.w)
        p_ab = self.model.predict_proba([("Milwaukee", "Baltimore")])
        p_ba = self.model.predict_proba([("Baltimore", "Milwaukee")])
        self.assertEqual(p_ab.shape, (1, 2))
        self.assertAlmostEqual(p_ab[0, 1] + p_ba[0, 1], 1.0)
        self.assertAlmostEqual(p_ab[0, 0] + p_ab[0, 1], 1.0)
        self.assertGreater(p_ab[0, 1], 0.5)
        self.assertEqual(self.model.predict([("Milwaukee", "Baltimore")])[0], 1)

    def test_predict_ci_contains_point_estimate(self):
        """Test that both CI methods bracket the point estimate."""
        self.model.fit(self.X, self.y, sample_weight=self.w)
        p = self.model.predict_proba(self.X)[:, 1]
        for method in ("logit", "proba"):
            ci = self.model.predict_ci(self.X, method=method)
            self.assertEqual(ci.shape, (len(p), 2))
            self.assertTrue(np.all(ci[:, 0] <= p + 1e-12))
            self.assertTrue(np.all(ci[:, 1] >= p - 1e-12))
        with self.assertRaises(ValueError):
            self.model.predict_ci(self.X, method="nope")

    def test_comparison_covariate_recovery(self):
        """Recover a known comparison-level effect from an extra column of X."""
        rng = np.random.default_rng(0)
        items = np.array([f"P{i}" for i in range(8)])
        true_lambda = rng.normal(size=8)
        true_lambda -= true_lambda.mean()
        true_beta = 0.8
        n = 6000
        i = rng.integers(0, 8, n)
        j = (i + rng.integers(1, 8, n)) % 8
        x = rng.normal(size=n)
        eta = true_lambda[i] - true_lambda[j] + true_beta * x
        y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
        X = pd.DataFrame({"item1": items[i], "item2": items[j], "rest_diff": x})
        model = BradleyTerry().fit(X, y)
        self.assertEqual(model.feature_names, ["rest_diff"])
        self.assertAlmostEqual(model.coef_["rest_diff"], true_beta, delta=0.1)
        np.testing.assert_allclose(model.abilities_.to_numpy(), true_lambda, atol=0.2)
        # Prediction needs the covariate column when the model was fitted with it.
        with self.assertRaises(ValueError):
            model.predict_proba([("P0", "P1")])
        self.assertEqual(model.predict_proba([("P0", "P1", 0.0)]).shape, (1, 2))

    def test_item_feature_recovery(self):
        """Abilities explained by item covariates: lambda_i = z_i' gamma."""
        rng = np.random.default_rng(1)
        items = np.array([f"P{i}" for i in range(10)])
        Z = pd.DataFrame(rng.normal(size=(10, 2)), index=items, columns=["a", "b"])
        gamma = np.array([1.0, -0.5])
        lam = Z.to_numpy() @ gamma
        n = 6000
        i = rng.integers(0, 10, n)
        j = (i + rng.integers(1, 10, n)) % 10
        eta = lam[i] - lam[j]
        y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
        model = BradleyTerry().fit(
            np.column_stack([items[i], items[j]]), y, item_features=Z
        )
        self.assertEqual(model.param_names_, ["a", "b"])
        np.testing.assert_allclose(model.coef_.to_numpy(), gamma, atol=0.1)
        np.testing.assert_allclose(model.abilities_.to_numpy(), lam, atol=0.15)
        self.assertEqual(len(model.abilities_se_), 10)
        with self.assertRaises(ValueError):
            BradleyTerry().fit([("P0", "P1"), ("Q", "P2")], item_features=Z)

    def test_empirical_information(self):
        """Test that empirical information gives the same MLE with finite SEs."""
        model = BradleyTerry(information="empirical")
        model.fit(self.X, self.y, sample_weight=self.w)
        expected = BradleyTerry().fit(self.X, self.y, sample_weight=self.w)
        np.testing.assert_allclose(
            model.abilities_.to_numpy(), expected.abilities_.to_numpy(), atol=1e-5
        )
        self.assertTrue(np.all(np.isfinite(model.standard_errors)))

    def test_invalid_information_type(self):
        """Test that an unknown information type raises ValueError."""
        with self.assertRaises(ValueError):
            BradleyTerry(information="foo").fit(self.X, self.y)

    def test_default_y_means_item1_won(self):
        """Test that omitting y treats item1 as the winner of every row."""
        eX, ey = _expand(self.X, self.y, self.w)
        winners = np.where(ey == 1, eX[:, 0], eX[:, 1])
        losers = np.where(ey == 1, eX[:, 1], eX[:, 0])
        a = BradleyTerry().fit(np.column_stack([winners, losers]))
        b = BradleyTerry().fit(eX, ey)
        np.testing.assert_allclose(a.abilities_.to_numpy(), b.abilities_.to_numpy())

    def test_ties_count_as_half_a_win(self):
        """Test that a tie (y = 0.5) equals one win and one loss at half weight."""
        tie = BradleyTerry().fit([("A", "B"), ("B", "C"), ("A", "C")], [0.5, 1, 0])
        split = BradleyTerry().fit(
            [("A", "B"), ("A", "B"), ("B", "C"), ("A", "C")],
            [1, 0, 1, 0],
            sample_weight=[0.5, 0.5, 1, 1],
        )
        np.testing.assert_allclose(
            tie.abilities_.to_numpy(), split.abilities_.to_numpy(), atol=1e-8
        )

    def test_same_item_rows_inform_covariates(self):
        """Test that same-item rows carry no ability information but fit covariates."""
        # The A-vs-A rows alone determine the covariate effect: beta = log(3).
        same = BradleyTerry().fit(
            [
                ("A", "A", 1),
                ("A", "A", 1),
                ("A", "A", 1),
                ("A", "A", -1),
                ("A", "B", 0),
                ("B", "A", 0),
            ],
            [1, 1, 0, 0, 1, 1],
        )
        self.assertEqual(len(same.abilities_), 2)
        self.assertAlmostEqual(same.coef_.iloc[0], np.log(3), places=5)
        np.testing.assert_allclose(same.abilities_.to_numpy(), 0.0, atol=1e-8)

    def test_fit_input_validation(self):
        """Test that fit rejects malformed X, y, sample_weight, offset and features."""
        X = [("A", "B"), ("B", "C")]
        cases = {
            "y outside [0, 1]": {"y": [2, 0]},
            "y wrong length": {"y": [1]},
            "negative sample_weight": {"sample_weight": [1, -1]},
            "sample_weight wrong length": {"sample_weight": [1]},
            "offset wrong length": {"offset": [0.0]},
        }
        for label, kwargs in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                BradleyTerry().fit(X, **kwargs)
        bad_inputs = {
            "1-D X": ["A", "B"],
            "one column": [("A",), ("B",)],
            "one-column DataFrame": pd.DataFrame({"a": ["A", "B"]}),
        }
        for label, bad_X in bad_inputs.items():
            with self.subTest(label), self.assertRaises(ValueError):
                BradleyTerry().fit(bad_X)
        with self.assertRaises(ValueError):
            BradleyTerry(reference="Z").fit([("A", "B")])
        with self.assertRaises(TypeError):
            BradleyTerry().fit([("A", "B")], item_features=np.ones((2, 1)))

    def test_predict_input_validation(self):
        """Test that predict rejects unknown items and mismatched covariates."""
        model = BradleyTerry().fit(self.X, self.y, sample_weight=self.w)
        with self.assertRaises(ValueError):
            model.predict_proba([("Milwaukee", "Nowhere")])
        with self.assertRaises(ValueError):
            model.predict_proba([("Milwaukee", "Boston", 1.0)])
        with_x = BradleyTerry().fit(self.X.assign(c=1.0), self.y, sample_weight=self.w)
        with self.assertRaises(ValueError):
            with_x.predict_proba([("Milwaukee", "Boston", 1.0, 2.0)])

    def test_l2_keeps_unbeaten_item_finite(self):
        """An undefeated item has no MLE; a ridge penalty makes it finite."""
        X = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "A"), ("C", "A")]
        y = [1, 1, 1, 0, 0]  # A wins everything
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            penalized = BradleyTerry(l2=0.1).fit(X, y)
        self.assertTrue(np.all(np.isfinite(penalized.abilities_)))
        self.assertEqual(penalized.abilities_.idxmax(), "A")

    def test_summary_shapes_and_frame(self):
        """Test summary array shapes, summary_frame layout and display_summary."""
        model = BradleyTerry(use_bias=True).fit(self.X, self.y, sample_weight=self.w)
        s = model.summary()
        k = len(model.items_) + 1
        for key in (
            "betas", "standard_errors", "wald_statistic",
            "p_values", "lower_bound", "upper_bound",
        ):  # fmt: skip
            self.assertEqual(s[key].shape, (k,))
        frame = model.summary_frame()
        self.assertEqual(frame.shape, (k, 7))
        self.assertIn("bias (order effect)", frame.index)
        model.display_summary()

    def test_fisher_information_is_graph_laplacian(self):
        """Expected information = D' W D, a Laplacian over the comparison graph."""
        model = BradleyTerry().fit(self.X, self.y, sample_weight=self.w)
        info = model.fisher_information()
        self.assertEqual(list(info.index), model.items_)
        self.assertEqual(list(info.columns), model.items_)
        np.testing.assert_allclose(info.sum(axis=1).to_numpy(), 0.0, atol=1e-10)
        np.testing.assert_allclose(info.to_numpy(), info.to_numpy().T)
        self.assertTrue(np.all(np.diag(info) > 0))
        self.assertTrue(np.all(info.to_numpy()[~np.eye(7, dtype=bool)] <= 0))
        self.assertEqual(np.linalg.matrix_rank(info.to_numpy()), 6)
        # Matches what fit used at the last iteration and the stored attribute.
        np.testing.assert_allclose(
            info.to_numpy(), model.information_matrix["information"][-1], atol=1e-8
        )
        pd.testing.assert_frame_equal(info, model.fisher_information_)

    def test_fisher_information_pinv_is_centered_covariance(self):
        """Test pinv(information) and the reduced inverse against the covariances."""
        model = BradleyTerry().fit(self.X, self.y, sample_weight=self.w)
        info = model.fisher_information().to_numpy()
        np.testing.assert_allclose(np.linalg.pinv(info), model.covariance_, atol=1e-8)
        # Dropping a reference item and inverting gives the reference covariance.
        ref = BradleyTerry(reference="Boston").fit(self.X, self.y, sample_weight=self.w)
        keep = [i for i, t in enumerate(ref.items_) if t != "Boston"]
        cov_ref = np.linalg.inv(info[np.ix_(keep, keep)])
        np.testing.assert_allclose(
            cov_ref, ref.covariance_[np.ix_(keep, keep)], atol=1e-8
        )

    def test_expected_information_is_negative_hessian(self):
        """Observed information = -Hessian of the log-likelihood.

        The Hessian is computed here by second-order finite differences of the
        log-likelihood value. With the canonical logit link it coincides with
        the expected information.
        """
        model = BradleyTerry(use_bias=True).fit(self.X, self.y, sample_weight=self.w)
        D, y, w = model._fit_data
        beta = model.beta.copy()

        def loglik(b: np.ndarray) -> float:
            return model.compute_loss(y, model.logistic_function(D @ b), w)

        k = len(beta)
        h = 1e-4
        hessian = np.zeros((k, k))
        for a in range(k):
            for b in range(a, k):
                ea, eb = np.eye(k)[a] * h, np.eye(k)[b] * h
                hessian[a, b] = hessian[b, a] = (
                    loglik(beta + ea + eb)
                    - loglik(beta + ea - eb)
                    - loglik(beta - ea + eb)
                    + loglik(beta - ea - eb)
                ) / (4 * h * h)
        np.testing.assert_allclose(
            model.fisher_information("expected").to_numpy(), -hessian, atol=1e-4
        )

    def test_empirical_information_is_score_outer_product(self):
        """Empirical information = sum over observations of s_i s_i'.

        s_i is the gradient of observation i's own (weighted) log-likelihood,
        obtained here by central finite differences without using the model's
        score code.
        """
        model = BradleyTerry(use_bias=True).fit(self.X, self.y, sample_weight=self.w)
        D, y, w = model._fit_data
        beta = model.beta.copy()
        k = len(beta)
        h = 1e-6
        opg = np.zeros((k, k))
        for i in range(len(y)):
            grad = np.zeros(k)
            for a in range(k):
                e = np.eye(k)[a] * h
                lp = model.logistic_function(D[i] @ (beta + e))
                lm = model.logistic_function(D[i] @ (beta - e))
                grad[a] = (
                    model.compute_loss(y[i : i + 1], np.atleast_1d(lp))
                    - model.compute_loss(y[i : i + 1], np.atleast_1d(lm))
                ) / (2 * h)
            # A row with frequency weight w_i stands for w_i identical games,
            # each contributing (score of one game) (score of one game)'.
            opg += w[i] * np.outer(grad, grad)
        np.testing.assert_allclose(
            model.fisher_information("empirical").to_numpy(), opg, rtol=1e-5, atol=1e-6
        )

    def test_information_matrices_match_r_glm(self):
        """Both information matrices against an independent R glm on one row per game.

        See tests/data/baseball_btm_reference.R: expected = solve(vcov(g)),
        empirical = crossprod(sandwich::estfun(g)). Baltimore is the dropped
        reference column, so R's matrices are the corresponding sub-blocks.
        """
        model = BradleyTerry(use_bias=True).fit(self.X, self.y, sample_weight=self.w)
        keep = [n for n in model.param_names_ if n != "Baltimore"]
        for kind, fname in (
            ("expected", "baseball_glm_expected_information.csv"),
            ("empirical", "baseball_glm_empirical_information.csv"),
        ):
            ref = pd.read_csv(DATA_DIR / fname, index_col=0)
            ref.index = ref.index.str.replace("at.home", "bias (order effect)")
            ref.columns = ref.columns.str.replace("at.home", "bias (order effect)")
            got = model.fisher_information(kind).loc[keep, keep]
            # R's glm stops at a looser tolerance; the empirical matrix depends on
            # the residuals and so inherits that difference (~1e-5 relative).
            np.testing.assert_allclose(
                got.to_numpy(), ref.loc[keep, keep].to_numpy(), rtol=1e-4, atol=1e-6
            )

    def test_information_identity_under_correct_specification(self):
        """Bartlett identity: at the true parameter E[s s'] = E[-Hessian].

        Simulating games from the fitted model, the average empirical information
        must approach the expected information, and both inverse-information
        standard errors must match the Monte Carlo spread of the estimates.
        """
        rng = np.random.default_rng(42)
        truth = BradleyTerry(use_bias=True).fit(self.X, self.y, sample_weight=self.w)
        D, _, _ = truth._fit_data
        beta_true = truth.beta.copy()
        eX, _ = _expand(self.X, self.y, self.w)
        p_game = np.repeat(truth.logistic_function(D @ beta_true), self.w.astype(int))
        n_sim = 300
        expected_info = truth.fisher_information("expected").to_numpy()
        empirical_sum = np.zeros_like(expected_info)
        estimates, se_exp, se_emp = [], [], []
        for _ in range(n_sim):
            y_sim = (rng.random(len(p_game)) < p_game).astype(float)
            fit = BradleyTerry(use_bias=True).fit(eX, y_sim)
            estimates.append(fit.beta)
            se_exp.append(fit.standard_errors)
            se_emp.append(
                BradleyTerry(use_bias=True, information="empirical")
                .fit(eX, y_sim)
                .standard_errors
            )
            # Empirical information evaluated at the *true* parameter.
            p_true = truth.logistic_function(fit._fit_data[0] @ beta_true)
            empirical_sum += truth.compute_information_matrix(
                fit._fit_data[0], p_true, y_sim, np.ones(len(y_sim)), "empirical"
            )
        mean_empirical = empirical_sum / n_sim
        # Element-wise agreement within Monte Carlo error (entries are ~17 on the
        # diagonal; the standard error of a mean of 300 Bernoulli-variance sums is
        # well under 0.5).
        np.testing.assert_allclose(mean_empirical, expected_info, atol=0.6)
        mc_sd = np.std(np.array(estimates), axis=0, ddof=1)
        np.testing.assert_allclose(np.mean(se_exp, axis=0), mc_sd, rtol=0.15)
        np.testing.assert_allclose(np.mean(se_emp, axis=0), mc_sd, rtol=0.15)

    def test_fisher_information_variants(self):
        """Test expected, empirical and penalized information matrices."""
        model = BradleyTerry(use_bias=True, l2=0.5).fit(
            self.X, self.y, sample_weight=self.w
        )
        expected = model.fisher_information("expected")
        empirical = model.fisher_information("empirical")
        self.assertEqual(expected.shape, (8, 8))
        self.assertIn("bias (order effect)", expected.index)
        self.assertFalse(np.allclose(expected.to_numpy(), empirical.to_numpy()))
        penalized = model.fisher_information(penalized=True)
        np.testing.assert_allclose(
            penalized.to_numpy() - expected.to_numpy(), 0.5 * np.eye(8)
        )
        with self.assertRaises(ValueError):
            model.fisher_information("observed")
        with self.assertRaises(NotFittedError):
            BradleyTerry().fisher_information()

    def test_reference_row_is_flagged_not_tested(self):
        """The reference item is fixed at zero: no SE, no test, flagged explicitly."""
        model = BradleyTerry(reference="Boston").fit(
            self.X, self.y, sample_weight=self.w
        )
        frame = model.summary_frame()
        self.assertTrue(frame.loc["Boston", "reference"])
        self.assertEqual(frame["reference"].sum(), 1)
        self.assertEqual(frame.loc["Boston", "estimate"], 0.0)
        self.assertEqual(frame.loc["Boston", "std_error"], 0.0)
        self.assertTrue(np.isnan(frame.loc["Boston", "wald"]))
        self.assertTrue(np.isnan(frame.loc["Boston", "p_value"]))
        self.assertTrue(
            np.isfinite(frame.loc[frame.index != "Boston", "p_value"]).all()
        )
        model.display_summary()  # renders "0 (reference)" rather than nan
        centered = BradleyTerry().fit(self.X, self.y, sample_weight=self.w)
        self.assertFalse(centered.summary_frame()["reference"].any())
        self.assertTrue(np.isfinite(centered.summary_frame()["p_value"]).all())

    def test_get_set_params(self):
        """Test get_params and set_params."""
        params = self.model.get_params()
        self.assertEqual(params["information"], "expected")
        self.assertEqual(params["max_halvings"], 20)
        self.model.set_params(max_iter=5, use_bias=True)
        self.assertEqual(self.model.max_iter, 5)
        self.assertTrue(self.model.use_bias)

    def test_sklearn_clone(self):
        """Test that sklearn.clone copies parameters and returns an unfitted model."""
        model = BradleyTerry(use_bias=True, l2=0.1, reference="Boston")
        model.fit(self.X, self.y, sample_weight=self.w)
        copy = clone(model)
        self.assertEqual(copy.get_params(), model.get_params())
        self.assertFalse(copy.is_fitted_)

    def test_refit_resets_state(self):
        """Test that refitting does not accumulate history or change max_iter."""
        self.model.fit(self.X, self.y, sample_weight=self.w)
        first = self.model.n_iter_
        self.model.fit(self.X, self.y, sample_weight=self.w)
        self.assertEqual(self.model.n_iter_, first)
        self.assertEqual(len(self.model.loss_history), first)
        self.assertEqual(len(self.model.information_matrix["information"]), first)
        self.assertEqual(self.model.max_iter, 100)

    def test_max_iter_warns_without_convergence(self):
        """Test that hitting max_iter emits a warning."""
        with self.assertWarns(UserWarning):
            BradleyTerry(max_iter=1).fit(self.X, self.y, sample_weight=self.w)

    def test_rank_sorted_by_ability(self):
        """Test that rank returns items in descending ability with SEs."""
        self.model.fit(self.X, self.y, sample_weight=self.w)
        ranking = self.model.rank()
        self.assertEqual(list(ranking.columns), ["ability", "se"])
        self.assertEqual(set(ranking.index), set(self.model.items_))
        self.assertTrue(ranking["ability"].is_monotonic_decreasing)
        self.assertEqual(ranking.index[0], "Milwaukee")

    def test_predict_returns_binary_labels(self):
        """Test that predict returns 0/1 labels of the right shape."""
        self.model.fit(self.X, self.y, sample_weight=self.w)
        labels = self.model.predict(self.X)
        self.assertEqual(labels.shape, (len(self.X),))
        self.assertTrue(np.isin(labels, [0, 1]).all())

    def test_verbose_output(self):
        """Test that verbose=True prints the iteration log."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            BradleyTerry(verbose=True).fit(self.X, self.y, sample_weight=self.w)
        output = buffer.getvalue()
        self.assertIn("Starting Fisher Scoring Iterations...", output)
        self.assertIn("Convergence reached", output)

    def test_damped_steps_keep_the_likelihood_increasing(self):
        """Step halving keeps the penalised log-likelihood non-decreasing.

        A large offset puts most comparisons near p = 0 or 1, where a full
        Fisher scoring step overshoots.
        """
        rng = np.random.default_rng(0)
        n = 4000
        X = pd.DataFrame({"item1": "c", "item2": "c", "x": rng.normal(size=n)})
        offset = 6.0 * rng.normal(size=n)  # extreme known logits
        y = rng.random(n) < BradleyTerry.logistic_function(offset + 0.5 * X["x"])
        model = BradleyTerry(l2=1e-6).fit(X, y.astype(float), offset=offset)
        self.assertTrue(np.all(np.isfinite(model.beta)))
        self.assertAlmostEqual(model.coef_["x"], 0.5, delta=0.3)
        self.assertTrue(np.all(np.diff(model.loss_history) >= -1e-6))
        # No halving needed on a well-behaved problem: plain Fisher scoring steps.
        plain = BradleyTerry().fit(self.X, self.y, sample_weight=self.w)
        self.assertEqual(plain.n_iter_, 5)

    def test_offset_is_added_and_not_estimated(self):
        """A known logit term enters the fit and the predictions unchanged."""
        eX, ey = _expand(self.X, self.y, self.w)
        base = BradleyTerry().fit(eX, ey)
        pb = base.predict_proba(eX)
        off = np.log(pb[:, 1] / pb[:, 0])
        # offset = the base model's own logits: the refit has nothing left to explain
        refit = BradleyTerry(l2=1e-6).fit(eX, ey, offset=off)
        np.testing.assert_allclose(refit.abilities_.to_numpy(), 0.0, atol=1e-4)
        np.testing.assert_allclose(refit.predict_proba(eX, offset=off), pb, atol=1e-4)
        # a constant offset equal to the fitted order effect reproduces the biased fit
        biased = BradleyTerry(use_bias=True).fit(eX, ey)
        const = np.full(len(ey), biased.coef_.iloc[0])
        with_off = BradleyTerry().fit(eX, ey, offset=const)
        np.testing.assert_allclose(
            with_off.abilities_.to_numpy(), biased.abilities_.to_numpy(), atol=1e-6
        )
        with self.assertRaises(ValueError):
            BradleyTerry().fit(eX, ey, offset=[1.0, 2.0])

    def test_accepts_tuples_arrays_and_dataframes(self):
        """Test that X as tuples, an object array or a DataFrame gives the same fit."""
        eX, ey = _expand(self.X, self.y, self.w)
        fits = [
            BradleyTerry().fit(X, ey)
            for X in (
                [tuple(row) for row in eX],
                eX,
                pd.DataFrame(eX, columns=["home", "away"]),
            )
        ]
        for fit in fits[1:]:
            np.testing.assert_allclose(
                fit.abilities_.to_numpy(), fits[0].abilities_.to_numpy()
            )

    def test_score_is_mean_log_likelihood(self):
        """Test that score is the weighted mean log-likelihood per comparison."""
        self.model.fit(self.X, self.y, sample_weight=self.w)
        p = self.model.predict_proba(self.X)[:, 1]
        y, w = self.y.to_numpy(), self.w.to_numpy()
        expected = np.sum(w * (xlogy(y, p) + xlogy(1 - y, 1 - p))) / np.sum(w)
        got = self.model.score(self.X, self.y, sample_weight=self.w)
        self.assertAlmostEqual(got, expected, places=12)
        self.assertLess(got, 0.0)
        # Ties are scored as half a win, not rejected as a non-binary label.
        ties = np.full(len(self.X), 0.5)
        self.assertTrue(np.isfinite(self.model.score(self.X, ties)))

    def test_train_test_split(self):
        """Test that train_test_split output fits and predicts directly."""
        eX, ey = _expand(self.X, self.y, self.w)
        X_train, X_test, y_train, y_test = train_test_split(
            eX, ey, test_size=0.25, random_state=0, stratify=ey
        )
        model = BradleyTerry().fit(X_train, y_train)
        self.assertEqual(model.predict_proba(X_test).shape, (len(X_test), 2))
        self.assertTrue(np.isfinite(model.score(X_test, y_test)))

    def test_cross_val_score(self):
        """Test cross_val_score with the default score (mean log-likelihood)."""
        eX, ey = _expand(self.X, self.y, self.w)
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(BradleyTerry(), eX, ey, cv=cv)
        self.assertEqual(scores.shape, (5,))
        self.assertTrue(np.all(np.isfinite(scores)))
        # Better than a coin flip, whose log-likelihood is log(0.5) per game.
        self.assertGreater(scores.mean(), np.log(0.5))

    def test_grid_search_over_l2(self):
        """Test that GridSearchCV tunes l2 and refits the best model."""
        eX, ey = _expand(self.X, self.y, self.w)
        cv = KFold(n_splits=3, shuffle=True, random_state=0)
        search = GridSearchCV(BradleyTerry(), {"l2": [0.0, 1.0, 1000.0]}, cv=cv)
        search.fit(eX, ey)
        # A huge penalty shrinks every ability to zero, which fits worse.
        self.assertNotEqual(search.best_params_["l2"], 1000.0)
        self.assertTrue(search.best_estimator_.is_fitted_)
        self.assertEqual(search.predict_proba(eX[:3]).shape, (3, 2))

    def test_pairs_from_counts_layout(self):
        """Test that pairs_from_counts emits a win row and a loss row per cell."""
        df = pd.DataFrame({"a": ["X"], "b": ["Y"], "wa": [3], "wb": [2]})
        X, y, w = pairs_from_counts(df, "a", "b", "wa", "wb")
        self.assertEqual(list(X.columns), ["a", "b"])
        self.assertEqual(list(X.itertuples(index=False, name=None)), [("X", "Y")] * 2)
        self.assertEqual(list(y), [1.0, 0.0])
        self.assertEqual(list(w), [3.0, 2.0])

    def test_pairs_from_counts_carries_covariates(self):
        """Test that covariate columns follow the item columns in X."""
        df = pd.DataFrame(
            {
                "a": ["X", "Y"],
                "b": ["Y", "X"],
                "rest": [1.0, -1.0],
                "wa": [3, 0],
                "wb": [2, 4],
            }
        )
        X, y, w = pairs_from_counts(df, "a", "b", "wa", "wb", covariates=["rest"])
        self.assertEqual(list(X.columns), ["a", "b", "rest"])
        self.assertEqual(len(X), 3)  # the zero-count row is dropped
        self.assertEqual(list(w), [3.0, 2.0, 4.0])
        model = BradleyTerry(l2=0.1).fit(X, y, sample_weight=w)
        self.assertEqual(model.feature_names, ["rest"])

    def test_package_exports(self):
        """Test that BradleyTerry and pairs_from_counts are exported at top level."""
        self.assertIs(fisher_scoring.BradleyTerry, BradleyTerry)
        self.assertIs(fisher_scoring.pairs_from_counts, pairs_from_counts)

    def test_pairs_from_counts_drops_zero_rows(self):
        """Test that pairs_from_counts drops zero-count rows and preserves total games."""
        n_zero = int(
            (BASEBALL["home_wins"] == 0).sum() + (BASEBALL["away_wins"] == 0).sum()
        )
        self.assertEqual(len(self.X), 2 * len(BASEBALL) - n_zero)
        self.assertEqual(
            float(self.w.sum()),
            float(BASEBALL[["home_wins", "away_wins"]].to_numpy().sum()),
        )


if __name__ == "__main__":
    unittest.main()
