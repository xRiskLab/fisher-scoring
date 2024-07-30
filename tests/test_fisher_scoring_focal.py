import unittest

import numpy as np
from fisher_scoring_focal import (
    FisherScoringFocalRegression,
)
from fisher_scoring_logistic import (
    FisherScoringLogisticRegression,
)
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score


class TestFisherScoringFocalRegression(unittest.TestCase):

    def setUp(self):
        self.model = FisherScoringFocalRegression()
        # Generate a synthetic dataset with Bernoulli distribution
        np.random.seed(0)
        self.X = np.random.rand(100, 2)
        self.y = np.random.randint(0, 2, 100)

    def test_fit_sets_is_fitted(self):
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
        with self.assertRaises(NotFittedError):
            self.model.predict(self.X)

    def test_predict_proba_raises_not_fitted_error(self):
        with self.assertRaises(NotFittedError):
            self.model.predict_proba(self.X)

    def test_fit_predict(self):
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
        self.model.fit(self.X, self.y)
        probabilities = self.model.predict_proba(self.X)[
            :, 1
        ]
        self.assertEqual(
            probabilities.shape,
            (self.y.shape[0],),
            "Probabilities shape should match the number of samples.",
        )
        self.assertTrue(
            (probabilities >= 0).all()
            and (probabilities <= 1).all(),
            "Probabilities should be between 0 and 1.",
        )

    def test_focal_vs_standard_logistic(self):
        model_standard = FisherScoringLogisticRegression()
        model_focal = FisherScoringFocalRegression(
            gamma=0.0
        )

        model_standard.fit(self.X, self.y)
        model_focal.fit(self.X, self.y)

        probas_standard = model_standard.predict_proba(
            self.X
        )[:, 1]
        probas_focal = model_focal.predict_proba(self.X)[
            :, 1
        ]

        gini_standard = (
            2 * roc_auc_score(self.y, probas_standard) - 1
        )
        gini_focal = (
            2 * roc_auc_score(self.y, probas_focal) - 1
        )

        self.assertAlmostEqual(
            gini_standard,
            gini_focal,
            places=4,
            msg="Gini coefficients should be approximately equal for gamma=0.0",
        )


if __name__ == "__main__":
    unittest.main()