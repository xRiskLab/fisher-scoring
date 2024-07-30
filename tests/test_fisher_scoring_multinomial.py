import unittest

import numpy as np
from fisher_scoring_multinomial import (
    FisherScoringMultinomialRegression,
)
from sklearn.exceptions import NotFittedError


class TestFisherScoringMultinomialRegression(
    unittest.TestCase
):

    def setUp(self):
        self.model = FisherScoringMultinomialRegression()
        # Generate a synthetic dataset with 3 classes
        np.random.seed(0)
        self.X = np.random.rand(100, 5)
        self.y = np.random.randint(0, 3, 100)

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
            ((predictions >= 0) & (predictions < 3)).all(),
            "Predictions should be within the class range.",
        )

    def test_fit_predict_proba(self):
        self.model.fit(self.X, self.y)
        probabilities = self.model.predict_proba(self.X)
        self.assertEqual(
            probabilities.shape,
            (self.y.shape[0], 3),
            "Probabilities shape should match the number of samples and classes.",
        )
        self.assertTrue(
            (probabilities >= 0).all()
            and (probabilities <= 1).all(),
            "Probabilities should be between 0 and 1.",
        )


if __name__ == "__main__":
    unittest.main()
