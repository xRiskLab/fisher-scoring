"""test_utils.py."""

import unittest

import matplotlib.pyplot as plt
import numpy as np
from fisher_scoring.utils import plot_observed_vs_predicted


class TestUtils(unittest.TestCase):
    """Test cases for the utility functions in utils.py."""

    def test_plot_observed_vs_predicted(self):
        """Test the plot_observed_vs_predicted function for Poisson data."""
        y = np.array([0, 1, 2, 3, 4, 5])
        mu = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

        # Create a plot
        fig, ax = plt.subplots()
        results_df = plot_observed_vs_predicted(y, mu, ax=ax)

        # Check if the results DataFrame is not None
        self.assertIsNotNone(results_df)

        # Check if the DataFrame contains the expected columns
        expected_columns = [
            "Count",
            "Frequency Observed",
            "Frequency Predicted",
            "Probability Observed",
            "Probability Predicted",
        ]
        self.assertTrue(
            all(column in results_df.columns for column in expected_columns)
        )


if __name__ == "__main__":
    unittest.main()
