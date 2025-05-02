"""
utils.py.

This module contains utility functions for the Fisher Scoring package.

A function to plot observed vs predicted probabilities for count data.
Source: J. Hilbe. Modeling Count Data. Cambridge University Press, 2014.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.special import gammaln


def plot_observed_vs_predicted(
    y,
    mu,
    max_count=15,
    alpha=None,
    title="Observed vs Predicted Probabilities",
    model_name="Model",
    ax=None,
    plot_params=None,
):
    """
    Plot observed vs predicted probabilities for count data.

    Parameters:
    - y (array-like): Observed count data.
    - mu (array-like): Predicted mean values from the model.
    - max_count (int): Maximum count to consider for probabilities.
    - alpha (float, optional): Overdispersion parameter for Negative Binomial.
      If None, assumes Poisson (alpha=0).
    - title (str): Title for the plot.
    - model_name (str): Name of the model for labeling.
    - ax (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlib axis to plot on.

    Returns:
    - None. Plots the chart on the provided axis or creates a new one.
    """
    counts = np.arange(0, max_count + 1)
    observed_probs = []
    predicted_probs = []
    observed_count = []
    predicted_count = []

    for count in counts:
        if alpha is None or alpha == 0:  # Poisson case
            pred_prob = np.mean(np.exp(-mu) * (mu**count) / np.exp(gammaln(count + 1)))
        else:  # Negative Binomial case
            amu = mu * alpha
            pred_prob = np.mean(
                np.exp(
                    count * np.log(amu / (1 + amu))
                    - (1 / alpha) * np.log(1 + amu)
                    + gammaln(count + 1 / alpha)
                    - gammaln(count + 1)
                    - gammaln(1 / alpha)
                )
            )
        # Predict counts
        obs_count = np.sum(y == count)
        pred_count = pred_prob * len(y)
        predicted_count.append(pred_count)
        observed_count.append(obs_count)
        predicted_probs.append(pred_prob)
        observed_probs.append(np.mean(y == count))

    # Create a DataFrame for plotting
    results_df = pd.DataFrame(
        {
            "Count": counts,  # Discrete count values
            "Frequency Observed": observed_count,  # Frequency of observations for each count
            "Frequency Predicted": predicted_count,  # Predicted frequency for each count
            "Probability Observed": observed_probs,  # Observed probability P(X = k)
            "Probability Predicted": predicted_probs,  # Predicted probability P(X = k)
        }
    )

    # Use the provided axis or create a new one
    if ax is None:
        ax = plt.gca()
    if plot_params in {"frequency"}:
        # Plot observed vs predicted probabilities
        ax.plot(
            results_df["Count"],
            results_df["Frequency Observed"],
            label="Observed",
            marker="o",
            linestyle="--",
            color="dodgerblue",
        )
        ax.plot(
            results_df["Count"],
            results_df["Frequency Predicted"],
            label="Predicted",
            marker="o",
            linestyle="-",
            color="red",
        )
    else:
        ax.plot(
            results_df["Count"],
            results_df["Probability Observed"],
            label="Observed",
            marker="o",
            linestyle="--",
            color="dodgerblue",
        )
        ax.plot(
            results_df["Count"],
            results_df["Probability Predicted"],
            label="Predicted",
            marker="o",
            linestyle="-",
            color="red",
        )
    ax.set_title(f"{title}\n{model_name}")
    ax.set_xlabel("Count")
    ax.set_ylabel("Probability" if plot_params is None else "Frequency")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return results_df
