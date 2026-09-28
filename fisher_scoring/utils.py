"""
utils.py.

This module contains utility functions for the Fisher Scoring package.

A function to plot observed vs predicted probabilities for count data.
Source: J. Hilbe. Modeling Count Data. Cambridge University Press, 2014.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from scipy.special import gammaln

from ._typing import VectorLike


def plot_observed_vs_predicted(
    y: VectorLike,
    mu: VectorLike,
    max_count: int = 15,
    alpha: Optional[float] = None,
    title: str = "Observed vs Predicted Probabilities",
    model_name: str = "Model",
    ax: Optional[Axes] = None,
    plot_params: Optional[str] = None,
) -> pd.DataFrame:
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
    - ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on.
    - plot_params (str, optional): "frequency" plots counts instead of probabilities.

    Returns:
    - pd.DataFrame with observed and predicted frequencies and probabilities per
      count. The chart is drawn on the provided axis or the current one.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
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
