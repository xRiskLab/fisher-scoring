"""Shared type aliases for the public estimator APIs."""

from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

# Feature matrix of shape (n_samples, n_features).
MatrixLike = Union[np.ndarray, pd.DataFrame, Sequence[Sequence[float]]]

# One numeric value per sample, e.g. targets, weights or offsets.
VectorLike = Union[np.ndarray, pd.Series, Sequence[float]]

# One row per comparison: (item1, item2, *covariates), e.g. a list of tuples,
# a 2-D object array or a DataFrame.
ComparisonsLike = Union[np.ndarray, pd.DataFrame, Sequence[Sequence[Any]]]
