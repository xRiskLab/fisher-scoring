# Changelog

- **v2.0.6**
  - **New**: Added `DivergenceClassifier` class — maximizes FICO divergence via quadratic programming, following Hoadley (2000). Supports unconstrained (analytical Fisher LDA) and constrained (score engineering) modes with weight-of-evidence rescaling.
  - **New**: Switched build system from setuptools to hatchling with dynamic versioning.
  - **New**: Moved package from `src/fisher_scoring/` to `fisher_scoring/` at repo root.
  - **New**: Added strict mypy type checking with pandas and scikit-learn stubs.
  - **Fixed**: Numerically stable sigmoid across all logistic models to eliminate overflow warnings.
  - **CI**: Updated workflows for new package paths and enabled mypy in CI.

- **v2.0.5**
  - **New**: Added `RobustLogisticRegression` class with epsilon-contamination for outlier-resistant classification.
  - **Enhanced**: Poisson and Negative Binomial regression with empirical Fisher information matrix support.
  - **Enhanced**: Converted Negative Binomial from IWLS to proper Fisher scoring for consistency.
  - **Added**: Comprehensive offset support for Poisson regression rate modeling.
  - **Fixed**: Critical bugs in Negative Binomial prediction and standard error calculations.
  - **Added**: `summary()` and `display_summary()` methods with rich statistical output.
  - **Validated**: Mathematical correctness verified against statsmodels with machine precision accuracy.

- **v2.0.4**
  - Added a beta version of Poisson and Negative Binomial regression using Fisher Scoring.
  - Changed naming conventions for simplicity and consistency.
  - Changed poetry to uv for packaging.

- **v2.0.3**
  - Added a new functionality of inference of mean responses with confidence intervals for all algorithms.
  - Focal logistic regression now supports all model statistics, including standard errors, Wald statistics, p-values, and confidence intervals.

- **v2.0.2**
  - **Bug Fixes**: Fixed the `MultinomialLogisticRegression` class to have flexible NumPy data types.

- **v2.0.1**
  - **Bug Fixes**: Removed the debug print statement from the `LogisticRegression` class.

- **v2.0**
  - **Performance Improvements**: Performance Enhancements: Optimized matrix calculations for substantial speed and memory efficiency improvements across all models. Leveraging streamlined operations, this version achieves up to 290x faster convergence. Performance gains per model:
    - *Multinomial Logistic Regression*: Training time reduced from 125.10s to 0.43s (~290x speedup).
    - *Logistic Regression*: Training time reduced from 0.24s to 0.05s (~5x speedup).
    - *Focal Loss Logistic Regression*: Training time reduced from 0.26s to 0.01s (~26x speedup).
  - **Bug Fixes**: `verbose` parameter in Focal Loss Logistic Regression now functions as expected, providing accurate logging during training.

- **v0.1.4**
  - Updated log likelihood for Multinomial Regression and minor changes to Logistic Regression for integration with scikit-learn.

- **v0.1.3**
  - Added coefficients, standard errors, p-values, and confidence intervals for Multinomial Regression.

- **v0.1.2**
  - Updated NumPy dependency.

- **v0.1.1**
  - Added support for Python 3.9+ 🐍.

- **v0.1.0**
  - Initial release of Fisher Scoring Logistic, Multinomial, and Focal Loss Regression.
