# Fisher Scoring Logistic Regression

**Author:** [xRiskLab](https://github.com/xRiskLab)<br>
**Version:** Beta v0.1<br>
**License:** [MIT License](https://opensource.org/licenses/MIT) (2024)

## Overview

This repository contains Python implementations of the Fisher Scoring algorithm for various logistic regression models:

1. **Fisher Scoring Logistic Regression**: Standard logistic regression using Fisher scoring.
2. **Fisher Scoring Multinomial Regression**: Multinomial logistic regression for multi-class classification.
3. **Fisher Scoring Focal Loss Regression**: Logistic regression with focal loss for imbalanced classification problems.

The Fisher Scoring algorithm is an iterative optimization algorithm that updates model parameters using the observed or expected Fisher information matrix.

## Models

### Fisher Scoring Logistic Regression

The `FisherScoringLogisticRegression` class is a custom implementation of logistic regression using the Fisher scoring algorithm. It provides methods for fitting the model, making predictions, and computing model statistics, including standard errors, Wald statistics, p-values, and confidence intervals.

**Parameters:**
- `epsilon`: Convergence threshold for the algorithm.
- `max_iter`: Maximum number of iterations for the algorithm.
- `information`: Type of information matrix to use ('expected' or 'observed').
- `use_bias`: Include a bias term in the model.
- `significance`: Significance level for computing confidence intervals.

**Methods:**
- `fit(X, y)`: Fit the model to the data.
- `predict(X)`: Predict target labels for input data.
- `predict_proba(X)`: Predict class probabilities for input data.
- `get_params()`: Get model parameters.
- `set_params(**params)`: Set model parameters.
- `summary()`: Get a summary of model parameters, standard errors, p-values, and confidence intervals.
- `display_summary()`: Display a summary of model parameters, standard errors, p-values, and confidence intervals.

### Fisher Scoring Multinomial Regression

The `FisherScoringMultinomialRegression` class implements the Fisher Scoring algorithm for multinomial logistic regression, suitable for multi-class classification tasks.

**Parameters:**
- `epsilon`: Convergence threshold for the algorithm.
- `max_iter`: Maximum number of iterations for the algorithm.
- `information`: Type of information matrix to use ('expected' or 'observed').
- `use_bias`: Include a bias term in the model.
- `significance`: Significance level for computing confidence intervals.
- `verbose`: Enable verbose output.

**Methods:**
- `fit(X, y)`: Fit the model to the data.
- `predict(X)`: Predict target labels for input data.
- `predict_proba(X)`: Predict class probabilities for input data.
- `summary(class_idx)`: Get a summary of model parameters, standard errors, p-values, and confidence intervals for a specific class.
- `display_summary(class_idx)`: Display a summary of model parameters, standard errors, p-values, and confidence intervals for a specific class.

The algorithm is in a beta version and may require further testing and optimization to speed up matrix operations.

### Fisher Scoring Focal Loss Regression

The `FisherScoringFocalRegression` class implements the Fisher Scoring algorithm with focal loss, designed for imbalanced classification problems where the positive class is rare.

**Parameters:**
- `gamma`: Focusing parameter for focal loss.
- `epsilon`: Convergence threshold for the algorithm.
- `max_iter`: Maximum number of iterations for the algorithm.
- `information`: Type of information matrix to use ('expected' or 'observed').
- `use_bias`: Include a bias term in the model.
- `verbose`: Enable verbose output.

The algorithm is experimental and may require further testing and optimization.

## Installation

To use the models, clone the repository and install the required dependencies.

```bash
git clone https://github.com/xRiskLab/fisher-scoring.git
cd fisher-scoring
pip install -r requirements.txt
```

## Change Log

- **v0.1.3**
  - Added coefficients, standard errors, p-values, and confidence intervals for Multinomial Regression.

- **v0.1.2**
  - Updated NumPy dependency.

- **v0.1.1**
  - Added support for Python 3.9+ 🐍.

- **v0.1.0**
  - Initial release of Fisher Scoring Logistic, Multinomial, and Focal Loss Regression.
