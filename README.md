# Linear Regression and Regularization — Ames Housing

Coding Project 1 of *Machine Learning in Finance & Insurance* (ETH Zurich, Fall 2026).

**Group members:** Paul Vidal, Victor Denner, Jules Roux, Thibaut Doulat, Célian Dalla-Vale

The goal is to predict house sale prices in Ames, Iowa (1,460 houses, 79 explanatory variables) and to compare ordinary least squares with four regularization methods.

## Contents

| File | Description |
|---|---|
| `Project1_ML.ipynb` | Fully executed notebook: answers to Questions 1–3 and the project report |
| `Housing.csv` | Data set (1,460 houses, 35 numerical and 44 categorical predictors, target `SalePrice`) |
| `data_description.txt` | Description of every variable, including the variables where `NA` is a real category |
| `requirements.txt` | Python packages used |

## Approach

1. **Data preparation.** `NA` is kept as a category in the 14 variables where it means "no such feature" (e.g. no garage) and treated as missing elsewhere. The target is modelled as `log(SalePrice)`, since `SalePrice` is strongly right-skewed. 70/30 train/test split; imputation, standardization and one-hot encoding (311 columns) are fitted on the training set only.
2. **OLS on the numerical features**, with `sklearn`, with matrix algebra (normal equations and Moore–Penrose pseudoinverse) and with `statsmodels`, including a rank and singular-value analysis of the design matrix.
3. **OLS on all features**, then **Ridge, Lasso, Elastic Net and truncated pseudoinverse (TPI)**, each tuned with `GridSearchCV` (8-fold CV, same folds for all methods) inside a `Pipeline` that refits the preprocessing in every fold.

## Main results

| Model | Selected hyperparameter(s) | Mean CV MSE (log scale) | Test $R^2$ (price scale) |
|---|---|---|---|
| OLS, numerical features | – | – | 0.881 |
| OLS, all features | – | – | 0.874 |
| Ridge | $\alpha = 30$ | 0.0226 | 0.908 |
| Lasso | $\alpha = 0.001$ | 0.0230 | 0.916 |
| Elastic Net | $\alpha = 0.003$, `l1_ratio` = 0.25 | 0.0225 | 0.912 |
| TPI | 150 singular directions | 0.0228 | 0.901 |

All regularized models outperform both OLS benchmarks out of sample. Elastic Net has the lowest cross-validation error and is our recommended model, although the differences between the four regularized methods are within the CV noise.

## How to run

```bash
pip install -r requirements.txt
jupyter lab Project1_ML.ipynb
```

Keep `Housing.csv` in the same folder as the notebook, then run all cells from top to bottom.

**Reproducibility note.** All random steps use `random_state=42`. The numerical design matrix is rank-deficient (rank 34 for 36 columns), so the normal-equation coefficients of the eight collinear surface variables in Question 2.b, and possibly the last digits of some OLS results, can differ from one computer or library version to another. The fitted values, the regularized models and the conclusions do not change.

## Data

Ames Housing data set (De Cock, 2011, *Journal of Statistics Education*), as provided by the course.
