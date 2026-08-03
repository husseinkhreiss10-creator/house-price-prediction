# House Price Prediction

A regression project predicting residential sale prices from property features
(quality rating, living area, basement size, garage capacity, neighborhood, etc.),
built on the Kaggle "House Prices: Advanced Regression Techniques" (Ames Housing)
dataset.

## Problem

Given ~18 structural and location features of a house, predict its sale price.
A classic supervised regression problem — useful both as a learning exercise
in the end-to-end ML workflow and as a stand-in for real-world price/valuation
modeling (e.g. property valuation, telecom pricing, insurance risk pricing).

## Approach

1. **EDA** — inspected distributions, missing values, and correlation of each
   feature with `SalePrice`.
2. **Cleaning** — imputed missing numeric values with the column median
   (computed on train, reused on test — to avoid data leakage).
3. **Feature engineering** — derived `HouseAge`, `WasRemodeled`, and `TotalSF`
   (living area + basement); encoded `KitchenQual`/`ExterQual` as ordinal
   (Ex > Gd > TA > Fa > Po) rather than one-hot, since they have a natural order.
4. **Encoding** — one-hot encoded `Neighborhood` and `HouseStyle`.
5. **Modeling** — trained and compared six approaches:
   Linear Regression, Linear Regression + log-transform, Gradient Boosting,
   Gradient Boosting + log-transform, XGBoost, and Gradient Boosting tuned
   via GridSearchCV.
6. **Evaluation** — compared RMSE and R² on a held-out 20% validation split.
7. **Submission** — applied identical preprocessing to Kaggle's `test.csv`
   and submitted predictions to the live leaderboard.

## Results (validation set)

| Model                      | RMSE      | R²     |
|-----------------------------|-----------|--------|
| **XGBoost**                 | **$26,014** | **0.912** |
| Gradient Boosting (tuned)   | $26,232   | 0.910  |
| Gradient Boosting           | $26,737   | 0.907  |
| Gradient Boosting + log     | $28,000   | 0.898  |
| Linear Regression + log     | $28,014   | 0.898  |
| Linear Regression           | $34,120   | 0.848  |

**Kaggle leaderboard score: 0.13923** (competition metric: RMSE of log(SalePrice))

Top price drivers (from feature importance): **overall quality rating**,
**total square footage**, and **neighborhood**.

## Key finding: log-transforming the target only helps linear models

One deliberate experiment in this project: does log-transforming `SalePrice`
(a common recommendation for right-skewed housing data) actually help?

- **Linear Regression:** RMSE improved ~18% with the log-transform ($34,120 → $28,014)
- **Gradient Boosting:** RMSE got *slightly worse* with the log-transform ($26,737 → $28,000)

**Why:** Linear Regression assumes normally-distributed, additive relationships,
so compressing the right-skewed price distribution directly helps it fit better.
Gradient Boosting builds decision trees that split on thresholds — a largely
monotonic-transformation-invariant approach — so the log-transform doesn't fix
anything it was struggling with, and slightly distorts the loss it's optimizing.

**Takeaway:** don't apply "best practice" transforms blindly — test whether
they actually help the specific model you're using.

## Project structure

```
house-price-prediction/
├── data/
│   └── train.csv                # from Kaggle competition
├── house_price_prediction.py    # full pipeline: EDA → clean → engineer → train → evaluate → submit
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn xgboost
python house_price_prediction.py
```

Requires `train.csv`, `test.csv`, and `sample_submission.csv` from the
[Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
in the working directory.

## What I'd improve next

- Use more of the ~79 available columns (only ~18 used here) — biggest
  remaining lever for score improvement (`GarageQual`, `BsmtFinType1`,
  `Fireplaces`, `Functional`, etc.)
- Remove known outliers (a few homes with very large `GrLivArea` but
  anomalously low price — a documented quirk of this specific dataset)
- Try target/mean encoding for `Neighborhood` instead of one-hot
- Blend XGBoost + Gradient Boosting predictions (simple averaging often
  beats either alone)
- Cross-validation instead of a single train/test split, for more reliable
  model comparison
