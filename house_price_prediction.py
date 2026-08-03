"""
House Price Prediction — end-to-end ML regression pipeline.

Trains and compares 6 models on the Kaggle "House Prices: Advanced
Regression Techniques" dataset, then generates a submission file.

Requires train.csv, test.csv, and sample_submission.csv in the working
directory (download from the Kaggle competition data page).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# 1. LOAD TRAINING DATA
# ---------------------------------------------------------------------------
df_full = pd.read_csv("data/train.csv")

feature_cols = ["OverallQual", "OverallCond", "GrLivArea", "TotalBsmtSF",
                 "GarageCars", "GarageArea", "FullBath", "HalfBath",
                 "BedroomAbvGr", "YearBuilt", "YearRemodAdd", "LotArea",
                 "1stFlrSF", "2ndFlrSF", "KitchenQual", "ExterQual",
                 "Neighborhood", "HouseStyle"]

df = df_full[feature_cols + ["SalePrice"]].copy()

# ---------------------------------------------------------------------------
# 2. ENCODE ORDINAL QUALITY COLUMNS
# ---------------------------------------------------------------------------
qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
df["KitchenQual"] = df["KitchenQual"].map(qual_map)
df["ExterQual"] = df["ExterQual"].map(qual_map)

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["HouseAge"] = 2024 - df["YearBuilt"]
df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
df["WasRemodeled"] = (df["YearRemodAdd"] > df["YearBuilt"]).astype(int)

# ---------------------------------------------------------------------------
# 3.5. HANDLE MISSING VALUES (before encoding/splitting)
# ---------------------------------------------------------------------------
print("Missing values before fixing:")
print(df.isnull().sum()[df.isnull().sum() > 0])

numeric_cols_with_na = df.select_dtypes(include=[np.number]).columns[
    df.select_dtypes(include=[np.number]).isnull().any()
]
train_medians = {}
for col in numeric_cols_with_na:
    median_val = df[col].median()
    train_medians[col] = median_val
    df[col] = df[col].fillna(median_val)
    print(f"Filled {col} with median: {median_val}")

# ---------------------------------------------------------------------------
# 4. ENCODE REMAINING CATEGORICALS
# ---------------------------------------------------------------------------
df_encoded = pd.get_dummies(df, columns=["Neighborhood", "HouseStyle"], drop_first=True)
print(f"\nShape after encoding: {df_encoded.shape}")

# ---------------------------------------------------------------------------
# 5. TRAIN/TEST SPLITS (internal validation split)
# ---------------------------------------------------------------------------
X = df_encoded.drop(columns=["SalePrice"])
y = df_encoded["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_log = np.log(y)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_test_actual = np.exp(y_test_log)

assert X_train.isnull().sum().sum() == 0, "X_train still has NaNs!"
assert X_test.isnull().sum().sum() == 0, "X_test still has NaNs!"
print("\nNo missing values in train/test sets — safe to proceed.")

results = {}

# ---------------------------------------------------------------------------
# MODEL 1: Linear Regression (raw target)
# ---------------------------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
results["Linear Regression"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, lr_preds)),
    "R2": r2_score(y_test, lr_preds),
}

# ---------------------------------------------------------------------------
# MODEL 2: Linear Regression + log-transform
# ---------------------------------------------------------------------------
lr_log_model = LinearRegression()
lr_log_model.fit(X_train_log, y_train_log)
lr_log_preds_actual = np.exp(lr_log_model.predict(X_test_log))
results["Linear Regression + log"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test_actual, lr_log_preds_actual)),
    "R2": r2_score(y_test_actual, lr_log_preds_actual),
}

# ---------------------------------------------------------------------------
# MODEL 3: Gradient Boosting (raw target)
# ---------------------------------------------------------------------------
gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)
results["Gradient Boosting"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, gb_preds)),
    "R2": r2_score(y_test, gb_preds),
}

# ---------------------------------------------------------------------------
# MODEL 4: Gradient Boosting + log-transform
# ---------------------------------------------------------------------------
gb_log_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
gb_log_model.fit(X_train_log, y_train_log)
gb_log_preds_actual = np.exp(gb_log_model.predict(X_test_log))
results["Gradient Boosting + log"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test_actual, gb_log_preds_actual)),
    "R2": r2_score(y_test_actual, gb_log_preds_actual),
}

# ---------------------------------------------------------------------------
# MODEL 5: XGBoost (raw target)
# ---------------------------------------------------------------------------
xgb_model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
results["XGBoost"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, xgb_preds)),
    "R2": r2_score(y_test, xgb_preds),
}

# ---------------------------------------------------------------------------
# MODEL 6: Gradient Boosting tuned with GridSearchCV
# ---------------------------------------------------------------------------
param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
}
grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"\nBest GridSearchCV params: {grid_search.best_params_}")

best_gb_model = grid_search.best_estimator_
best_gb_preds = best_gb_model.predict(X_test)
results["Gradient Boosting (tuned)"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, best_gb_preds)),
    "R2": r2_score(y_test, best_gb_preds),
}

# ---------------------------------------------------------------------------
# 6. COMPARE ALL SIX MODELS
# ---------------------------------------------------------------------------
comparison = pd.DataFrame(results).T.sort_values("RMSE")
print("\n" + "=" * 55)
print("MODEL COMPARISON (sorted by RMSE, best first)")
print("=" * 55)
print(comparison.to_string(float_format=lambda x: f"{x:,.4f}"))

best_model_name = comparison.index[0]
print(f"\nBest model: {best_model_name}")

model_lookup = {
    "Gradient Boosting": gb_model,
    "Gradient Boosting + log": gb_log_model,
    "XGBoost": xgb_model,
    "Gradient Boosting (tuned)": best_gb_model,
}

if best_model_name in model_lookup:
    best_model_obj = model_lookup[best_model_name]
    importance = pd.Series(best_model_obj.feature_importances_, index=X.columns)
    print(f"\nTop 10 most important features ({best_model_name}):")
    print(importance.sort_values(ascending=False).head(10))

# ===========================================================================
# 7. KAGGLE SUBMISSION — apply identical preprocessing to test.csv
# ===========================================================================
print("\n" + "=" * 55)
print("BUILDING KAGGLE SUBMISSION")
print("=" * 55)

test_df = pd.read_csv("data/test.csv")

test_feature_cols = ["Id"] + feature_cols
test = test_df[test_feature_cols].copy()

# Same ordinal mapping
test["KitchenQual"] = test["KitchenQual"].map(qual_map)
test["ExterQual"] = test["ExterQual"].map(qual_map)

# Fill missing values BEFORE feature engineering, so NaNs don't propagate
print("\nMissing values in test set (before fixing):")
print(test.isnull().sum()[test.isnull().sum() > 0])

numeric_test_cols = test.select_dtypes(include=[np.number]).columns
for col in numeric_test_cols:
    if test[col].isnull().sum() > 0:
        fill_val = train_medians.get(col, df[col].median() if col in df.columns else 0)
        test[col] = test[col].fillna(fill_val)
        print(f"Filled {col} with: {fill_val}")

# Feature engineering (on already-clean columns)
test["HouseAge"] = 2024 - test["YearBuilt"]
test["TotalSF"] = test["GrLivArea"] + test["TotalBsmtSF"]
test["WasRemodeled"] = (test["YearRemodAdd"] > test["YearBuilt"]).astype(int)

print(f"\nRemaining NaNs after fixing: {test.isnull().sum().sum()}")

# Same one-hot encoding, aligned to training columns
test_encoded = pd.get_dummies(test, columns=["Neighborhood", "HouseStyle"], drop_first=True)
test_encoded = test_encoded.reindex(columns=X.columns.tolist() + ["Id"], fill_value=0)

assert test_encoded.drop(columns=["Id"]).isnull().sum().sum() == 0, "Test set still has NaNs!"
print("Test set cleaned and aligned — safe to predict.")

# ---------------------------------------------------------------------------
# 8. PREDICT WITH BEST MODEL AND BUILD SUBMISSION FILE
# ---------------------------------------------------------------------------
test_ids = test_encoded["Id"]
test_features = test_encoded.drop(columns=["Id"])

final_model = model_lookup.get(best_model_name, xgb_model)
final_preds = final_model.predict(test_features)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": final_preds
})
submission.to_csv("submission.csv", index=False)

print(f"\nsubmission.csv created with {len(submission)} rows")
print(submission.head())

# ---------------------------------------------------------------------------
# 9. SANITY CHECK AGAINST SAMPLE FORMAT
# ---------------------------------------------------------------------------
sample = pd.read_csv("data/sample_submission.csv")
print("\nSample submission format:")
print(sample.head())
print(f"\nSample shape: {sample.shape} | Your submission shape: {submission.shape}")
