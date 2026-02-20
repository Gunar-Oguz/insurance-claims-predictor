# ============================================================
# Insurance Claims Severity Predictor
# Predicts how severe an insurance claim will be using
# the Allstate dataset from Kaggle (188K claims, 130 features)
# ============================================================

# - - - Imports - - - 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# - - - Load Data - - -
print("Loading data ...")
df = pd.read_csv('train.csv')
print(f"Dataset shape: {df.shape}")


# - - - Prepare Features and Trget - - - 
# Use only the 14 continious features for now
cont_cols = [c for c in df.columns if c.startswith('cont')]
print(f"Using {len(cont_cols)} continious fetures")

# Log transform the target to handle right skew
# Original loss: mean=$3037, median=$2115 (43% gap = heavy skew)
# Log transform compresses big values, makes distribution bell-shaped
df['log_loss'] = np.log(df['loss'])

# X = features (what the model sees), y = target (what it predicts)
X = df[cont_cols]
y = df['log_loss']

# --- Split Data ---
# 80% for training, 20% for testing on unseen data
# random_state=42 makes the split reproducible every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

print(f"Train size: {X_train.shape[0]} rows")
print(f"Test size {X_test.shape[0]} rowa")

# --- Train Models ---

# Model 1: Linear Regression
# Draws one straight line through the data
print("\nTraining Linear regression ... ")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model 2: Decision Tree (limited depth to prevent overfitting)
# Asks yes/no questions about features, max 5 levels deep
print("Training Decision Tree (max_depth = 5) ... ")
tree_model = DecisionTreeRegressor(max_depth = 5, random_state = 42)
tree_model.fit(X_train, y_train)

# --- Evaluate Models ---

# Linear Regression results
lr_pred = lr_model.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_pred) ** 0.5
lr_r2 = r2_score(y_test, lr_pred)

# Decision Tree results
tree_pred = tree_model.predict(X_test)
tree_rmse = mean_squared_error(y_test, tree_pred) ** 0.5
tree_r2 = r2_score(y_test, tree_pred)

# --- Print Results ---
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"{'Model':<25} {'R²':>8} {'RMSE':>8}")
print("-" * 50)
print(f"{'Linear Regression':<25} {lr_r2:>8.4f} {lr_rmse:>8.4f}")
print(f"{'Decision Tree (depth=5)':<25} {tree_r2:>8.4f} {tree_rmse:>8.4f}")
print("-" * 50)
print(f"\nUsing {len(cont_cols)} continuous features only")
print(f"Next step: Add 116 categorical features to improve R²")

# ============================================================
# WEEK 2: ADD CATEGORICAL FEATURES
# ============================================================

# --- Encode Categorical Features ---
# Convert letters (A, B, C) into numbers using one-hot encoding
# drop_first=True avoids the dummy variable trap
# 116 categorical columns expand to 1023 encoded columns
cat_cols = [c for c in df.columns if c.startswith('cat')]
df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)

# Combine continuous features (14) with encoded categorical features (1023)
# Total: 1037 features
X_all = pd.concat([df[cont_cols], df_encoded], axis=1)
y = df['log_loss']
print(f"\nTotal features after encoding: {X_all.shape[1]}")

# --- Split with all features ---
X_all_train, X_all_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

# --- Train with all features ---
print("Training Linear Regression with all features...")
lr_all = LinearRegression()
lr_all.fit(X_all_train, y_train)

print("Training Decision Tree (depth=5) with all features...")
tree_all = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_all.fit(X_all_train, y_train)

# --- Evaluate with all features ---
lr_all_pred = lr_all.predict(X_all_test)
lr_all_rmse = mean_squared_error(y_test, lr_all_pred) ** 0.5
lr_all_r2 = r2_score(y_test, lr_all_pred)

tree_all_pred = tree_all.predict(X_all_test)
tree_all_rmse = mean_squared_error(y_test, tree_all_pred) ** 0.5
tree_all_r2 = r2_score(y_test, tree_all_pred)

# --- Full Comparison ---
print("\n" + "=" * 55)
print("FULL MODEL COMPARISON")
print("=" * 55)
print(f"{'Model':<35} {'R2':>8} {'RMSE':>8}")
print("-" * 55)
print(f"{'LR (14 features)':<35} {lr_r2:>8.4f} {lr_rmse:>8.4f}")
print(f"{'Tree (14 features, depth=5)':<35} {tree_r2:>8.4f} {tree_rmse:>8.4f}")
print(f"{'LR (1037 features)':<35} {lr_all_r2:>8.4f} {lr_all_rmse:>8.4f}")
print(f"{'Tree (1037 features, depth=5)':<35} {tree_all_r2:>8.4f} {tree_all_rmse:>8.4f}")
print("-" * 55)

# --- Feature Importance ---
# Show top 10 most influential features by absolute weight
importance = pd.Series(lr_all.coef_, index=X_all.columns)
top_10 = importance.abs().sort_values(ascending=False).head(10)
print("\nTop 10 most important features:")
print(top_10)

# --- Save Best Model ---
# Save the best model (LR with all features, R2=0.52)
# so it can be loaded later without retraining
joblib.dump(lr_all, 'model.joblib')
print("\nBest model saved to model.joblib")


