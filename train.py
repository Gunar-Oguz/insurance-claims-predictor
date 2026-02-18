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
