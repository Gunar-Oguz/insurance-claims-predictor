# Insurance Claims Severity Predictor

Predicts how severe an insurance claim will be using the Allstate dataset from Kaggle.

## Dataset
- 188,318 claims with 130 features
- 116 categorical features, 14 continuous features
- Target: claim loss amount (right-skewed, log-transformed)

## Current Results (continuous features only)

| Model | R² | RMSE |
|---|---|---|
| Linear Regression | 0.0230 | 0.7988 |
| Decision Tree (depth=5) | 0.0400 | 0.7918 |

## What I Learned
- Log transform fixes skewed target distributions
- Unlimited Decision Trees overfit (R² went negative)
- Limiting tree depth prevents overfitting
- 14 continuous features alone are weak predictors

## Next Steps
- Add 116 categorical features with encoding
- Try more models and hyperparameter tuning
- Add Streamlit app for predictions

## How to Run
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```