# Insurance Claims Severity Predictor

Predicts how severe an insurance claim will be using the Allstate dataset from Kaggle.

## Dataset
- 188,318 claims with 130 features
- 116 categorical features, 14 continuous features
- Target: claim loss amount (right-skewed, log-transformed)

## Results

| Model | Features | R² | RMSE |
|---|---|---|---|
| Linear Regression | 14 continuous | 0.0230 | 0.7988 |
| Decision Tree (depth=5) | 14 continuous | 0.0400 | 0.7918 |
| Linear Regression | 1037 (all encoded) | 0.5178 | 0.5612 |
| Decision Tree (depth=5) | 1037 (all encoded) | 0.3769 | 0.6380 |

## Key Findings
- Categorical features contain far more predictive power than continuous features
- Adding 116 categorical features (one-hot encoded) boosted R² from 0.02 to 0.52
- Linear Regression outperformed Decision Tree with encoded features
- Top predictive features: cat116, cat113, cat105, cat103
- Unlimited Decision Trees overfit badly (R² went negative)

## What I Learned
- Log transform fixes skewed target distributions
- One-hot encoding converts categories to numbers without fake ordering
- drop_first=True avoids the dummy variable trap
- Features matter more than model choice
- Limiting tree depth prevents overfitting

## How to Run
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```