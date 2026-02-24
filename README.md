# 🏠 House Price Prediction

End-to-end regression project predicting house sale prices using the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset. Covers the full ML workflow from EDA to model deployment.

---

## 📌 Project Overview

Given 79 features describing residential homes in Ames, Iowa, the goal is to predict the final sale price of each home. This project walks through every stage of a real-world regression pipeline including deep EDA, thoughtful missing value imputation, feature engineering, and a comparison between a baseline linear model and XGBoost.

---

## 📂 Repository Structure

```
├── house_price_prediction.ipynb   # Main notebook with full pipeline
├── submission.csv         # Final predictions for Kaggle leaderboard
└── README.md
```

---

## 🔄 Workflow

### 1. Exploratory Data Analysis
- Analyzed distribution of the target variable `SalePrice`
- Identified right skew and applied `log1p` transformation to normalize distribution
- Built a correlation heatmap to identify top predictive features
- Key finding: `OverallQual`, `GrLivArea`, and `GarageArea` are the strongest predictors

### 2. Data Preprocessing
- Combined train and test sets before preprocessing to ensure consistency
- **Missing value strategy:**
  - Numerical features (e.g. `GarageArea`, `BsmtFinSF1`) → filled with `0` (absence of feature)
  - `LotFrontage` → filled with neighborhood median (smarter than global median)
  - Categorical features (e.g. `PoolQC`, `Alley`) → filled with `'None'` (absence of feature)
  - Other categoricals (e.g. `MSZoning`, `Electrical`) → filled with mode

### 3. Feature Engineering
Created three new features to improve predictive power:
- `TotalSF` = Basement + 1st Floor + 2nd Floor square footage
- `TotalBath` = Full baths + (0.5 × Half baths) across all floors
- `Age` = Year sold − Year built

### 4. Categorical Encoding
Applied `pd.get_dummies` with `drop_first=True` to one-hot encode all categorical variables, eliminating multicollinearity.

### 5. Feature Scaling
Used `StandardScaler` — fit only on training data, then applied to validation and test sets to prevent data leakage.

### 6. Modeling

| Model | Description |
|---|---|
| Linear Regression | Baseline model assuming linear feature relationships |
| XGBoost | Gradient boosted trees — sequential tree building where each tree corrects the errors of the previous |

### 7. Evaluation

Models evaluated on a held-out validation set (20% of training data) using:
- **RMSE** — primary metric, penalizes large errors
- **MAE** — average absolute error, easy to interpret
- **R²** — proportion of variance explained by the model

XGBoost significantly outperformed Linear Regression on all three metrics.

### 8. Submission
Predictions were reverse-transformed using `np.expm1` to convert log-prices back to actual dollar values before submission.

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

## 📈 Key Concepts Demonstrated

- Log transformation for skewed targets
- Stratified missing value imputation
- Train/validation split with proper scaler fitting (no data leakage)
- Gradient boosted trees (XGBoost)
- Regression evaluation metrics (RMSE, MAE, R²)

---

## 🚀 How to Run

1. Clone this repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place the CSVs in the working directory
3. Open `21_Days_B6_D3.ipynb` in Jupyter or Google Colab and run all cells
