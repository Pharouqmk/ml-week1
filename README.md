# ShopSmart Retail Analytics & AI Implementation
## Machine Learning Track — Week 1

**Author:** Umar Muhammad
**Programme:** AnalystLab Africa Experience Lab, Batch F
**Client:** ShopSmart Retail Ltd.

## Project Summary

This project builds two baseline machine learning models for ShopSmart Retail Ltd. using historical retail data (customers, orders, products, stores, returns, inventory, and suppliers):

1. **Regression model** — predicts an order's total value (`Order_Total`) using Linear Regression
2. **Classification model** — predicts whether an order will be returned (`Return_Flag`) using Logistic Regression, with class balancing to address a 4.88% return rate

## Key Results

| Model | Metric | Result |
|---|---|---|
| Regression | R² Score | 0.8876 |
| Regression | MAE | ₦29,879.88 |
| Classification (balanced) | Recall | 46.11% |
| Classification (balanced) | F1 Score | 0.0864 |

Full details, methodology, and business insights are in `reports/ML_Report.pdf`.

## Folder Structure

```
Week1_MachineLearning/
├── ML01_Foundations.ipynb      # Full notebook: data prep, modeling, evaluation
├── data/
│   ├── master_dataset.csv      # Fully merged and feature-engineered dataset
│   ├── train.csv                # Training split (80%)
│   └── test.csv                  # Test split (20%)
├── models/
│   ├── linear_regression.pkl    # Trained regression model
│   ├── logistic_regression.pkl  # Trained classification model
│   └── scaler.pkl                # Fitted StandardScaler
├── reports/
│   ├── ML_Report.pdf             # Full written report (all parts)
│   └── Presentation.pptx         # Summary slide deck
├── requirements.txt
└── README.md
```

## How to Reproduce

1. Install dependencies: `pip install -r requirements.txt`
2. Open `ML01_Foundations.ipynb` in Jupyter or Google Colab
3. Run cells in order — the notebook loads the 9 source CSVs, cleans and merges them, engineers features, trains both models, and evaluates them
4. Trained models can be reloaded directly:
   ```python
   import joblib
   lr_model = joblib.load('models/linear_regression.pkl')
   log_model = joblib.load('models/logistic_regression.pkl')
   scaler = joblib.load('models/scaler.pkl')
   ```

## Notable Findings

- Fixed a data join bug during preparation: Inventory initially matched on Product+Store, causing 96% of rows to fail; corrected to a Product-only join.
- Identified and excluded a case of trivial model "leakage" — an interaction term that let the regression model simply reconstruct known arithmetic (R²=1.0), rather than learn genuine patterns.
- Demonstrated how accuracy can mislead on imbalanced data: an unbalanced classifier scored 95.12% accuracy while catching 0% of actual returns; a balanced model traded accuracy for real recall (46.11%).
- Found that return likelihood is most associated with customer age group (56+), not product or pricing factors — a genuine, actionable business insight.
