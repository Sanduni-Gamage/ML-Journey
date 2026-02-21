# 📌 Machine Learning Fundamentals 

A comprehensive machine learning project covering **supervised learning** (regression & classification), **unsupervised learning** (K-Means clustering), and **reinforcement learning** (Q-Learning). Built as part of a structured ML course across two sessions.

## 📂 Project Structure

```
Machine-Learning-Projects/
├── Datasets/                              # Training datasets
├── machine-learning-part1.ipynb           # Session 1: ML fundamentals & core concepts
├── machine-learning-part2.ipynb           # Session 2: Regression, classification, regularization
├── requirements.txt                       # Python dependencies
└── .gitignore
```

## 📌 What's Covered

### Part 1 — ML Fundamentals & Core Concepts

- **Linear Regression** on California Housing dataset (EDA, correlation heatmap, train/test split, evaluation metrics, residual analysis, learning curves)
- **Unsupervised Learning**: K-Means customer segmentation on synthetic data (elbow method, cluster visualization)
- **Reinforcement Learning**: Q-Learning grid world navigation (reward curves, policy visualization, state visits, animated trajectories)
- **Features & Labels** with Wine classification using Random Forest (100% test accuracy)
- **Complete ML Pipeline**: Data loading → EDA → splitting → scaling → training → evaluation
- **Underfitting vs Overfitting**: Polynomial degree comparison (degree 1, 5, 15) on synthetic sine data
- **Data Handling**: Loading, EDA, and preprocessing on synthetic customer data (missing value imputation, outlier removal via IQR, feature scaling, categorical encoding)

### Part 2 — Regression, Classification & Regularization

- **Simple Linear Regression** — CO2 emissions from engine size (R² = 0.76)
- **Multivariable Linear Regression** — CO2 from 6 vehicle features (R² = 0.90)
- **Linear Regression Assumptions** — visual examples of linearity, homoscedasticity, normality (good vs violated)
- **Comprehensive CO2 Analysis** — EDA, VIF multicollinearity check, OLS vs Ridge vs Lasso vs ElasticNet comparison, feature importance, residual diagnostics
- **Polynomial Regression** — capturing non-linear relationships, degree comparison (underfitting → good fit → overfitting)
- **Evaluation Metrics Deep Dive** — MSE, RMSE, MAE, R², Adjusted R² across normal, outlier, and heteroscedastic datasets
- **Company Profit Prediction** — linear vs polynomial regression on 1000 Companies dataset (R² = 0.98)
- **Logistic Regression** — sigmoid function, binary classification on Iris, multi-class decision boundaries
- **Decision Trees** — Iris classification with decision boundary visualization
- **Wine Quality Classification** — multi-class on UCI Red Wine dataset with detailed classification reports
- **Bias-Variance Tradeoff** — training on 50 random datasets to decompose bias², variance, and irreducible error
- **Regularization** — Ridge (L2), Lasso (L1), ElasticNet with alpha tuning via GridSearchCV
- **Model Saving/Loading** with `joblib`

## 📊 Datasets Used

| Dataset | Task | Type | Source |
|---------|------|------|--------|
| California Housing | House price prediction | Regression | scikit-learn |
| FuelConsumptionCo2.csv | CO2 emission prediction | Regression | IBM |
| 1000_Companies.csv | Company profit prediction | Regression | — |
| Wine (sklearn) | Wine type classification | Classification | scikit-learn |
| Iris | Flower classification | Classification | scikit-learn |
| Wine Quality (Red) | Quality rating prediction | Classification | UCI ML Repository |
| Synthetic Customer Data | Customer segmentation | Clustering | Generated |
| GridWorld | Navigation agent | Reinforcement Learning | Generated |

## 🔑 Key Results

| Model | Dataset | Metric |
|-------|---------|--------|
| Linear Regression (1 feature) | CO2 Emissions | R² = 0.762 |
| Multivariable Linear Regression | CO2 Emissions | R² = 0.903 |
| OLS / Ridge / Lasso / ElasticNet | CO2 Emissions | R² ≈ 0.903 |
| Linear Regression | California Housing | R² = 0.576 |
| Linear Regression | Company Profit | R² = 0.983 |
| Random Forest Classifier | Wine (sklearn) | 100% accuracy |
| Logistic Regression | Iris (multi-class) | 97.3% accuracy |
| Decision Tree (depth=2) | Iris | 96.0% accuracy |
| Q-Learning Agent | 5×5 GridWorld | Avg reward = 92.94 |

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/GamithaManawadu/Machine-Learning-Projects.git
cd Machine-Learning-Projects

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🛠 Tech Stack

- **Python 3.8+**
- pandas, numpy - data manipulation
- matplotlib, seaborn - visualization
- scikit-learn - ML models, preprocessing, and evaluation
- statsmodels - VIF and statistical tests
- scipy - residual analysis and statistical distributions
- joblib - model persistence

