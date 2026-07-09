# Car Price Analysis — Capstone Unit 2

## Project Overview

This project analyses a car price dataset from the US automobile market 
to identify key pricing factors and build a machine learning model to 
predict car prices.

The project follows the CRISP-DM methodology and includes:
* ETL pipeline for data cleaning and feature engineering
* Exploratory data analysis with hypothesis testing
* Machine learning model (ExtraTreesRegressor, R²=0.911)
* Interactive Tableau dashboard for data storytelling
* Streamlit app for interactive price predictions

**Live App:** [Streamlit App](XXXXXXXX)
**Tableau Dashboard:** [Car Price Analysis](https://public.tableau.com/shared/N6ZSWTY39?:display_count=n&:origin=viz_share_link)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Business Hypotheses](#business-hypotheses)
4. [ETL Pipeline](#etl-pipeline)
5. [EDA & Hypothesis Testing](#eda--hypothesis-testing)
6. [Machine Learning](#machine-learning)
7. [Dashboard & Visualisation](#dashboard--visualisation)
8. [Streamlit App](#streamlit-app)
9. [Project Management](#project-management)
10. [Technologies Used](#technologies-used)
11. [How to Run](#how-to-run)
12. [Credits](#credits)

---

## Dataset

* **Source:** [Kaggle — Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)
* **Size:** 205 rows, 26 columns
* **Target variable:** Price (USD)
* **Features:** Engine size, horsepower, curb weight, body style, car brand and more

### Data Dictionary
| Column | Description |
|---|---|
| CarName | Car brand and model name |
| fueltype | Fuel type (gas/diesel) |
| carbody | Body style (sedan, hatchback, etc.) |
| enginesize | Engine displacement |
| horsepower | Engine power output |
| curbweight | Weight of the car |
| price | Car price in USD (target variable) |

---

## Business Hypotheses

| Hypothesis | Result | p-value |
|---|---|---|
| H1: Engine size and horsepower are the strongest predictors of price | ✅ Partially confirmed | Correlation: 0.87 |
| H2: Luxury brands have significantly higher prices than economy brands | ✅ Confirmed | 0.0000 |
| H3: Diesel cars have higher prices than petrol cars | ❌ Rejected | 0.8659 |
| H4: Car body style significantly influences price | ✅ Confirmed | 0.0000 |

### Key Finding
Engine size (correlation: 0.87) is the strongest predictor of car price, 
followed by curbweight (0.84) and horsepower (0.81).

----

## ETL Pipeline

### Extract
* Raw dataset loaded from Kaggle (205 rows, 26 columns)
* No missing values found

### Transform
* Ordinal encoding of cylindernumber and doornumber (text→integers)
* Price distribution analysed (mean $13,276, median $10,295, std $7,988)
* 15 price outliers identified using IQR method — retained as legitimate luxury vehicles

### Feature Engineering
* **price_per_horsepower** — cost efficiency relative to performance
* **price_per_enginesize** — cost relative to engine displacement
* Note: Both features excluded from ML to avoid data leakage

### Load
* Cleaned dataset saved to `outputs/datasets/cleaned/car_prices_cleaned.csv`

### Note
* CarBrand extraction and typo corrections are handled in the ML pipeline (03_ML.ipynb) 
  to ensure consistent transformations during model training and prediction

---

## EDA & Hypothesis Testing

### Statistical Methods Used
* Pearson correlation — to measure relationships between numerical variables
* Independent samples t-test (parametric) — to compare two groups
* One-way ANOVA (parametric) — to compare more than two groups

### Hypothesis Results

**H1 — Engine size and horsepower are the strongest predictors of price**
* Engine size correlation with price: 0.87
* Horsepower correlation with price: 0.81
* Unexpectedly, curbweight (0.84) outperformed horsepower
* Result: Partially confirmed ✅

**H2 — Luxury brands have significantly higher prices than economy brands**
* Brands above $20,000 average classified as luxury (Jaguar, Porsche, BMW, Buick)
* T-test p-value: 0.0000
* Result: Confirmed ✅

**H3 — Diesel cars have higher prices than petrol cars**
* Dataset imbalance: 185 gas vs 20 diesel cars
* Balanced sampling used (20 vs 20) for fair comparison
* T-test p-value: 0.8659
* Result: Rejected ❌

**H4 — Car body style significantly influences price**
* Convertibles have highest median price
* ANOVA p-value: 0.0000
* Result: Confirmed ✅

----
## Machine Learning

### Pipeline
* OrdinalEncoder — categorical variable encoding
* SimpleImputer — handles missing values
* StandardScaler — feature scaling
* SelectFromModel — automatic feature selection

### Algorithm Selection
Five regression algorithms were tested using a two-step search strategy:
1. Quick search with default hyperparameters
2. Extensive hyperparameter optimisation with GridSearchCV

### Best Model: ExtraTreesRegressor
* **Hyperparameters:** max_depth=10, min_samples_leaf=1, n_estimators=100
* **Train R²:** 0.998
* **Test R²:** 0.911
* **Test RMSE:** $2,619
* **Test MAE:** $1,572

### Feature Importance
| Feature | Importance |
|---|---|
| enginesize | 0.20 |
| curbweight | 0.18 |
| cylindernumber | 0.15 |
| carwidth | 0.12 |
| horsepower | 0.11 |
| citympg | 0.09 |
| drivewheel | 0.08 |
| highwaympg | 0.07 |

### Note on Overfitting
The gap between train R² (0.998) and test R² (0.911) indicates mild overfitting, 
partly expected given the small dataset size (205 rows).

### Target Variable Transformation
To address mild overfitting and right-skewed price distribution (skewness=1.78), 
four transformers were tested:

| Transformer | Mean R² (CV) |
|---|---|
| No transform | **0.9129** |
| PowerTransformer | 0.9125 |
| LogTransformer | 0.8971 |
| YeoJohnsonTransformer | 0.8918 |
| BoxCoxTransformer | 0.8914 |

No transformation improved model performance — tree-based models are inherently 
robust to skewness in the target variable.

---

