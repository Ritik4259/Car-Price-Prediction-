# Car Price Prediction System

## End to End Regression Pipeline using Machine Learning

UCI Machine Learning Repository - https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

A production structured Machine Learning project that predicts automobile prices using structured vehicle specifications.

This project demonstrates strong understanding of:

Data preprocessing and feature engineering
Statistical analysis and correlation testing
Regression modeling and regularization
Model evaluation and performance comparison
Clean ML pipeline construction using Scikit-learn

---

# Project Objective

## Business Problem

Automobile pricing depends on multiple technical and categorical specifications such as engine size, horsepower, fuel type, body style, and drive configuration.

The objective is to build a regression model that accurately predicts car prices using these features.

Formally,

$$
\hat{y} = f(X)
$$

Where

$X$ → Feature matrix
$y$ → Actual price
$\hat{y}$ → Predicted price

---

# Dataset

## Source

Automobile Dataset from the UCI Machine Learning Repository

The dataset contains:

26 attributes

Technical specifications

Categorical and numerical variables

Target variable → price

---

# Technical Implementation

## Data Preprocessing

Handled missing values using:

Mean imputation for continuous features

Mode imputation for categorical features

Dropped rows with missing target values

Converted data types appropriately for modeling.

---

## Feature Engineering

Created new fuel efficiency feature:

$$
\text{city-L/100km} = \frac{235}{\text{city-mpg}}
$$

Normalized dimensional features:

length
width
height

Performed horsepower binning for categorical analysis.

---

## Exploratory Data Analysis

Performed:

- Correlation matrix analysis
- Pearson correlation test
- Heatmaps and scatter plots
- Boxplots for categorical impact

Pearson coefficient formula used:

$$
r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}
$$

Strong positive correlation observed between:

- engine-size and price
- horsepower and price
- curb-weight and price

---

# Modeling Approach

## Baseline Model

Linear Regression with:

- Train Test Split 70-30
- ColumnTransformer
- OneHotEncoder
- StandardScaler

Evaluation metric:

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

---

## Feature Selection

Applied SelectKBest with f_regression to retain top predictive features and reduce noise.

---

## Regularization

### Ridge Regression

L2 regularization:

$$
\min \sum (y - \hat{y})^2 + \alpha \sum w^2
$$

Improves generalization and reduces overfitting.

---

### Lasso Regression

L1 regularization:

$$
\min \sum (y - \hat{y})^2 + \alpha \sum |w|
$$

Performs embedded feature selection.

---

# Model Evaluation

Metrics Used:

R2 Score
Mean Squared Error

$$
MSE = \frac{1}{n} \sum (y - \hat{y})^2
$$

Visualization Techniques:

- Actual vs Predicted density plots
- Residual plots
- Actual vs Predicted scatter with ideal fit line
- Pairwise feature relationship plots

---

# Key Insights

- Engine size and horsepower are strong predictors of price.
- Regularization improves generalization performance.
- Feature selection enhances model robustness.
- Proper preprocessing significantly impacts model accuracy.

---

# Tech Stack

```
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
SciPy
```

---

# Project Structure

```
carpriceprediction.py
README.md
```
---

# How to Run

Clone the repository

Install dependencies

```
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

Run the script

```
python carpriceprediction.py
```
---

# Professional Highlights

- Built an end-to-end ML regression pipeline
- Implemented statistical validation using Pearson correlation
- Applied feature selection and regularization techniques
- Produced model performance visualizations for interpretation
- Demonstrated structured ML workflow suitable for real-world deployment

