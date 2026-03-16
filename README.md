# Airbnb Listing Popularity Prediction

## Overview

This project builds a machine learning model to predict **Airbnb listing popularity**, measured by **reviews per month**, using listing features such as price, location, room type, and host activity.

The objective is to identify which factors influence listing popularity and build a predictive model that can estimate expected review activity for Airbnb listings.

The dataset used is the **New York City Airbnb Open Data (2019)**.

## Dataset

Source:  
https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

After cleaning the dataset, the project uses approximately **38,843 Airbnb listings**.

Target variable: `reviews_per_month`

This represents how frequently a listing receives reviews and acts as a proxy for listing popularity.

## Feature Engineering

Several additional features were created to capture pricing dynamics, location effects, and host activity.

| Feature | Description |
|------|------|
| price_per_review | Price divided by number of reviews |
| total_expected_reviews | Availability relative to minimum nights |
| centrality | Distance from Times Square (proxy for city center) |
| host_experience | Host listing count scaled by availability |
| price_per_minimum_night | Price normalized by minimum stay requirement |

These features aim to capture economic and spatial signals affecting listing popularity.

## Machine Learning Pipeline

A preprocessing pipeline was built to handle both numerical and categorical variables.

### Numerical Features

- SimpleImputer(strategy="median")
- StandardScaler

### Categorical Features

- SimpleImputer(strategy="most_frequent")
- OneHotEncoder(handle_unknown="ignore")

This preprocessing ensures that all models receive properly scaled and encoded data.

## Models Evaluated

Multiple machine learning models were compared.

| Model | Cross Validation MAE |
|------|------|
| DummyRegressor | 1.256 |
| Ridge Regression | 0.884 |
| KNN | 0.771 |
| Gradient Boosting | 0.710 |
| Random Forest | 0.701 |

Tree-based models performed significantly better than linear models, indicating that the relationships between variables are likely **nonlinear**.

## Hyperparameter Tuning

Hyperparameter tuning was performed using **RandomizedSearchCV**.

Example parameters tuned for the Random Forest model:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

The best-performing configuration used **600 trees** with optimized split and leaf constraints.

## Final Model

The final model selected was a **tuned RandomForestRegressor**.

Performance:

- Cross Validation MAE: 0.684
- Test MAE: 0.693

This means that on average, predictions differ from the true value by approximately **0.693 reviews per month**.

## Model Diagnostics

Several diagnostic visualizations were generated to evaluate the model:

- Feature importance plots
- Permutation importance analysis
- Residual plots
- Actual vs predicted comparisons
- Learning curves
- Correlation matrix of numerical variables
- Distribution analysis of key features

These analyses help evaluate model reliability and potential overfitting.

## Feature Importance

The most influential features identified by the Random Forest model include:

1. number_of_reviews
2. price_per_review
3. total_expected_reviews
4. availability_365
5. host_experience
6. minimum_nights
7. geographic location (latitude and longitude)

Permutation importance confirmed similar patterns.

## Key Insights

Several insights emerged from the analysis:

- Listings with **higher availability and more host activity** tend to receive more reviews.
- Pricing dynamics significantly influence listing popularity.
- Geographic location remains an important factor.
- Tree-based models outperform linear models due to nonlinear relationships between features.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

## Repository Structure

```text
airbnb-listing-popularity-prediction
│
├── airbnb_project.ipynb
├── data
│   └── AB_NYC_2019.csv
├── requirements.txt
└── README.md