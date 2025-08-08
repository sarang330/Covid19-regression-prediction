COVID-19 Case Prediction Using Regression Models

#Project Overview
This project builds an end-to-end machine learning pipeline to forecast COVID-19 case trends using multiple regression techniques. The goal is to compare different models and evaluate their performance on real COVID-19 case data.

The models used include:

1. Linear Regression
2. Polynomial Regression
3. Bayesian Ridge Regression
4. Polynomial Ridge Regression

The project covers data preprocessing, feature engineering, model training, evaluation, and result visualization.

📂 Dataset
Source: covid_19_data.csv (or replace with external link if too large)

The dataset contains:

1. Date
2. Country/Region
3. Confirmed Cases
4. Deaths
5. Recovered Cases

⚙️ Features Implemented
-> Data loading and cleaning
->Feature engineering for date/time analysis
-> Training multiple regression models
-> Model performance evaluation (MAE, MSE)
-> Comparison table of model results
-> Insights & conclusions

📊 Results Summary
| Model                       | MAE         | MSE                    |
| --------------------------- | ----------- | ---------------------- |
| Polynomial Regression       | 77,294,775  | 8,068,180,713,630,094  |
| Polynomial Ridge Regression | 86,576,701  | 10,138,729,099,548,060 |
| Bayesian Ridge Regression   | 247,320,826 | 61,303,228,096,201,056 |
| Linear Regression           | 247,340,631 | 61,313,046,731,319,840 |


Best Model: Polynomial Regression — lowest MAE and MSE.

