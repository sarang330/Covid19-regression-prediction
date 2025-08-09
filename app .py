
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(page_title="COVID-19 Regression Dashboard", layout="wide")

# -----------------------
# Sidebar - About Section
# -----------------------
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This dashboard analyzes COVID-19 trends using multiple regression models.
    Data is sourced from the [COVID-19 Data Repository](https://github.com/datasets/covid-19).
    You can:
    - Select any country
    - Choose models and configure parameters
    - View results and download them
    """
)

# -----------------------
# Sidebar - Data Selection
# -----------------------
st.sidebar.header("📂 Data Options")
data_source = st.sidebar.radio("Choose Data Source:", ["Default GitHub Data", "Upload CSV"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your COVID-19 dataset", type=["csv"])
    if uploaded_file:
        df_orig = pd.read_csv(uploaded_file)
    else:
        st.stop()
else:
    path = "https://raw.githubusercontent.com/datasets/covid-19/main/data/time-series-19-covid-combined.csv"
    df_orig = pd.read_csv(path)

df_orig.columns = df_orig.columns.str.strip()
df_orig['Country/Region'] = df_orig['Country/Region'].str.strip()
df_orig['Date'] = pd.to_datetime(df_orig['Date'], errors='coerce')

# -----------------------
# Sidebar - Country Selection
# -----------------------
countries = sorted(df_orig['Country/Region'].dropna().unique())
selected_country = st.sidebar.selectbox("Select Country", countries, index=countries.index("India") if "India" in countries else 0)

df_filtered = df_orig[df_orig['Country/Region'] == selected_country].copy()
df_filtered = df_filtered.groupby('Date').agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).reset_index()

if df_filtered.empty:
    st.error("No data available for the selected country.")
    st.stop()

# -----------------------
# Main Title
# -----------------------
st.title(f"📊 COVID-19 Regression Analysis Dashboard - {selected_country}")

# -----------------------
# Data Processing
# -----------------------
date_lst = df_filtered['Date'].dt.strftime('%Y-%m-%d').tolist()
date_dict = {date: df_filtered.loc[df_filtered['Date'].dt.strftime('%Y-%m-%d') == date] for date in date_lst}
date_tot_tup_dict = {date: (df_d['Confirmed'].sum(), df_d['Deaths'].sum(), df_d['Recovered'].sum()) for date, df_d in date_dict.items()}

df_date_tots = pd.DataFrame(date_tot_tup_dict).transpose()
df_date_tots.columns = ['Confirmed', 'Deaths', 'Recovered']
df_date_tots['Closed Cases'] = df_date_tots['Deaths'] + df_date_tots['Recovered']
df_date_tots['Active Cases'] = df_date_tots['Confirmed'] - df_date_tots['Closed Cases']

# -----------------------
# EDA
# -----------------------
st.subheader("📊 Exploratory Data Analysis")
col1, col2 = st.columns(2)
with col1:
    st.write("### Summary Statistics")
    st.write(df_date_tots.describe())

with col2:
    st.write("### Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_date_tots.corr(), annot=True, cmap='coolwarm', center=0, ax=ax_corr)
    st.pyplot(fig_corr)

# -----------------------
# Visualizations
# -----------------------
st.subheader("📈 Visualizations")
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df_date_tots['Confirmed'], label="Confirmed Cases")
ax1.plot(df_date_tots['Deaths'], label="Deaths")
ax1.plot(df_date_tots['Recovered'], label="Recovered")
ax1.legend()
ax1.set_title("COVID-19 Cases Over Time")
st.pyplot(fig1)

# -----------------------
# Modeling Config
# -----------------------
st.sidebar.header("⚙️ Model Configuration")
selected_models = st.sidebar.multiselect(
    "Select Models",
    ["Linear Regression", "Polynomial Regression", "Bayesian Ridge", "Polynomial Ridge Regression"],
    default=["Linear Regression", "Polynomial Regression"]
)
poly_degree = st.sidebar.slider("Polynomial Degree", 2, 8, 5)
alpha_value = st.sidebar.slider("Ridge Alpha", 0.1, 10.0, 1.0)

# -----------------------
# Train-Test Split
# -----------------------
df_date_tots["Days Since"] = range(len(date_lst))
X = np.array(df_date_tots["Days Since"]).reshape(-1, 1)
y = np.array(df_date_tots["Confirmed"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

models_results = {}
predictions = {}

if "Linear Regression" in selected_models:
    model = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    pred = model.predict(X)
    models_results["Linear Regression"] = {
        "MAE": mean_absolute_error(y_test, model.predict(X_test)),
        "MSE": mean_squared_error(y_test, model.predict(X_test))
    }
    predictions["Linear Regression"] = pred

if "Polynomial Regression" in selected_models:
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression(fit_intercept=False).fit(poly.fit_transform(X_train), y_train)
    pred = model.predict(X_poly)
    models_results["Polynomial Regression"] = {
        "MAE": mean_absolute_error(y_test, model.predict(poly.fit_transform(X_test))),
        "MSE": mean_squared_error(y_test, model.predict(poly.fit_transform(X_test)))
    }
    predictions["Polynomial Regression"] = pred

if "Bayesian Ridge" in selected_models:
    model = BayesianRidge(fit_intercept=False).fit(X_train, y_train)
    pred = model.predict(X)
    models_results["Bayesian Ridge"] = {
        "MAE": mean_absolute_error(y_test, model.predict(X_test)),
        "MSE": mean_squared_error(y_test, model.predict(X_test))
    }
    predictions["Bayesian Ridge"] = pred

if "Polynomial Ridge Regression" in selected_models:
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X)
    model = Ridge(alpha=alpha_value).fit(poly.fit_transform(X_train), y_train)
    pred = model.predict(X_poly)
    models_results["Polynomial Ridge Regression"] = {
        "MAE": mean_absolute_error(y_test, model.predict(poly.fit_transform(X_test))),
        "MSE": mean_squared_error(y_test, model.predict(poly.fit_transform(X_test)))
    }
    predictions["Polynomial Ridge Regression"] = pred

# -----------------------
# Results Display
# -----------------------
if models_results:
    st.subheader("📋 Model Performance")
    results_df = pd.DataFrame([{"Model": m, **metrics} for m, metrics in models_results.items()])
    st.dataframe(results_df)

# -----------------------
# Predictions Visualization
# -----------------------
if predictions:
    st.subheader("🔮 Predictions vs Actual")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y, label="Actual", color='black', linewidth=2)
    for name, pred in predictions.items():
        ax.plot(pred, label=name, linestyle="--")
    ax.legend()
    st.pyplot(fig)

# -----------------------
# Download Button
# -----------------------
if predictions:
    export_df = pd.DataFrame({"Day": range(len(y)), "Actual": y, **{name: pred for name, pred in predictions.items()}})
    csv = export_df.to_csv(index=False)
    st.download_button("Download Predictions CSV", csv, "covid_predictions.csv", "text/csv")
