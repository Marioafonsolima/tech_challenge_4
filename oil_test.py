import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import *
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import streamlit as st

def oil_price_fetch():
    """
    Function to fetch and standardize oil price data from the IPEA website.

    Returns:
        DataFrame: DataFrame with standardized date and price columns.
    """

    dados = pd.read_html("http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view", skiprows=1, decimal=",", thousands=".")[0]
    dados = dados.rename(columns={0: "data", 1: "preco"})
    dados["data"] = pd.to_datetime(dados["data"], format='%d/%m/%Y')
    dados = dados.set_index("data")
    dados["preco"] = dados["preco"].astype(float)

    return dados

def oil_price_analysis(data):
    """
    Function to analyze and visualize oil price data.

    Arguments:
        data (DataFrame): DataFrame with oil price data.

    Returns:
        None: Displays a descriptive table and line plot of the data.
    """

    # Descriptive statistics
    print(data.describe())

    # Line plot of oil prices
    plt.figure(figsize=(10, 6))
    data["preco"].plot()
    plt.title("Preços do Petróleo")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.grid(True)
    plt.show()

def avaliar_previsao(df, a, b, c, metric='mape'):
    """
    Function to evaluate time series forecasting using Prophet and cross-validation.

    Arguments:
        df (DataFrame): DataFrame with time series data.
        a (str): Initial cutoff date (YYYY-MM-DD).
        b (str): Intermediate cutoff date (YYYY-MM-DD).
        c (str): Final cutoff date (YYYY-MM-DD).
        metric (str): Performance metric to use (default: 'mape').

    Returns:
        DataFrame: Cross-validation results with the specified metric.
        Figure: Cross-validation metric plot over time.
    """

    # Data verification and preparation
    if not isinstance(df, pd.DataFrame):
        raise TypeError('The `df` argument must be a DataFrame.')

    df.columns = df.columns.str.lower()  # Convert column names to lowercase
    if not ('ds' in df.columns and 'y' in df.columns):
        raise ValueError('The DataFrame must contain columns named "ds" and "y".')

    # Split data based on cutoffs
    cutoffs = pd.to_datetime([a, b, c])
    df_train = df[df['ds'] < cutoffs[0]]
    df_val1 = df[(df['ds'] >= cutoffs[0]) & (df['ds'] < cutoffs[1])]
    df_val2 = df[(df['ds'] >= cutoffs[1]) & (df['ds'] < cutoffs[2])]
    df_test = df[df['ds'] >= cutoffs[2]]

    # Train and predict using Prophet model
    prophet_model = Prophet()
    prophet_model.fit(df_train)

    future = prophet_model.make_future_dataframe(periods=365, include_history=True)
    forecast = prophet_model.predict(future)

    # Calculate performance metrics
    df_cv_train = cross_validation(prophet_model, initial='5717', period='1430 days', horizon='365 days')
    df_cv_val1 = cross_validation(prophet_model, cutoffs=[cutoffs[0]], horizon='365 days')
    df_cv_val2 = cross_validation(prophet_model, cutoffs=[cutoffs[1]], horizon='365 days')

    df_cv = pd.concat([df_cv_train, df_cv_val1, df_cv_val2], ignore_index=True)
    df_metrics = performance_metrics(df_cv, metric=metric)

    # Visualize the forecast and cross-validation metric
    fig1 = prophet_model.plot(forecast)
    fig2 = plot_cross_validation_metric(df_cv, metric=metric)

    return df_metrics, fig1, fig2

# Streamlit application
st.title("Previsão de Preços do Petróleo")

# Fetch and display data analysis
oil_data = oil_price_fetch()
st.header("Análise Exploratória")
with st.expander("Ver Dados Brutos"):
    st.write(oil_data)
oil_price_analysis(oil_data.copy())  # Avoid modifying original data

# User interface for date cutoffs
st.header("Defina Cortes para Previsão")
cutoff_a = st.date_input("Data de Corte Inicial (AAAA-MM-DD)", value=oil_data.index.min().strftime('%Y-%m-%d'))
cutoff_b = st.date_input("Data de Corte Intermediária (AAAA-MM-DD)", value=(oil_data.index.min() + pd.DateOffset(months=6)).strftime('%Y-%m-%d'))
cutoff_c = st.date_input("Data de Corte Final (AAAA-MM-DD)", value=oil_data.index.max().strftime('%Y-%m-%d'))

# Perform evaluation and display results
if st.button("Prever Preços"):
    try:
        # Ensure cutoffs are in ascending order
        if cutoff_a > cutoff_b or cutoff_b > cutoff_c:
            st.error("As datas de corte devem estar em ordem crescente.")
            raise ValueError("Invalid cutoff dates")

        df_metrics, fig1, fig2 = avaliar_previsao(oil_data.copy(), cutoff_a, cutoff_b, cutoff_c)

        st.subheader("Resultados da Previsão")
        st.write(df_metrics)

        st.subheader("Visualização da Previsão")
        st.plotly_chart(fig1)

        st.subheader("Métricas de Validação Cruzada")
        st.plotly_chart(fig2)

    except ValueError as e:
        st.error(f"Erro: {e}")

# Offer option to download results as CSV
if st.button("Baixar Resultados (CSV)"):
    if df_metrics is not None:
        csv_content = df_metrics.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_content,
            file_name="previsao_petroleo.csv",
            mime="text/csv",
        )
    else:
        st.info("Nenhum resultado disponível para download.")



