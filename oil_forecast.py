import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


def oil_price_fetch():
    """
    Function to fetch and standardize oil price data from the IPEA website.

    Returns:
        DataFrame: DataFrame with standardized date and price columns.
    """
    dados = pd.read_html("http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view", skiprows=1,
                         decimal=",", thousands=".")[0]
    dados = dados.rename(columns={0: "data", 1: "preco"})
    dados["data"] = pd.to_datetime(dados["data"], format='%d/%m/%Y')
    dados["preco"] = dados["preco"].astype(float)
    # Filter data for the last 4500 days
    dados = dados.head(4500)
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
    st.write("Descriptive Statistics:")
    st.write(data.describe())

    # Line plot with Altair
    st.write("Oil Price Analysis:")
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('data:T', title='Date'),
        y=alt.Y('preco:Q', title='Oil Price'),
        tooltip=[alt.Tooltip('data:T', title='Date'), alt.Tooltip('preco:Q', title='Price')]
    ).properties(
        width=800,
        height=400,
        title='Oil Prices'
    ).interactive()
    st.altair_chart(chart)


def prepare_data_prophet(df: pd.DataFrame, ds: str, y: str) -> pd.DataFrame:
    """
    Function to prepare a DataFrame for use with the Prophet model.

    Arguments:
        df (DataFrame): The DataFrame with time series data.
        ds (str): Name of the column containing the timestamp of the time series.
        y (str): Name of the column containing the value to be predicted (target time series).

    Returns:
        DataFrame: The DataFrame prepared for the Prophet model.

    Raises:
        ValueError: If the DataFrame is not valid or the ds and y columns do not exist or are in the incorrect format.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError('The `df` argument must be a DataFrame.')

    if ds not in df.columns or y not in df.columns:
        raise ValueError(f'The columns "{ds}" and "{y}" do not exist in the DataFrame.')

    # Check format of the ds column
    if df[ds].dtype != np.datetime64:
        try:
            df[ds] = pd.to_datetime(df[ds])
        except Exception as e:
            raise ValueError(f'The column "{ds}" is not in datetime format. Error: {e}')

    # Check for missing values
    if df[ds].isna().sum() > 0:
        raise ValueError(f'The column "{ds}" contains missing values.')

    if df[y].isna().sum() > 0:
        raise ValueError(f'The column "{y}" contains missing values.')

    # Rename columns to Prophet standards
    ready_df = df.rename(columns={ds: "ds", y: "y"})

    return ready_df


def train_and_predict_prophet(df, period, a=None, b=None, c=None, confidence_intervals=False):
    """
    Function to train and predict time series using Prophet.

    Arguments:
        df (DataFrame): The DataFrame with time series data.
        period (int): Analysis period for future prediction.
        a (str): Initial cutoff date (format YYYY-MM-DD) (optional).
        b (str): Intermediate cutoff date (format YYYY-MM-DD) (optional).
        c (str): Final cutoff date (format YYYY-MM-DD) (optional).
        confidence_intervals (bool): If True, calculate confidence intervals (optional).

    Returns:
        DataFrame: Cross-validation results with various metrics.
        Figure: Plot of model predictions.
        List[Figure]: List of plots for each cross-validation metric.
    """
    # Check and prepare data
    if not isinstance(df, pd.DataFrame):
        raise TypeError('The `df` argument must be a DataFrame.')

    df.columns = df.columns.str.lower()  # Convert column names to lowercase
    if not ('ds' in df.columns and 'y' in df.columns):
        raise ValueError('The DataFrame must contain columns named "ds" and "y".')

    # Sort data by date
    df = df.sort_values(by='ds')

    # Set cutoff dates automatically if not provided
    if not a or not b or not c:
        total_length = len(df)
        cut1_idx = total_length // 3
        cut2_idx = 2 * total_length // 3

        a = df.iloc[cut1_idx]['ds']
        b = df.iloc[cut2_idx]['ds']
        c = df.iloc[-1]['ds']

    # Data split by cutoffs
    cutoffs = pd.to_datetime([a, b, c])
    df_train = df[df['ds'] < cutoffs[0]]

    # Prophet model training
    prophet_model = Prophet()
    prophet_model.fit(df_train)

    # Forecast for the specified period
    future = prophet_model.make_future_dataframe(periods=period, include_history=True)
    forecast = prophet_model.predict(future)

    # Calculation of confidence intervals (if requested)
    if confidence_intervals:
        forecast = forecast.set_dataframe(df_train.copy())  # Copy the original DataFrame
        forecast = prophet_model.predict(future)  # Predict with confidence intervals
        forecast = prophet_model.make_dataframe(forecast)  # Extract the DataFrame

    # Calculation of performance metrics using cross-validation
    try:
        df_cv = cross_validation(prophet_model, initial="30 days", period="30 days", horizon="30 days")
        df_metrics = performance_metrics(df_cv)
    except ValueError:
        raise ValueError('Less data than horizon after initial window. Make horizon or initial shorter.')

    # Generate plots with Altair
    fig1 = alt.Chart(forecast).mark_line().encode(
        x='ds:T',
        y='yhat:Q',
        tooltip=['ds:T', 'yhat:Q']
    ).properties(
        title='Prophet Model Predictions'
    )

    # Generate subplots for all metrics
    figs = []
    if not confidence_intervals:
        df_metrics = None  # Set metrics to None if confidence intervals are not calculated
    else:
        metrics = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'coverage']
        for metric in metrics:
            fig = alt.Chart(df_metrics).mark_line().encode(
                x='horizon',
                y=metric,
                tooltip=['horizon', metric]
            ).properties(
                title=f'Cross-validation Metric: {metric.upper()}'
            )
            figs.append(fig)

    return df_metrics, fig1, figs


# Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("Oil Price Forecasting App")

    # Fetch and analyze oil price data
    data = oil_price_fetch()
    oil_price_analysis(data)

    # Prepare data for Prophet
    ready_df = prepare_data_prophet(data, 'data', 'preco')

    # Train and predict with Prophet
    period = st.slider("Select period for forecast (days):", 30, 365, 90, 30)
    df_metrics, fig1, figs = train_and_predict_prophet(ready_df, period)

    # Display KPIs
    st.header("KPIs")
    st.subheader("Oil Price Statistics")
    st.metric(label="Minimum", value=data['preco'].min())
    st.metric(label="Maximum", value=data['preco'].max())
    st.metric(label="Mean", value=data['preco'].mean())
    st.metric(label="Standard Deviation", value=data['preco'].std())

    st.subheader("Prophet Forecast Statistics")
    if df_metrics is not None:
        st.metric(label="Minimum", value=df_metrics['yhat_lower'].min())
        st.metric(label="Maximum", value=df_metrics['yhat_upper'].max())
        st.metric(label="Mean", value=df_metrics['yhat'].mean())
        st.metric(label="Standard Deviation", value=df_metrics['yhat'].std())
    else:
        st.write("Confidence intervals were not calculated. No forecast statistics available.")

    # Display Prophet forecast plots
    st.header("Prophet Forecast")
    st.subheader("Model Predictions")
    st.altair_chart(fig1, use_container_width=True)

    st.subheader("Cross-validation Metrics")
    for fig in figs:
        st.altair_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()