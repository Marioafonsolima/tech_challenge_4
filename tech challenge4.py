import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import *
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric



def oil_price_fetch():
    """
    Função para atualizar a base de dados de preço do petróleo, através do site do IPEA, e padronizar para uso futuro.

    Retornos:
        DataFrame: dataframe com data e preço já padronizados.
    """

    dados = pd.read_html("http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view", skiprows=1, decimal=",", thousands=".")[0]
    dados = dados.rename(columns={0: "data", 1: "preco"})
    dados["data"] = pd.to_datetime(dados["data"], format='%d/%m/%Y')
    dados = dados.set_index("data")
    dados["preco"] = dados["preco"].astype(float)

    return dados


def oil_price_analysis(data):
    """
    Função para analisar e visualizar dados de preços do petróleo.

    Argumentos:
        data (DataFrame): DataFrame com dados de preços do petróleo.

    Retornos:
        None: Visualiza a tabela com o describe e o line plot dos dados.
    """

    # Descrição dos dados
    print(data.describe())

    # Line plot dos preços do petróleo
    plt.figure(figsize=(10, 6))
    data["preco"].plot()
    plt.title("Preços do Petróleo")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.grid(True)
    plt.show()


def evaluate_forecast(df, a, b, c, metric='mape'):
    
    """
    Função para avaliar a previsão de séries temporais usando Prophet e validação cruzada.

    Argumentos:
        df (DataFrame): O DataFrame com os dados da série temporal.
        a (str): Data de corte inicial (formato YYYY-MM-DD).
        b (str): Data de corte intermediária (formato YYYY-MM-DD).
        c (str): Data de corte final (formato YYYY-MM-DD).
        metric (str): Métrica de desempenho a ser utilizada (padrão: 'mape').

    Retornos:
        DataFrame: Resultados da validação cruzada com a métrica especificada.
        Figura: Gráfico da métrica de validação cruzada ao longo do tempo.
    """

    # Verificação e preparação dos dados
    if not isinstance(df, pd.DataFrame):
        raise TypeError('O argumento `df` deve ser um DataFrame.')

    df.columns = df.columns.str.lower()  # Converter nomes de colunas para minúsculas
    if not ('ds' in df.columns and 'y' in df.columns):
        raise ValueError('O DataFrame deve conter colunas com nomes "ds" e "y".')

    # Divisão dos dados por cortes
    cutoffs = pd.to_datetime([a, b, c])
    df_train = df[df['ds'] < cutoffs[0]]
    df_val1 = df[(df['ds'] >= cutoffs[0]) & (df['ds'] < cutoffs[1])]
    df_val2 = df[(df['ds'] >= cutoffs[1]) & (df['ds'] < cutoffs[2])]
    df_test = df[df['ds'] >= cutoffs[2]]

    # Treinamento e previsão do modelo Prophet
    prophet_model = Prophet()
    prophet_model.fit(df_train)

    future = prophet_model.make_future_dataframe(periods=365, include_history=True)
    forecast = prophet_model.predict(future)

    # Cálculo das métricas de desempenho
    df_cv_train = cross_validation(prophet_model, initial='5717', period='1430 days', horizon='365 days')
    df_cv_val1 = cross_validation(prophet_model, cutoffs=[cutoffs[0]], horizon='365 days')
    df_cv_val2 = cross_validation(prophet_model, cutoffs=[cutoffs[1]], horizon='365 days')

    df_cv = pd.concat([df_cv_train, df_cv_val1, df_cv_val2], ignore_index=True)
    df_metrics = performance_metrics(df_cv, metric=metric)

    # Visualização dos resultados
    fig1 = prophet_model.plot(forecast)
    fig2 = plot_cross_validation_metric(df_cv, metric=metric)

    return df_metrics, fig1, fig2



