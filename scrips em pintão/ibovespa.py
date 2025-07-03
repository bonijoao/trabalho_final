import pandas as pd

# URL direta para o arquivo CSV de dados diários do IBOVESPA
url_ibovespa = 'https://stooq.com/q/d/l/?s=^bvp&i=d'

try:
    # Tenta ler os dados diretamente da URL
    df_ibovespa = pd.read_csv(url_ibovespa)

    # Converte a coluna 'Date' para o formato de data
    df_ibovespa['Date'] = pd.to_datetime(df_ibovespa['Date'])

    # Filtra os dados apenas para o ano de 2025
    df_ibovespa_2025 = df_ibovespa[df_ibovespa['Date'].dt.year == 2025]

    # Define a data como o índice do DataFrame
    df_ibovespa_2025.set_index('Date', inplace=True)

    # Ordena os dados pela data, pois eles vêm em ordem decrescente
    df_ibovespa_2025.sort_index(inplace=True)

    # Exibe as primeiras 5 linhas dos dados
    print(df_ibovespa_2025.head())

    # o tamanho do dataframe
    print(df_ibovespa_2025.shape)

except Exception as e:
    print(f"Ocorreu um erro ao baixar ou processar os dados: {e}")