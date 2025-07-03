import requests
import pandas as pd
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO ---
# IMPORTANTE: Substitua pela sua chave de API real.
MINHA_API_KEY = 'YdrDvLrNzDWUBAqo1NTQUsZW7vOZCsI3' 
NOME_ARQUIVO_DB = 'dados/db_bitcoin_1dia.csv'
SIMBOLO = 'BTCUSD'
# Tamanho da nossa fila: 24 horas * 60 minutos * 2 dias = 2880 registros  
TAMANHO_FILA = 2880

def carregar_base_inicial(api_key: str):
    """
    Busca um grande volume de dados históricos (últimas 48h) e salva os 1440
    registros mais recentes para inicializar nosso banco de dados.
    """
    if api_key == 'SUA_CHAVE_DE_API_AQUI':
        print("!!! ATENÇÃO: Substitua 'SUA_CHAVE_DE_API_AQUI' pela sua chave de API. !!!")
        return

    # Calcula as datas para buscar um período maior (ex: últimas 48 horas)
    # para garantir que teremos dados suficientes.
    data_final = datetime.now()
    data_inicial = data_final - timedelta(days=2) # Busca 2 dias para ter margem

    url = (f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{SIMBOLO}"
           f"?from={data_inicial.strftime('%Y-%m-%d')}"
           f"&to={data_final.strftime('%Y-%m-%d')}")
    
    params = {'apikey': api_key}
    
    print(f"Buscando dados históricos de {SIMBOLO} para carga inicial...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        dados = response.json()
        
        if not dados:
            print("A API não retornou dados para a carga inicial.")
            return

        df = pd.DataFrame(dados)
        df = df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Pega apenas os últimos 1440 registros para formar nossa base inicial
        df_final = df.tail(TAMANHO_FILA)
        
        # Salva a base inicial no arquivo CSV
        df_final.to_csv(NOME_ARQUIVO_DB)
        
        print("-" * 50)
        print(f"Base de dados inicial '{NOME_ARQUIVO_DB}' criada com sucesso!")
        print(f"Total de registros: {len(df_final)}")
        print("Primeiro registro:")
        print(df_final.head(1))
        print("\nÚltimo registro:")
        print(df_final.tail(1))
        print("-" * 50)

    except Exception as e:
        print(f"Ocorreu um erro na carga inicial: {e}")

# --- Execução do Script ---
if __name__ == "__main__":
    carregar_base_inicial(MINHA_API_KEY)