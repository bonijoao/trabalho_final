import requests
import pandas as pd
import os

# --- CONFIGURAÇÃO ---
MINHA_API_KEY = 'YdrDvLrNzDWUBAqo1NTQUsZW7vOZCsI3'
NOME_ARQUIVO_DB = 'dados/db_bitcoin_1dia.csv'
SIMBOLO = 'BTCUSD'
TAMANHO_FILA = 1440

def atualizar_fila_de_dados(api_key: str):
    """
    Lê a base de dados existente, busca novos dados, atualiza a fila
    e a salva de volta no arquivo CSV, mantendo o tamanho fixo.
    """
    if api_key == 'SUA_CHAVE_DE_API_AQUI':
        print("!!! ATENÇÃO: Substitua 'SUA_CHAVE_DE_API_AQUI' pela sua chave de API. !!!")
        return

    # 1. Verifica se a base de dados existe. Se não, avisa para rodar o script inicial.
    if not os.path.exists(NOME_ARQUIVO_DB):
        print(f"Erro: O arquivo '{NOME_ARQUIVO_DB}' não foi encontrado.")
        print("Por favor, execute o script '1_carregar_dados_iniciais.py' primeiro.")
        return

    # 2. Carrega os dados existentes do CSV
    print("Carregando dados existentes...")
    df_antigo = pd.read_csv(NOME_ARQUIVO_DB, index_col='Date', parse_dates=True)

    # 3. Busca os dados mais recentes na API
    print("Buscando novos dados na API...")
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{SIMBOLO}"
    params = {'apikey': api_key}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        dados_novos = response.json()

        if not dados_novos:
            print("Nenhum dado novo retornado pela API nesta atualização.")
            return

        df_novo = pd.DataFrame(dados_novos)
        df_novo = df_novo.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df_novo['Date'] = pd.to_datetime(df_novo['Date'])
        df_novo = df_novo.set_index('Date')

        # 4. Combina os dados antigos e novos
        df_combinado = pd.concat([df_antigo, df_novo])
        
        # 5. Remove duplicados (baseado no índice 'Date'), mantendo o último registro
        df_combinado = df_combinado[~df_combinado.index.duplicated(keep='last')]
        
        # 6. Ordena e garante o tamanho da fila (pega os últimos 1440)
        df_final = df_combinado.sort_index().tail(TAMANHO_FILA)
        
        # 7. Salva (sobrescreve) o arquivo com a fila atualizada
        df_final.to_csv(NOME_ARQUIVO_DB)
        
        print("-" * 50)
        print("Fila de dados atualizada com sucesso!")
        print(f"Total de registros agora: {len(df_final)}")
        print("Dados mais antigos na fila:")
        print(df_final.head(1))
        print("\nDados mais recentes na fila:")
        print(df_final.tail(1))
        print("-" * 50)

    except Exception as e:
        print(f"Ocorreu um erro durante a atualização: {e}")

# --- Execução do Script ---
if __name__ == "__main__":
    atualizar_fila_de_dados(MINHA_API_KEY)