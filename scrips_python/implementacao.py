import requests
import pandas as pd
from datetime import datetime

def obter_dados_bitcoin_1min(api_key: str):
    """
    Busca os dados de preço do Bitcoin (BTCUSD) de minuto a minuto na API da Financial Modeling Prep.

    Args:
        api_key (str): Sua chave de API pessoal da Financial Modeling Prep.

    Returns:
        pandas.DataFrame: Um DataFrame com os dados de 'Open', 'High', 'Low', 'Close', 'Volume',
                          ou None se ocorrer um erro.
    """
    # O símbolo para Bitcoin vs Dólar Americano
    simbolo = 'BTCUSD'
    
    # Monta a URL da API
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{simbolo}"
    
    # Parâmetros da requisição, incluindo a sua chave de API
    params = {
        'apikey': api_key
    }
    
    print("Buscando dados na API da Financial Modeling Prep...")
    
    try:
        # Faz a requisição para a API
        response = requests.get(url, params=params)
        
        # Verifica se a requisição foi bem-sucedida (código 200)
        response.raise_for_status() 
        
        # Converte a resposta JSON em uma lista de dicionários
        dados = response.json()
        
        if not dados:
            print("A API não retornou dados. Verifique o símbolo ou sua chave de API.")
            return None
            
        # Converte a lista de dados para um DataFrame do pandas
        df = pd.DataFrame(dados)
        
        # Renomeia as colunas para um padrão mais comum
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Converte a coluna 'Date' para o formato de data e hora
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Define a data como o índice do DataFrame e ordena do mais antigo para o mais novo
        df = df.set_index('Date').sort_index()
        
        print("Dados obtidos e processados com sucesso!")
        return df

    except requests.exceptions.HTTPError as http_err:
        print(f"Erro HTTP: {http_err}")
        print(f"Detalhes da resposta: {response.text}")
    except requests.exceptions.RequestException as err:
        print(f"Erro na requisição: {err}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        
    return None

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # IMPORTANTE: Substitua 'SUA_CHAVE_DE_API_AQUI' pela chave que você recebeu por e-mail.
    MINHA_API_KEY = 'YdrDvLrNzDWUBAqo1NTQUsZW7vOZCsI3' 

    if MINHA_API_KEY == 'SUA_CHAVE_DE_API_AQUI':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!  ATENÇÃO: Você precisa substituir 'SUA_CHAVE_DE_API_AQUI'  !!!")
        print("!!!  pela sua chave de API no código para funcionar.         !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # Chama a função para obter os dados
        dados_btc = obter_dados_bitcoin_1min(MINHA_API_KEY)

        # Se os dados foram obtidos, mostra as 5 linhas mais recentes
        if dados_btc is not None:
            print("\nÚltimos 5 minutos de dados do Bitcoin (BTCUSD):")
            # O .tail(5) mostra os últimos 5 registros
            print(dados_btc.tail(500))