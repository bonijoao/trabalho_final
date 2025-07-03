import pandas as pd
import os
from estrutura_fila_bitcoin import FilaDeslizante

def processar_dados_bitcoin(tamanhos_janela=[15, 30, 60]):
    df = pd.read_csv('dados/db_bitcoin_1dia.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    cotacoes = df['Close'].values
    
    for tamanho in tamanhos_janela:
        fila = FilaDeslizante(tamanho)
        dados_formatados = []
        
        for i, cotacao in enumerate(cotacoes):
            if not fila.esta_cheia():
                fila.enfileirar(cotacao)
            else:
                elementos_atuais = fila.get_elementos()
                alvo = cotacao
                
                linha = elementos_atuais + [alvo]
                dados_formatados.append(linha)
                
                fila.enfileirar(cotacao)
        
        colunas = [f't-{tamanho-1-i}' for i in range(tamanho)] + ['alvo']
        df_resultado = pd.DataFrame(dados_formatados, columns=colunas)
        
        os.makedirs('dados', exist_ok=True)
        arquivo_saida = f'dados/bitcoin_janela_{tamanho}min.csv'
        df_resultado.to_csv(arquivo_saida, index=False)
        
        print(f"Dados processados para janela {tamanho}: {len(df_resultado)} amostras salvas em {arquivo_saida}")

if __name__ == "__main__":
    processar_dados_bitcoin() 