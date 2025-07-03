import pandas as pd
import os
from estrutura_fila_bitcoin import FilaDeslizante, PilhaBuffer

def processar_dados_bitcoin(janelas_temporais=[15, 30, 60], tamanho_buffer=5):
    print(f"🚀 Iniciando processamento com buffer de pilha = {tamanho_buffer} min")
    
    df = pd.read_csv('dados/db_bitcoin_1dia.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    cotacoes = df['Close'].values
    print(f"📊 Carregadas {len(cotacoes)} cotações")
    
    for janela_temporal in janelas_temporais:
        # Calcular número de amostras baseado na semântica correta
        num_amostras = janela_temporal // tamanho_buffer
        
        if num_amostras < 2:
            print(f"⚠️  Janela {janela_temporal} com buffer {tamanho_buffer}: precisa de pelo menos 2 amostras!")
            continue
            
        print(f"\n📏 Processando janela temporal de {janela_temporal} min:")
        print(f"   Buffer: {tamanho_buffer} min → {num_amostras} amostras")
        print(f"   Sequência: {[f't-{janela_temporal-i*tamanho_buffer}' for i in range(num_amostras)]} → t-0")
        
        fila = FilaDeslizante(num_amostras)
        pilha_buffer = PilhaBuffer(tamanho_buffer)
        dados_formatados = []
        
        for i, cotacao in enumerate(cotacoes):
            # Empilha a cotação no buffer
            cotacao_amostrada = pilha_buffer.empilhar(cotacao)
            
            # Se a pilha retornou um valor, processa na fila
            if cotacao_amostrada is not None:
                if not fila.esta_cheia():
                    fila.enfileirar(cotacao_amostrada)
                else:
                    elementos_atuais = fila.get_elementos()
                    alvo = cotacao_amostrada
                    
                    linha = elementos_atuais + [alvo]
                    dados_formatados.append(linha)
                    
                    fila.enfileirar(cotacao_amostrada)
        
        # Criar colunas com a semântica temporal correta
        colunas = [f't-{janela_temporal-i*tamanho_buffer}' for i in range(num_amostras)] + ['t-0']
        df_resultado = pd.DataFrame(dados_formatados, columns=colunas)
        
        os.makedirs('../dados', exist_ok=True)
        arquivo_saida = f'../dados/bitcoin_janela_{janela_temporal}min_buffer_{tamanho_buffer}min.csv'
        df_resultado.to_csv(arquivo_saida, index=False)
        
        print(f"✅ Dados processados: {len(df_resultado)} amostras salvas em {arquivo_saida}")
        print(f"   Espaço temporal: {janela_temporal} min com {num_amostras} amostras espaçadas de {tamanho_buffer} min")

if __name__ == "__main__":
    processar_dados_bitcoin() 