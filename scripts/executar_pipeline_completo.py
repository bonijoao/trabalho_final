import subprocess
import sys
import os

def executar_pipeline(janelas_temporais=[15, 30, 60], tamanho_buffer=5):
    print("="*60)
    print("PIPELINE COMPLETO - ANÁLISE BITCOIN COM FILAS E PILHA")
    print("="*60)
    print(f"Configuração:")
    print(f"   Janelas Temporais: {janelas_temporais} min")
    print(f"   Buffer da Pilha: {tamanho_buffer} min")
    
    # Calcular número de amostras para cada janela
    for janela in janelas_temporais:
        num_amostras = janela // tamanho_buffer
        sequencia = [f't-{janela-i*tamanho_buffer}' for i in range(num_amostras)]
        print(f"   Janela {janela}min: {num_amostras} amostras {sequencia} → t-0")
    
    scripts = [
        ("1_carregar_dados_iniciais.py", "Processando dados com estrutura de fila e pilha..."),
        ("3_treinar_e_avaliar_modelo.py", "Treinando e avaliando modelos...")
    ]
    
    for script, descricao in scripts:
        print(f"\n{descricao}")
        print("-" * len(descricao))
        
        try:
            # Para o primeiro script, usa exec com replace
            if script == "1_carregar_dados_iniciais.py":
                cmd = [sys.executable, "-c", 
                      f"from estrutura_fila_bitcoin import *; exec(open('{script}').read().replace('processar_dados_bitcoin()', 'processar_dados_bitcoin(janelas_temporais={janelas_temporais}, tamanho_buffer={tamanho_buffer})'))"
                ]
            else:
                cmd = [sys.executable, script]
                
            resultado = subprocess.run(cmd, 
                                     capture_output=True, 
                                     text=True, 
                                     check=True)
            print(resultado.stdout)
            
            if resultado.stderr:
                print("Avisos:", resultado.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar {script}:")
            print(e.stdout)
            print(e.stderr)
            return False
        except FileNotFoundError:
            print(f"Script {script} não encontrado!")
            return False
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTADO COM SUCESSO!")
    print(f"Dados processados com semântica temporal correta:")
    for janela in janelas_temporais:
        num_amostras = janela // tamanho_buffer
        print(f"   {janela}min → {num_amostras} amostras espaçadas de {tamanho_buffer}min")
    print("="*60)
    return True

if __name__ == "__main__":
    # Configuração com semântica temporal correta
    executar_pipeline(janelas_temporais=[15, 30, 60], tamanho_buffer=5) 