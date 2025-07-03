import subprocess
import sys
import os

def executar_pipeline():
    print("="*60)
    print("PIPELINE COMPLETO - ANÁLISE BITCOIN COM FILAS")
    print("="*60)
    
    scripts = [
        ("1_carregar_dados_iniciais.py", "Processando dados com estrutura de fila..."),
        ("3_treinar_e_avaliar_modelo.py", "Treinando e avaliando modelos...")
    ]
    
    for script, descricao in scripts:
        print(f"\n{descricao}")
        print("-" * len(descricao))
        
        try:
            resultado = subprocess.run([sys.executable, script], 
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
    print("="*60)
    return True

if __name__ == "__main__":
    executar_pipeline() 