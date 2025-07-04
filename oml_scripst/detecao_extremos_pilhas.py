"""
Detecção de Máximos e Mínimos com Pilhas para Online Machine Learning
=====================================================================

Este módulo implementa detecção de extremos (picos e vales) 
usando estruturas de pilha (LIFO) para análise de dados de Bitcoin.

Autor: João Paulo Assis Bonifácio
Disciplina: Estrutura de Dados (GES-115)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')


class PilhaExtremos:
    """
    Implementa detecção de extremos usando pilha (LIFO) - versão simplificada baseada no app.py
    """
    
    def __init__(self):
        self.picos = []  # Pilha de máximos locais
        self.vales = []  # Pilha de mínimos locais


def detectar_extremos(dados, janela=5):
    """
    Detecta picos e vales usando uma janela móvel simples.
    Algoritmo baseado no app.py - simples e eficaz!
    
    Args:
        dados: Array de preços
        janela: Tamanho da janela para análise local
    
    Returns:
        tuple: (pilha_extremos, extremos_detectados)
    """
    pilha = PilhaExtremos()
    extremos_detectados = []
    
    for i in range(janela, len(dados) - janela):
        valor_atual = dados[i]
        janela_dados = dados[i-janela:i+janela+1]
        
        # É um pico se for o maior na janela
        if valor_atual == max(janela_dados):
            pilha.picos.append((i, valor_atual))
            extremos_detectados.append((i, valor_atual, "pico"))
        
        # É um vale se for o menor na janela  
        elif valor_atual == min(janela_dados):
            pilha.vales.append((i, valor_atual))
            extremos_detectados.append((i, valor_atual, "vale"))
    
    return pilha, extremos_detectados


class AnalisadorExtremosBitcoin:
    """
    Analisador simples para detecção de extremos em dados de Bitcoin.
    """
    
    def __init__(self, caminho_dados: str = 'dados/db_bitcoin_1dia.csv'):
        """
        Inicializa o analisador.
        
        Args:
            caminho_dados (str): Caminho para o arquivo de dados
        """
        self.caminho_dados = caminho_dados
        self.dados = None
        self.extremos = []
        self.pilha_extremos = None
        
    def carregar_dados(self) -> bool:
        """Carrega os dados de Bitcoin."""
        try:
            self.dados = pd.read_csv(self.caminho_dados)
            self.dados['Date'] = pd.to_datetime(self.dados['Date'])
            self.dados = self.dados.sort_values('Date').reset_index(drop=True)
            return True
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False
    
    def detectar_extremos(self, janela: int = 5) -> None:
        """
        Detecta extremos usando pilhas - algoritmo simples e eficaz do app.py.
        
        Args:
            janela (int): Tamanho da janela para análise local
        """
        if self.dados is None:
            print("Dados não carregados!")
            return
        
        precos = self.dados['Close'].values
        self.pilha_extremos, self.extremos = detectar_extremos(precos, janela)
        
        print(f"✅ Detecção concluída:")
        print(f"   • Máximos detectados: {len(self.pilha_extremos.picos)}")
        print(f"   • Mínimos detectados: {len(self.pilha_extremos.vales)}")
    
    def visualizar_extremos(self) -> None:
        """
        Visualiza apenas a série completa com máximos e mínimos identificados.
        """
        if self.dados is None or not self.extremos:
            print("Dados ou extremos não disponíveis!")
            return
        
        # Criar figura simples
        plt.figure(figsize=(15, 8))
        
        # Plotar série completa
        plt.plot(self.dados['Date'], self.dados['Close'], 
                color='blue', linewidth=1, label='Preço Bitcoin', alpha=0.8)
        
        # Plotar máximos e mínimos
        for idx, valor, tipo in self.extremos:
            timestamp = self.dados.iloc[idx]['Date']
            if tipo == "pico":
                plt.scatter(timestamp, valor, color='red', s=60, marker='^', 
                           zorder=5, edgecolors='black', linewidth=0.5)
            else:
                plt.scatter(timestamp, valor, color='green', s=60, marker='v', 
                           zorder=5, edgecolors='black', linewidth=0.5)
        
        # Adicionar pontos de legenda
        plt.scatter([], [], color='red', s=60, marker='^', 
                   label=f'Máximos ({len(self.pilha_extremos.picos)})', 
                   edgecolors='black', linewidth=0.5)
        plt.scatter([], [], color='green', s=60, marker='v', 
                   label=f'Mínimos ({len(self.pilha_extremos.vales)})', 
                   edgecolors='black', linewidth=0.5)
        
        plt.title('Detecção de Máximos e Mínimos com Pilhas (LIFO) - Bitcoin USD', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Data')
        plt.ylabel('Preço (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def relatorio_extremos(self) -> dict:
        """Gera relatório simples da detecção de extremos."""
        if not self.pilha_extremos:
            return {}
        
        picos_valores = [p[1] for p in self.pilha_extremos.picos]
        vales_valores = [v[1] for v in self.pilha_extremos.vales]
        
        return {
            'total_dados': len(self.dados),
            'total_picos': len(picos_valores),
            'total_vales': len(vales_valores),
            'maior_pico': max(picos_valores) if picos_valores else 0,
            'menor_vale': min(vales_valores) if vales_valores else 0,
            'amplitude_extremos': max(picos_valores) - min(vales_valores) if picos_valores and vales_valores else 0
        }


def main():
    """Função principal de demonstração."""
    print("🚀 Detecção de Extremos em Bitcoin com Pilhas (LIFO)")
    print("=" * 60)
    
    # Inicializar analisador
    analisador = AnalisadorExtremosBitcoin()
    
    # Carregar dados
    if not analisador.carregar_dados():
        print("❌ Falha ao carregar dados!")
        return
    
    print(f"✅ Dados carregados: {len(analisador.dados)} registros")
    
    # Detectar extremos (algoritmo simples e eficaz do app.py)
    analisador.detectar_extremos(janela=5)
    
    # Visualizar
    print("\n📊 Gerando visualização...")
    analisador.visualizar_extremos()
    
    # Relatório
    relatorio = analisador.relatorio_extremos()
    print(f"\n📋 RELATÓRIO:")
    print(f"🔺 Máximos: {relatorio['total_picos']}")
    print(f"🔻 Mínimos: {relatorio['total_vales']}")
    print(f"💰 Maior pico: ${relatorio['maior_pico']:.2f}")
    print(f"💰 Menor vale: ${relatorio['menor_vale']:.2f}")
    print(f"📊 Amplitude: ${relatorio['amplitude_extremos']:.2f}")
    
    print("\n✅ Análise concluída!")


if __name__ == "__main__":
    main() 