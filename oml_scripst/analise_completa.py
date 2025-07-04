"""
Análise Completa: Filas e Pilhas em Online Machine Learning
===========================================================

Este script demonstra o uso integrado de filas e pilhas para análise
completa de dados de Bitcoin, combinando:

1. Janelas deslizantes com filas (FIFO) para médias móveis
2. Detecção de extremos com pilhas (LIFO) para máximos e mínimos

Autor: João Paulo Assis Bonifácio
Disciplina: Estrutura de Dados (GES-115)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Importar os módulos implementados
from janelas_deslizantes_filas import AnalisadorBitcoinFilas
from detecao_extremos_pilhas import AnalisadorExtremosBitcoin

# Configurar estilo
plt.style.use('default')


class AnalisadorCompletoOML:
    """
    Analisador que combina filas e pilhas para análise completa de dados Bitcoin.
    """
    
    def __init__(self, caminho_dados: str = 'dados/db_bitcoin_1dia.csv'):
        """
        Inicializa o analisador completo.
        
        Args:
            caminho_dados (str): Caminho para os dados de Bitcoin
        """
        self.caminho_dados = caminho_dados
        self.dados = None
        
        # Analisadores especializados
        self.analisador_filas = AnalisadorBitcoinFilas(caminho_dados)
        self.analisador_pilhas = AnalisadorExtremosBitcoin(caminho_dados)
        
    def carregar_dados(self) -> bool:
        """Carrega os dados em todos os analisadores."""
        print("📥 Carregando dados de Bitcoin...")
        
        # Carregar dados no analisador principal
        try:
            self.dados = pd.read_csv(self.caminho_dados)
            self.dados['Date'] = pd.to_datetime(self.dados['Date'])
            self.dados = self.dados.sort_values('Date').reset_index(drop=True)
            
            # Carregar nos analisadores especializados
            sucesso_filas = self.analisador_filas.carregar_dados()
            sucesso_pilhas = self.analisador_pilhas.carregar_dados()
            
            if sucesso_filas and sucesso_pilhas:
                print(f"✅ Dados carregados com sucesso: {len(self.dados)} registros")
                print(f"📊 Período: {self.dados['Date'].min()} a {self.dados['Date'].max()}")
                return True
            else:
                print("❌ Erro ao carregar dados nos analisadores")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def executar_analise_filas(self) -> None:
        """Executa análise com filas (médias móveis)."""
        print("\n🔄 ANÁLISE COM FILAS (MÉDIAS MÓVEIS):")
        print("-" * 40)
        
        # Configurar janelas deslizantes
        configuracoes_janelas = {
            'Curta (15min)': 15,
            'Média (60min)': 60,
            'Longa (240min)': 240
        }
        self.analisador_filas.configurar_janelas(configuracoes_janelas)
        
        # Processar dados
        self.analisador_filas.processar_dados()
        
        # Visualizar
        print("📊 Gerando visualização das médias móveis...")
        self.analisador_filas.visualizar_medias_moveis()
        
        # Relatório
        relatorio = self.analisador_filas.relatorio_filas()
        print(f"\n📋 RELATÓRIO FILAS:")
        for nome, stats in relatorio['estatisticas_por_janela'].items():
            print(f"   • {nome}: Média ${stats['media_atual']:.2f}")
    
    def executar_analise_pilhas(self) -> None:
        """Executa análise com pilhas (extremos)."""
        print("\n📚 ANÁLISE COM PILHAS (EXTREMOS):")
        print("-" * 40)
        
        # Detectar extremos (algoritmo simples e eficaz do app.py)
        self.analisador_pilhas.detectar_extremos(janela=5)
        
        # Visualizar
        print("📊 Gerando visualização dos extremos...")
        self.analisador_pilhas.visualizar_extremos()
        
        # Relatório
        relatorio = self.analisador_pilhas.relatorio_extremos()
        print(f"\n📋 RELATÓRIO PILHAS:")
        print(f"   • Máximos: {relatorio['total_picos']}")
        print(f"   • Mínimos: {relatorio['total_vales']}")
        print(f"   • Maior pico: ${relatorio['maior_pico']:.2f}")
        print(f"   • Menor vale: ${relatorio['menor_vale']:.2f}")
    
    def visualizar_comparativo(self) -> None:
        """
        Cria uma visualização comparativa simples das duas análises.
        """
        print("\n📊 Gerando visualização comparativa...")
        
        # Criar figura com 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Subplot 1: Médias móveis
        ax1.plot(self.dados['Date'], self.dados['Close'], 
                color='lightblue', alpha=0.7, linewidth=1, label='Preço Bitcoin')
        
        # Plotar médias móveis
        cores = ['red', 'green', 'orange']
        for i, (nome, janela) in enumerate(self.analisador_filas.janelas.items()):
            if len(janela.historico_medias) > 0:
                ax1.plot(janela.historico_timestamps, janela.historico_medias, 
                        color=cores[i], linewidth=2, alpha=0.8,
                        label=f'Fila {nome}')
        
        ax1.set_title('Análise com Filas (FIFO): Médias Móveis', fontweight='bold')
        ax1.set_ylabel('Preço (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Extremos
        ax2.plot(self.dados['Date'], self.dados['Close'], 
                color='blue', linewidth=1, label='Preço Bitcoin', alpha=0.8)
        
        # Plotar extremos
        if self.analisador_pilhas.extremos:
            for idx, valor, tipo in self.analisador_pilhas.extremos:
                timestamp = self.dados.iloc[idx]['Date']
                if tipo == "pico":
                    ax2.scatter(timestamp, valor, color='red', s=40, marker='^', zorder=5)
                else:
                    ax2.scatter(timestamp, valor, color='green', s=40, marker='v', zorder=5)
            
            # Legendas
            ax2.scatter([], [], color='red', s=40, marker='^', 
                       label=f'Máximos ({len(self.analisador_pilhas.pilha_extremos.picos)})')
            ax2.scatter([], [], color='green', s=40, marker='v', 
                       label=f'Mínimos ({len(self.analisador_pilhas.pilha_extremos.vales)})')
        
        ax2.set_title('Análise com Pilhas (LIFO): Extremos', fontweight='bold')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Preço (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Análise Completa: Filas e Pilhas em Online Machine Learning', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def gerar_relatorio_final(self) -> dict:
        """Gera relatório final consolidado."""
        relatorio_filas = self.analisador_filas.relatorio_filas()
        relatorio_pilhas = self.analisador_pilhas.relatorio_extremos()
        
        return {
            'dados_processados': len(self.dados),
            'periodo': {
                'inicio': self.dados['Date'].min(),
                'fim': self.dados['Date'].max()
            },
            'filas': {
                'janelas': relatorio_filas['janelas_configuradas'],
                'medias_atuais': {nome: stats['media_atual'] 
                                for nome, stats in relatorio_filas['estatisticas_por_janela'].items()}
            },
            'pilhas': {
                'maximos': relatorio_pilhas['total_picos'],
                'minimos': relatorio_pilhas['total_vales'],
                'amplitude': relatorio_pilhas['amplitude_extremos']
            }
        }


def main():
    """Função principal para executar a análise completa."""
    print("🚀 ANÁLISE COMPLETA: FILAS E PILHAS EM ONLINE MACHINE LEARNING")
    print("=" * 70)
    print("📊 Demonstração de estruturas de dados aplicadas ao Bitcoin")
    print("-" * 70)
    
    # Inicializar analisador completo
    analisador = AnalisadorCompletoOML()
    
    # Carregar dados
    if not analisador.carregar_dados():
        print("❌ Falha ao carregar dados! Verifique o caminho do arquivo.")
        return
    
    # Executar análise com filas
    analisador.executar_analise_filas()
    
    # Executar análise com pilhas
    analisador.executar_analise_pilhas()
    
    # Visualização comparativa
    analisador.visualizar_comparativo()
    
    # Relatório final
    relatorio = analisador.gerar_relatorio_final()
    
    print("\n" + "=" * 70)
    print("📊 RELATÓRIO FINAL")
    print("=" * 70)
    
    print(f"📈 Dados processados: {relatorio['dados_processados']} registros")
    print(f"📅 Período: {relatorio['periodo']['inicio']} a {relatorio['periodo']['fim']}")
    
    print(f"\n🔄 FILAS (MÉDIAS MÓVEIS):")
    for nome, media in relatorio['filas']['medias_atuais'].items():
        print(f"   • {nome}: ${media:.2f}")
    
    print(f"\n📚 PILHAS (EXTREMOS):")
    print(f"   • Máximos detectados: {relatorio['pilhas']['maximos']}")
    print(f"   • Mínimos detectados: {relatorio['pilhas']['minimos']}")
    print(f"   • Amplitude total: ${relatorio['pilhas']['amplitude']:.2f}")
    
    print(f"\n✅ ANÁLISE CONCLUÍDA!")
    print("🎯 Estruturas de dados demonstradas com sucesso:")
    print("   • Filas (FIFO): Janelas deslizantes para médias móveis")
    print("   • Pilhas (LIFO): Detecção de máximos e mínimos")


if __name__ == "__main__":
    main() 