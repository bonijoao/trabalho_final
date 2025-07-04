"""
Janelas Deslizantes com Filas para Online Machine Learning
==========================================================

Este módulo implementa janelas deslizantes usando estruturas de fila (FIFO) 
para processamento em tempo real de dados de Bitcoin - foco em médias móveis.

Autor: João Paulo Assis Bonifácio
Disciplina: Estrutura de Dados (GES-115)
"""

import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')


class JanelaDeslizanteFila:
    """
    Implementa uma janela deslizante usando fila (FIFO) para Online Machine Learning.
    
    A fila mantém um número fixo de observações mais recentes, removendo 
    automaticamente os dados mais antigos quando novos dados chegam.
    """
    
    def __init__(self, tamanho_janela: int, nome: str = "Janela"):
        self.tamanho_janela = tamanho_janela
        self.nome = nome
        self.fila = deque(maxlen=tamanho_janela)  # Fila FIFO com tamanho fixo
        self.timestamps = deque(maxlen=tamanho_janela)
        
        # Otimização: manter somas para cálculo eficiente O(1)
        self._soma_total = 0.0
        self._contador_updates = 0
        
        # Histórico para visualização
        self.historico_medias = []
        self.historico_timestamps = []
        
    def adicionar_valor(self, valor: float, timestamp):
        """
        Adiciona um novo valor à janela deslizante (operação FIFO).
        """
        # Se a janela está cheia, remove o valor mais antigo (FIFO)
        valor_removido = None
        if len(self.fila) == self.tamanho_janela:
            valor_removido = self.fila[0]  # Primeiro da fila (mais antigo)
            self._soma_total -= valor_removido
        
        # Adiciona o novo valor no final da fila (FIFO)
        self.fila.append(valor)
        self.timestamps.append(timestamp)
        self._soma_total += valor
        self._contador_updates += 1
        
        # Calcula média atual
        if self.esta_cheia():
            media_atual = self._soma_total / len(self.fila)
            self.historico_medias.append(media_atual)
            self.historico_timestamps.append(timestamp)
        
        return {
            'valor_adicionado': valor,
            'valor_removido': valor_removido,
            'media_atual': self._soma_total / len(self.fila) if len(self.fila) > 0 else 0,
            'janela_cheia': self.esta_cheia()
        }
    
    def esta_cheia(self) -> bool:
        """Verifica se a janela atingiu seu tamanho máximo."""
        return len(self.fila) == self.tamanho_janela
    
    def obter_media_atual(self) -> float:
        """Retorna a média atual da janela."""
        return self._soma_total / len(self.fila) if len(self.fila) > 0 else 0


class AnalisadorBitcoinFilas:
    """
    Analisador simples que processa dados de Bitcoin usando filas para médias móveis.
    """
    
    def __init__(self, caminho_dados: str = 'dados/db_bitcoin_1dia.csv'):
        self.caminho_dados = caminho_dados
        self.dados = None
        self.janelas = {}
        
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
    
    def configurar_janelas(self, configuracoes: dict) -> None:
        """
        Configura múltiplas janelas deslizantes.
        
        Args:
            configuracoes (dict): {'nome_janela': tamanho}
        """
        self.janelas = {}
        for nome, tamanho in configuracoes.items():
            self.janelas[nome] = JanelaDeslizanteFila(tamanho, nome)
        
        print(f"✅ Configuradas {len(configuracoes)} janelas deslizantes:")
        for nome, tamanho in configuracoes.items():
            print(f"   • {nome}: {tamanho} períodos")
    
    def processar_dados(self, limite: int = None) -> None:
        """
        Processa os dados usando as janelas deslizantes configuradas.
        
        Args:
            limite (int, optional): Limitar número de registros processados
        """
        if self.dados is None:
            print("Dados não carregados!")
            return
        
        dados_para_processar = self.dados.copy()
        if limite:
            dados_para_processar = dados_para_processar.head(limite)
        
        print(f"🔄 Processando {len(dados_para_processar)} registros...")
        
        # Processa cada linha sequencialmente
        for _, row in dados_para_processar.iterrows():
            valor = row['Close']
            timestamp = row['Date']
            
            # Processa em todas as janelas
            for janela in self.janelas.values():
                janela.adicionar_valor(valor, timestamp)
        
        print("✅ Processamento concluído!")
    
    def visualizar_medias_moveis(self) -> None:
        """
        Visualiza apenas as médias móveis com a série histórica original.
        """
        if not self.janelas:
            print("Nenhuma janela configurada!")
            return
        
        # Criar figura simples
        plt.figure(figsize=(15, 8))
        
        # Plotar preço original
        plt.plot(self.dados['Date'], self.dados['Close'], 
                color='lightblue', alpha=0.7, linewidth=1, label='Preço Bitcoin')
        
        # Plotar médias móveis de cada janela
        cores = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (nome, janela) in enumerate(self.janelas.items()):
            if len(janela.historico_medias) > 0:
                cor = cores[i % len(cores)]
                plt.plot(janela.historico_timestamps, janela.historico_medias, 
                        color=cor, linewidth=2.5, alpha=0.8,
                        label=f'Média Móvel {nome} (n={janela.tamanho_janela})')
        
        plt.title('Médias Móveis Calculadas com Filas (FIFO) - Bitcoin USD', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Data')
        plt.ylabel('Preço (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def relatorio_filas(self) -> dict:
        """Gera relatório das operações com filas."""
        relatorio = {
            'total_dados_processados': len(self.dados) if self.dados is not None else 0,
            'janelas_configuradas': len(self.janelas),
            'estatisticas_por_janela': {}
        }
        
        for nome, janela in self.janelas.items():
            if janela.esta_cheia():
                relatorio['estatisticas_por_janela'][nome] = {
                    'tamanho': janela.tamanho_janela,
                    'media_atual': janela.obter_media_atual(),
                    'updates_totais': janela._contador_updates,
                    'historico_length': len(janela.historico_medias)
                }
        
        return relatorio


def main():
    """Função principal de demonstração."""
    print("🚀 Análise de Bitcoin com Janelas Deslizantes (Filas FIFO)")
    print("=" * 60)
    
    # Inicializar analisador
    analisador = AnalisadorBitcoinFilas()
    
    # Carregar dados
    if not analisador.carregar_dados():
        print("❌ Falha ao carregar dados!")
        return
    
    print(f"✅ Dados carregados: {len(analisador.dados)} registros")
    
    # Configurar múltiplas janelas deslizantes
    configuracoes_janelas = {
        'Curta (15min)': 15,     # Janela de 15 minutos
        'Média (60min)': 60,     # Janela de 1 hora  
        'Longa (240min)': 240    # Janela de 4 horas
    }
    analisador.configurar_janelas(configuracoes_janelas)
    
    # Processar dados
    analisador.processar_dados()
    
    # Gerar visualizações
    print("\n📊 Gerando visualização das médias móveis...")
    analisador.visualizar_medias_moveis()
    
    # Relatório final
    relatorio = analisador.relatorio_filas()
    print(f"\n📋 RELATÓRIO:")
    print(f"📊 Total processado: {relatorio['total_dados_processados']} registros")
    print(f"🔄 Janelas ativas: {relatorio['janelas_configuradas']}")
    
    for nome, stats in relatorio['estatisticas_por_janela'].items():
        print(f"\n🎯 {nome}:")
        print(f"   • Tamanho da fila: {stats['tamanho']} elementos")
        print(f"   • Média atual: ${stats['media_atual']:.2f}")
        print(f"   • Total de updates: {stats['updates_totais']}")
        print(f"   • Pontos no histórico: {stats['historico_length']}")
    
    print("\n✅ Análise concluída! Estruturas de fila (FIFO) demonstradas com sucesso.")


if __name__ == "__main__":
    main() 