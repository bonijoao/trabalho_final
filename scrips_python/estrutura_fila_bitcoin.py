"""
Estrutura de Dados em Fila para Bitcoin
=====================================

Este módulo implementa uma fila (Queue) otimizada para trabalhar com dados de Bitcoin,
usando collections.deque para operações O(1) de inserção e remoção.

Funcionalidades:
- Carregar dados históricos de Bitcoin de arquivo CSV
- Adicionar novos dados à fila (FIFO - First In, First Out)
- Remover dados da fila
- Calcular médias móveis de forma eficiente
- Operações básicas de fila com complexidade O(1)
"""

from collections import deque
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
import os


class FilaBitcoin:
    """
    Classe que implementa uma fila para dados de Bitcoin usando collections.deque.
    
    A fila mantém dados de Bitcoin ordenados cronologicamente, permitindo
    operações eficientes de inserção e remoção para cálculos de médias móveis.
    """
    
    def __init__(self, maxlen: Optional[int] = None):
        """
        Inicializa a fila de Bitcoin.
        
        Args:
            maxlen (Optional[int]): Tamanho máximo da fila. Se especificado,
                                  elementos mais antigos são removidos automaticamente
                                  quando o limite é atingido.
        """
        self.fila = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self._total_soma = 0.0  # Para otimizar cálculo de médias
        
    def carregar_dados_csv(self, caminho_arquivo: str, limite_linhas: Optional[int] = None) -> int:
        """
        Carrega dados de Bitcoin de um arquivo CSV para a fila.
        
        Args:
            caminho_arquivo (str): Caminho para o arquivo CSV
            limite_linhas (Optional[int]): Número máximo de linhas a carregar
            
        Returns:
            int: Número de registros carregados
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            ValueError: Se o formato do arquivo for inválido
        """
        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
        
        try:
            # Carrega o CSV
            df = pd.read_csv(caminho_arquivo)
            
            # Verifica se as colunas necessárias existem
            colunas_necessarias = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for coluna in colunas_necessarias:
                if coluna not in df.columns:
                    raise ValueError(f"Coluna '{coluna}' não encontrada no arquivo CSV")
            
            # Limita o número de linhas se especificado
            if limite_linhas:
                df = df.head(limite_linhas)
            
            # Converte cada linha em um dicionário e adiciona à fila
            registros_adicionados = 0
            for _, row in df.iterrows():
                registro = {
                    'date': pd.to_datetime(row['Date']),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                }
                self.adicionar(registro)
                registros_adicionados += 1
            
            print(f"✅ {registros_adicionados} registros carregados com sucesso!")
            return registros_adicionados
            
        except Exception as e:
            raise ValueError(f"Erro ao processar arquivo CSV: {str(e)}")
    
    def adicionar(self, registro: Dict) -> None:
        """
        Adiciona um novo registro ao final da fila (FIFO).
        
        Args:
            registro (Dict): Dicionário com dados do Bitcoin contendo:
                           - date: datetime
                           - open, high, low, close, volume: float
        """
        if not isinstance(registro, dict):
            raise ValueError("Registro deve ser um dicionário")
        
        # Verifica se tem as chaves necessárias
        chaves_necessarias = ['date', 'open', 'high', 'low', 'close', 'volume']
        for chave in chaves_necessarias:
            if chave not in registro:
                raise ValueError(f"Chave '{chave}' não encontrada no registro")
        
        # Se a fila tem tamanho máximo e está cheia, remove o elemento mais antigo
        if self.maxlen and len(self.fila) >= self.maxlen:
            removido = self.fila.popleft()
            self._total_soma -= removido['close']
        
        # Adiciona o novo registro
        self.fila.append(registro)
        self._total_soma += registro['close']
    
    def remover(self) -> Optional[Dict]:
        """
        Remove e retorna o elemento mais antigo da fila (FIFO).
        
        Returns:
            Optional[Dict]: Registro removido ou None se a fila estiver vazia
        """
        if self.esta_vazia():
            return None
        
        removido = self.fila.popleft()
        self._total_soma -= removido['close']
        return removido
    
    def remover_direita(self) -> Optional[Dict]:
        """
        Remove e retorna o elemento mais recente da fila.
        
        Returns:
            Optional[Dict]: Registro removido ou None se a fila estiver vazia
        """
        if self.esta_vazia():
            return None
        
        removido = self.fila.pop()
        self._total_soma -= removido['close']
        return removido
    
    def primeiro(self) -> Optional[Dict]:
        """
        Retorna o primeiro elemento da fila sem removê-lo.
        
        Returns:
            Optional[Dict]: Primeiro registro ou None se a fila estiver vazia
        """
        if self.esta_vazia():
            return None
        return self.fila[0]
    
    def ultimo(self) -> Optional[Dict]:
        """
        Retorna o último elemento da fila sem removê-lo.
        
        Returns:
            Optional[Dict]: Último registro ou None se a fila estiver vazia
        """
        if self.esta_vazia():
            return None
        return self.fila[-1]
    
    def esta_vazia(self) -> bool:
        """
        Verifica se a fila está vazia.
        
        Returns:
            bool: True se a fila estiver vazia, False caso contrário
        """
        return len(self.fila) == 0
    
    def tamanho(self) -> int:
        """
        Retorna o número de elementos na fila.
        
        Returns:
            int: Número de elementos na fila
        """
        return len(self.fila)
    
    def limpar(self) -> None:
        """
        Remove todos os elementos da fila.
        """
        self.fila.clear()
        self._total_soma = 0.0
    
    def media_movel_simples(self) -> float:
        """
        Calcula a média móvel simples dos preços de fechamento na fila.
        
        Returns:
            float: Média móvel simples ou 0.0 se a fila estiver vazia
        """
        if self.esta_vazia():
            return 0.0
        
        return self._total_soma / len(self.fila)
    
    def obter_precos_fechamento(self) -> List[float]:
        """
        Retorna uma lista com todos os preços de fechamento na fila.
        
        Returns:
            List[float]: Lista de preços de fechamento
        """
        return [registro['close'] for registro in self.fila]
    
    def obter_dados_completos(self) -> List[Dict]:
        """
        Retorna uma lista com todos os registros na fila.
        
        Returns:
            List[Dict]: Lista com todos os registros
        """
        return list(self.fila)
    
    def estatisticas_resumo(self) -> Dict:
        """
        Calcula estatísticas resumo dos dados na fila.
        
        Returns:
            Dict: Dicionário com estatísticas resumo
        """
        if self.esta_vazia():
            return {
                'total_registros': 0,
                'preco_minimo': 0,
                'preco_maximo': 0,
                'preco_medio': 0,
                'volume_total': 0
            }
        
        precos = self.obter_precos_fechamento()
        volumes = [registro['volume'] for registro in self.fila]
        
        return {
            'total_registros': len(self.fila),
            'preco_minimo': min(precos),
            'preco_maximo': max(precos),
            'preco_medio': sum(precos) / len(precos),
            'volume_total': sum(volumes),
            'periodo_inicio': self.primeiro()['date'],
            'periodo_fim': self.ultimo()['date']
        }
    
    def exportar_para_csv(self, caminho_arquivo: str) -> None:
        """
        Exporta os dados da fila para um arquivo CSV.
        
        Args:
            caminho_arquivo (str): Caminho para salvar o arquivo CSV
        """
        if self.esta_vazia():
            print("⚠️  Fila vazia, nenhum dado para exportar")
            return
        
        dados = []
        for registro in self.fila:
            dados.append({
                'Date': registro['date'].strftime('%Y-%m-%d %H:%M:%S'),
                'Open': registro['open'],
                'High': registro['high'],
                'Low': registro['low'],
                'Close': registro['close'],
                'Volume': registro['volume']
            })
        
        df = pd.DataFrame(dados)
        df.to_csv(caminho_arquivo, index=False)
        print(f"✅ Dados exportados para: {caminho_arquivo}")
    
    def __str__(self) -> str:
        """
        Representação em string da fila.
        
        Returns:
            str: Informações sobre a fila
        """
        if self.esta_vazia():
            return "FilaBitcoin: Vazia"
        
        stats = self.estatisticas_resumo()
        return f"""FilaBitcoin:
├── Registros: {stats['total_registros']}
├── Período: {stats['periodo_inicio'].strftime('%Y-%m-%d %H:%M')} até {stats['periodo_fim'].strftime('%Y-%m-%d %H:%M')}
├── Preço (USD): ${stats['preco_minimo']:,.2f} - ${stats['preco_maximo']:,.2f}
├── Preço Médio: ${stats['preco_medio']:,.2f}
└── Volume Total: {stats['volume_total']:,.2f}"""


# Exemplo de uso
if __name__ == "__main__":
    # Demonstração de uso da FilaBitcoin
    print("🚀 Demonstração da Estrutura de Fila para Bitcoin\n")
    
    # Cria uma fila com tamanho máximo de 100 elementos
    fila_btc = FilaBitcoin(maxlen=100)
    
    # Carrega dados do arquivo CSV
    try:
        caminho_csv = os.path.join(os.path.dirname(__file__), '..', 'dados', 'db_bitcoin_1dia.csv')
        registros_carregados = fila_btc.carregar_dados_csv(caminho_csv, limite_linhas=50)
        
        print(f"\n📊 Estatísticas da Fila:")
        print(fila_btc)
        
        print(f"\n📈 Média Móvel Simples: ${fila_btc.media_movel_simples():,.2f}")
        
        # Demonstra operações básicas da fila
        print(f"\n🔄 Operações da Fila:")
        print(f"• Tamanho atual: {fila_btc.tamanho()}")
        print(f"• Primeiro registro: {fila_btc.primeiro()['date']} - ${fila_btc.primeiro()['close']:,.2f}")
        print(f"• Último registro: {fila_btc.ultimo()['date']} - ${fila_btc.ultimo()['close']:,.2f}")
        
        # Remove alguns elementos
        print(f"\n🗑️  Removendo 3 elementos mais antigos:")
        for i in range(3):
            removido = fila_btc.remover()
            if removido:
                print(f"   Removido: {removido['date']} - ${removido['close']:,.2f}")
        
        print(f"\n📊 Estatísticas após remoção:")
        print(f"• Novo tamanho: {fila_btc.tamanho()}")
        print(f"• Nova média móvel: ${fila_btc.media_movel_simples():,.2f}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")