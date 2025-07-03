#!/usr/bin/env python3
"""
Visualização Animada: Fila Deslizante + Pilha Buffer
Mostra como os dados fluem através das estruturas de dados
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import time

class VisualizadorFilaPilha:
    def __init__(self, dados_exemplo, tamanho_fila=5, tamanho_buffer=3):
        self.dados = dados_exemplo
        self.tamanho_fila = tamanho_fila
        self.tamanho_buffer = tamanho_buffer
        
        # Estados das estruturas
        self.pilha_buffer = []
        self.fila_deslizante = []
        self.dados_processados = []
        self.posicao_atual = 0
        
        # Setup da figura
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('🎯 VISUALIZAÇÃO: FILA DESLIZANTE + PILHA BUFFER', 
                         fontsize=16, fontweight='bold', color='darkblue')
        
        # Configurar subplots
        self.ax_dados = self.axes[0, 0]      # Dados originais
        self.ax_pilha = self.axes[0, 1]      # Pilha buffer
        self.ax_fila = self.axes[1, 0]       # Fila deslizante
        self.ax_resultado = self.axes[1, 1]   # Dados processados
        
        self.setup_plots()
        
    def setup_plots(self):
        """Configura os subplots iniciais"""
        
        # 1. Dados Originais
        self.ax_dados.set_title('📊 DADOS ORIGINAIS', fontweight='bold')
        self.ax_dados.set_xlim(-0.5, len(self.dados) + 0.5)
        self.ax_dados.set_ylim(-0.5, 1.5)
        self.ax_dados.set_xlabel('Posição no Dataset')
        
        # 2. Pilha Buffer
        self.ax_pilha.set_title(f'📚 PILHA BUFFER (Tamanho: {self.tamanho_buffer})', fontweight='bold')
        self.ax_pilha.set_xlim(-0.5, 2)
        self.ax_pilha.set_ylim(-0.5, self.tamanho_buffer + 0.5)
        self.ax_pilha.set_ylabel('Posição na Pilha (LIFO)')
        
        # 3. Fila Deslizante
        self.ax_fila.set_title(f'🚂 FILA DESLIZANTE (Tamanho: {self.tamanho_fila})', fontweight='bold')
        self.ax_fila.set_xlim(-0.5, self.tamanho_fila + 0.5)
        self.ax_fila.set_ylim(-0.5, 1.5)
        self.ax_fila.set_xlabel('Posição na Fila (FIFO)')
        
        # 4. Dados Processados
        self.ax_resultado.set_title('✅ DADOS PROCESSADOS (Janelas)', fontweight='bold')
        self.ax_resultado.set_xlim(-0.5, 10)
        self.ax_resultado.set_ylim(-0.5, 3)
        self.ax_resultado.set_xlabel('Janelas Geradas')
        
        # Remove ticks desnecessários
        for ax in [self.ax_pilha, self.ax_fila, self.ax_resultado]:
            ax.set_xticks([])
            ax.set_yticks([])
            
    def desenhar_dados_originais(self):
        """Desenha a sequência de dados originais"""
        self.ax_dados.clear()
        self.ax_dados.set_title('📊 DADOS ORIGINAIS', fontweight='bold')
        self.ax_dados.set_xlim(-0.5, min(20, len(self.dados)) + 0.5)
        self.ax_dados.set_ylim(-0.5, 1.5)
        
        # Desenhar todos os dados
        for i, valor in enumerate(self.dados[:20]):  # Mostrar só os primeiros 20
            cor = 'red' if i == self.posicao_atual else 'lightgray'
            rect = FancyBboxPatch((i-0.4, 0.1), 0.8, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor=cor, edgecolor='black', linewidth=1)
            self.ax_dados.add_patch(rect)
            self.ax_dados.text(i, 0.5, f'{int(valor)}', ha='center', va='center', 
                              fontweight='bold', fontsize=10)
        
        # Seta indicando posição atual
        if self.posicao_atual < 20:
            self.ax_dados.annotate('👆 ATUAL', xy=(self.posicao_atual, 1.2), 
                                  ha='center', fontweight='bold', color='red')
                                  
    def desenhar_pilha_buffer(self):
        """Desenha o estado atual da pilha buffer"""
        self.ax_pilha.clear()
        self.ax_pilha.set_title(f'📚 PILHA BUFFER (Tamanho: {self.tamanho_buffer})', fontweight='bold')
        self.ax_pilha.set_xlim(-0.5, 2)
        self.ax_pilha.set_ylim(-0.5, self.tamanho_buffer + 0.5)
        
        # Desenhar slots da pilha (do fundo para o topo)
        for i in range(self.tamanho_buffer):
            if i < len(self.pilha_buffer):
                # Slot ocupado
                rect = FancyBboxPatch((0.1, i + 0.1), 1.8, 0.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor='orange', edgecolor='black', linewidth=2)
                self.ax_pilha.add_patch(rect)
                self.ax_pilha.text(1, i + 0.5, f'{int(self.pilha_buffer[i])}', 
                                  ha='center', va='center', fontweight='bold', fontsize=12)
            else:
                # Slot vazio
                rect = FancyBboxPatch((0.1, i + 0.1), 1.8, 0.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor='white', edgecolor='gray', 
                                    linewidth=1, linestyle='--')
                self.ax_pilha.add_patch(rect)
                self.ax_pilha.text(1, i + 0.5, 'vazio', ha='center', va='center', 
                                  fontweight='normal', fontsize=10, color='gray')
        
        # Indicar topo da pilha
        if self.pilha_buffer:
            self.ax_pilha.annotate('⬆️ TOPO', xy=(2.1, len(self.pilha_buffer) - 0.5), 
                                  ha='left', va='center', fontweight='bold', color='red')
                                  
    def desenhar_fila_deslizante(self):
        """Desenha o estado atual da fila deslizante"""
        self.ax_fila.clear()
        self.ax_fila.set_title(f'🚂 FILA DESLIZANTE (Tamanho: {self.tamanho_fila})', fontweight='bold')
        self.ax_fila.set_xlim(-0.5, self.tamanho_fila + 0.5)
        self.ax_fila.set_ylim(-0.5, 1.5)
        
        # Desenhar slots da fila (da esquerda para direita)
        for i in range(self.tamanho_fila):
            if i < len(self.fila_deslizante):
                # Slot ocupado
                rect = FancyBboxPatch((i + 0.1, 0.1), 0.8, 0.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightblue', edgecolor='black', linewidth=2)
                self.ax_fila.add_patch(rect)
                self.ax_fila.text(i + 0.5, 0.5, f'{int(self.fila_deslizante[i])}', 
                                 ha='center', va='center', fontweight='bold', fontsize=10)
            else:
                # Slot vazio
                rect = FancyBboxPatch((i + 0.1, 0.1), 0.8, 0.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor='white', edgecolor='gray', 
                                    linewidth=1, linestyle='--')
                self.ax_fila.add_patch(rect)
                self.ax_fila.text(i + 0.5, 0.5, 'vazio', ha='center', va='center', 
                                 fontweight='normal', fontsize=8, color='gray')
        
        # Indicações de entrada e saída
        if len(self.fila_deslizante) > 0:
            self.ax_fila.annotate('⬅️ ENTRADA', xy=(-0.3, 0.5), ha='right', va='center', 
                                 fontweight='bold', color='green')
        if len(self.fila_deslizante) == self.tamanho_fila:
            self.ax_fila.annotate('SAÍDA ➡️', xy=(self.tamanho_fila + 0.1, 0.5), 
                                 ha='left', va='center', fontweight='bold', color='red')
                                 
    def desenhar_dados_processados(self):
        """Desenha as janelas de dados processados"""
        self.ax_resultado.clear()
        self.ax_resultado.set_title('✅ DADOS PROCESSADOS (Janelas)', fontweight='bold')
        
        num_janelas = len(self.dados_processados)
        if num_janelas == 0:
            self.ax_resultado.text(2, 1, 'Aguardando dados...', ha='center', va='center',
                                  fontsize=12, style='italic', color='gray')
            return
            
        # Mostrar apenas as últimas 8 janelas
        inicio = max(0, num_janelas - 8)
        self.ax_resultado.set_xlim(-0.5, 8.5)
        self.ax_resultado.set_ylim(-0.5, 2.5)
        
        for i, janela in enumerate(self.dados_processados[inicio:]):
            x_pos = i
            
            # Retângulo para a janela
            rect = FancyBboxPatch((x_pos + 0.05, 0.1), 0.9, 1.8,
                                boxstyle="round,pad=0.05",
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
            self.ax_resultado.add_patch(rect)
            
            # Mostrar elementos da janela
            janela_str = ' '.join([f'{int(x)}' for x in janela])
            self.ax_resultado.text(x_pos + 0.5, 1, janela_str, ha='center', va='center',
                                  fontweight='bold', fontsize=9, rotation=90)
                                  
        # Contador
        self.ax_resultado.text(4, 2.2, f'Total: {num_janelas} janelas', ha='center', va='center',
                              fontweight='bold', fontsize=12, color='darkgreen')
    
    def processar_proximo_passo(self):
        """Simula um passo do processamento"""
        if self.posicao_atual >= len(self.dados):
            return False
            
        # Pegar próximo dado
        valor_atual = self.dados[self.posicao_atual]
        
        # 1. Empilhar na pilha buffer
        self.pilha_buffer.append(valor_atual)
        
        # 2. Verificar se a pilha está cheia
        valor_para_fila = None
        if len(self.pilha_buffer) >= self.tamanho_buffer:
            valor_para_fila = self.pilha_buffer[-1]  # Pega o topo (último)
            self.pilha_buffer.clear()  # Limpa a pilha
            
        # 3. Se temos valor para a fila, processar
        if valor_para_fila is not None:
            # Se a fila está cheia, remover o primeiro
            if len(self.fila_deslizante) >= self.tamanho_fila:
                self.fila_deslizante.pop(0)
                
            # Adicionar o novo valor
            self.fila_deslizante.append(valor_para_fila)
            
            # Se a fila está cheia, criar uma janela
            if len(self.fila_deslizante) == self.tamanho_fila:
                nova_janela = self.fila_deslizante.copy()
                self.dados_processados.append(nova_janela)
        
        self.posicao_atual += 1
        return True
    
    def animate(self, frame):
        """Função de animação"""
        # Processar próximo passo
        continuar = self.processar_proximo_passo()
        
        # Redesenhar tudo
        self.desenhar_dados_originais()
        self.desenhar_pilha_buffer()
        self.desenhar_fila_deslizante()
        self.desenhar_dados_processados()
        
        # Adicionar informações de status
        status = f"Passo: {self.posicao_atual} | Pilha: {len(self.pilha_buffer)}/{self.tamanho_buffer} | Fila: {len(self.fila_deslizante)}/{self.tamanho_fila} | Janelas: {len(self.dados_processados)}"
        self.fig.suptitle(f'🎯 VISUALIZAÇÃO: FILA DESLIZANTE + PILHA BUFFER\n{status}', 
                         fontsize=14, fontweight='bold', color='darkblue')
        
        return not continuar
    
    def executar_animacao(self, intervalo=800):
        """Executa a animação"""
        anim = animation.FuncAnimation(self.fig, self.animate, interval=intervalo, 
                                     repeat=False, blit=False)
        plt.tight_layout()
        plt.show()
        return anim

def main():
    """Função principal"""
    print("🎬 Iniciando visualização animada...")
    
    # Dados de exemplo simples (números inteiros pequenos)
    dados_exemplo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    print(f"📊 Dados gerados: {len(dados_exemplo)} valores")
    print(f"   Sequência: {dados_exemplo[:10]}... até {dados_exemplo[-1]}")
    
    # Configurações
    tamanho_fila = 5
    tamanho_buffer = 3
    
    print(f"⚙️  Configuração:")
    print(f"   - Tamanho da Fila: {tamanho_fila}")
    print(f"   - Tamanho da Pilha Buffer: {tamanho_buffer}")
    print(f"   - Espaçamento: dados de {tamanho_buffer} em {tamanho_buffer}")
    
    # Criar visualizador
    visualizador = VisualizadorFilaPilha(dados_exemplo, tamanho_fila, tamanho_buffer)
    
    print("\n🎯 Iniciando animação...")
    print("   📚 PILHA BUFFER: Acumula dados e libera o topo quando cheia")
    print("   🚂 FILA DESLIZANTE: Mantém janela de valores para ML")
    print("   ✅ RESULTADO: Janelas prontas para treinamento")
    
    # Executar
    anim = visualizador.executar_animacao(intervalo=1000)  # 1 segundo por frame
    
    print("\n✅ Visualização concluída!")
    print(f"   Total de janelas geradas: {len(visualizador.dados_processados)}")

if __name__ == "__main__":
    main() 