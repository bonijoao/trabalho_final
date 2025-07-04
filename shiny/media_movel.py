# shiny run shiny/media_movel.py

from shiny import App, reactive, render, ui
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector

# Dados de exemplo baseados no formato fornecido
df = pd.read_csv('dados/db_bitcoin_1dia.csv')

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Configurações da Média Móvel"),
        ui.input_slider(
            "window_size", 
            "Tamanho da Janela:", 
            min=2, 
            max=50, 
            value=5, 
            step=1
        ),
        ui.input_select(
            "column",
            "Coluna para Média Móvel:",
            choices=["Close", "Open", "High", "Low", "Volume"],
            selected="Close"
        ),
        ui.div(
            ui.h5("Informações:"),
            ui.output_text("window_info"),
            ui.output_text("data_info")
        ),
        ui.br(),
        ui.input_action_button("reset_zoom", "Resetar Zoom", class_="btn-outline-primary")
    ),
    ui.card(
        ui.card_header("Gráfico com Média Móvel - Clique e arraste para dar zoom"),
        ui.output_plot("moving_average_plot")
    )
)

def server(input, output, session):
    
    # Armazenar os limites do zoom
    zoom_limits = reactive.value(None)
    
    @reactive.calc
    def calculate_moving_average():
        window = input.window_size()
        column = input.column()
        
        # Criar uma cópia do dataframe
        df_ma = df.copy()
        
        # Calcular média móvel
        df_ma[f'{column}_MA_{window}'] = df_ma[column].rolling(window=window).mean()
        
        return df_ma
    
    @render.text
    def window_info():
        return f"Janela atual: {input.window_size()} períodos"
    
    @render.text
    def data_info():
        return f"Total de dados: {len(df)} registros"
    
    @reactive.effect
    @reactive.event(input.reset_zoom)
    def reset_zoom():
        zoom_limits.set(None)
    
    @render.plot
    def moving_average_plot():
        df_ma = calculate_moving_average()
        window = input.window_size()
        column = input.column()
        
        # Configurar matplotlib para modo interativo
        plt.ioff()  # Desligar modo interativo para evitar conflitos
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plotar dados originais
        ax.plot(df_ma['Date'], df_ma[column], 
                label=f'{column} Original', 
                alpha=0.7, 
                linewidth=1)
        
        # Plotar média móvel
        ax.plot(df_ma['Date'], df_ma[f'{column}_MA_{window}'], 
                label=f'Média Móvel ({window} períodos)', 
                linewidth=2, 
                color='red')
        
        ax.set_title(f'Série Temporal com Média Móvel - {column}', fontsize=16)
        ax.set_xlabel('Data/Hora', fontsize=12)
        ax.set_ylabel(f'{column}', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Aplicar zoom se houver limites definidos
        limits = zoom_limits.get()
        if limits:
            ax.set_xlim(limits['x'])
            ax.set_ylim(limits['y'])
        
        # Otimizar eixo X para muitas observações
        total_points = len(df_ma)
        if total_points > 50:
            # Para muitos pontos, mostrar apenas alguns labels
            step = max(1, total_points // 10)
            tick_positions = range(0, total_points, step)
            ax.set_xticks([df_ma['Date'].iloc[i] for i in tick_positions])
        elif total_points > 20:
            # Para quantidade média, mostrar menos labels
            step = max(1, total_points // 5)
            tick_positions = range(0, total_points, step)
            ax.set_xticks([df_ma['Date'].iloc[i] for i in tick_positions])
        
        # Rotacionar labels do eixo x
        plt.xticks(rotation=45, ha='right')
        
        # Função para capturar seleção de zoom
        def on_select(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            
            if x1 and x2 and y1 and y2:
                # Converter coordenadas para datas
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                
                # Armazenar limites do zoom
                zoom_limits.set({
                    'x': (x_min, x_max),
                    'y': (y_min, y_max)
                })
        
        # Adicionar seletor de retângulo para zoom
        selector = RectangleSelector(
            ax, 
            on_select,
            useblit=True,
            button=[1],  # Apenas botão esquerdo do mouse
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        # Armazenar o seletor no axes para evitar garbage collection
        ax._selector = selector
        
        plt.tight_layout()
        
        return fig

app = App(app_ui, server)
