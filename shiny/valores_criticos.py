# shiny run shiny/valores_criticos.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from shiny import App, render, ui, reactive
from datetime import datetime
import os

# Função para encontrar máximos e mínimos locais
def find_local_extrema(data, column='Close', prominence=0.1):
    """
    Encontra máximos e mínimos locais em uma série temporal
    
    Args:
        data: DataFrame com os dados
        column: Nome da coluna para análise
        prominence: Proeminência mínima para considerar um pico
    
    Returns:
        Tupla com (DataFrame de máximos, DataFrame de mínimos)
    """
    prices = data[column].values
    
    # Encontrar máximos locais
    peaks_max, _ = find_peaks(prices, prominence=prominence * np.std(prices))
    
    # Encontrar mínimos locais (invertendo a série)
    peaks_min, _ = find_peaks(-prices, prominence=prominence * np.std(prices))
    
    # Criar DataFrame para máximos
    maximos_list = []
    for idx in peaks_max:
        maximos_list.append({
            'Date': data.iloc[idx]['Date'],
            'Price': prices[idx],
            'Index': idx
        })
    
    # Criar DataFrame para mínimos
    minimos_list = []
    for idx in peaks_min:
        minimos_list.append({
            'Date': data.iloc[idx]['Date'],
            'Price': prices[idx],
            'Index': idx
        })
    
    # Converter para DataFrames e ordenar por data (mais recente primeiro - LIFO)
    maximos_df = pd.DataFrame(maximos_list)
    minimos_df = pd.DataFrame(minimos_list)
    
    if not maximos_df.empty:
        maximos_df = maximos_df.sort_values('Date', ascending=False).reset_index(drop=True)
        maximos_df['Price'] = maximos_df['Price'].round(2)
    
    if not minimos_df.empty:
        minimos_df = minimos_df.sort_values('Date', ascending=False).reset_index(drop=True)
        minimos_df['Price'] = minimos_df['Price'].round(2)
    
    return maximos_df, minimos_df

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Configurações"),
        ui.input_file("file", "Carregar arquivo CSV (opcional)", 
                     accept=[".csv"], multiple=False),
        ui.input_numeric("prominence", "Proeminência (sensibilidade)", 
                        value=0.01, min=0.001, max=0.1, step=0.001),
        ui.input_action_button("analyze", "Analisar Dados", class_="btn-primary"),
        ui.hr(),
        ui.h4("Informações"),
        ui.output_text("info")
    ),
    ui.card(
        ui.card_header("Gráfico da Série Temporal com Extremos"),
        ui.output_plot("price_plot")
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Pilha de Máximos (LIFO)"),
            ui.output_data_frame("maximos_table")
        ),
        ui.card(
            ui.card_header("Pilha de Mínimos (LIFO)"),
            ui.output_data_frame("minimos_table")
        ),
        col_widths=[6, 6]
    )
)

def server(input, output, session):
    # Reactive values para armazenar os dados
    data = reactive.value(None)
    maximos_data = reactive.value(None)
    minimos_data = reactive.value(None)
    
    # Carregar dados iniciais se o arquivo existir
    @reactive.effect
    def load_initial_data():
        default_path = "dados/db_bitcoin_1dia.csv"
        if os.path.exists(default_path):
            try:
                df = pd.read_csv(default_path)
                # Converter Date para datetime se necessário
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                data.set(df)
            except Exception as e:
                print(f"Erro ao carregar arquivo padrão: {e}")
    
    # Processar arquivo carregado pelo usuário
    @reactive.effect
    @reactive.event(input.file)
    def process_uploaded_file():
        if input.file() is not None:
            try:
                file_info = input.file()[0]
                df = pd.read_csv(file_info["datapath"])
                
                # Converter Date para datetime se necessário
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                data.set(df)
            except Exception as e:
                print(f"Erro ao processar arquivo: {e}")
    
    # Analisar dados quando o botão for clicado
    @reactive.effect
    @reactive.event(input.analyze)
    def analyze_data():
        if data() is not None:
            try:
                maximos_df, minimos_df = find_local_extrema(data(), prominence=input.prominence())
                maximos_data.set(maximos_df)
                minimos_data.set(minimos_df)
            except Exception as e:
                print(f"Erro na análise: {e}")
    
    @render.text
    def info():
        if data() is not None:
            df = data()
            max_count = len(maximos_data()) if maximos_data() is not None else 0
            min_count = len(minimos_data()) if minimos_data() is not None else 0
            return f"Dados carregados: {len(df)} registros\nPeríodo: {df['Date'].min()} a {df['Date'].max()}\nMáximos encontrados: {max_count}\nMínimos encontrados: {min_count}"
        else:
            return "Nenhum dado carregado. Tentando carregar dados/db_bitcoin_1dia.csv..."
    
    @render.plot
    def price_plot():
        if data() is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'Nenhum dado disponível', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title('Série Temporal - Bitcoin')
            return fig
        
        df = data()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plotar a série temporal
        ax.plot(df['Date'], df['Close'], label='Preço de Fechamento', 
                color='blue', linewidth=1, alpha=0.7)
        
        # Plotar máximos se disponíveis
        if maximos_data() is not None and not maximos_data().empty:
            maximos_df = maximos_data()
            ax.scatter(maximos_df['Date'], maximos_df['Price'], 
                      color='red', s=50, marker='^', 
                      label=f'Máximos ({len(maximos_df)})', zorder=5)
        
        # Plotar mínimos se disponíveis
        if minimos_data() is not None and not minimos_data().empty:
            minimos_df = minimos_data()
            ax.scatter(minimos_df['Date'], minimos_df['Price'], 
                      color='green', s=50, marker='v', 
                      label=f'Mínimos ({len(minimos_df)})', zorder=5)
        
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço (USD)')
        ax.set_title('Série Temporal Bitcoin - Máximos e Mínimos Locais')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotacionar labels do eixo x para melhor legibilidade
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    @render.data_frame
    def maximos_table():
        if maximos_data() is not None and not maximos_data().empty:
            # Formatar a tabela para exibição (ordenada por data - mais recente primeiro)
            display_df = maximos_data().copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df = display_df[['Date', 'Price']].rename(columns={
                'Date': 'Data',
                'Price': 'Preço (USD)'
            })
            
            return render.DataGrid(
                display_df,
                width="100%",
                height="400px",
                summary=f"Pilha de Máximos (LIFO) - {len(display_df)} registros"
            )
        else:
            empty_df = pd.DataFrame({
                'Data': [],
                'Preço (USD)': []
            })
            return render.DataGrid(
                empty_df,
                width="100%",
                height="200px",
                summary="Pilha de Máximos vazia"
            )
    
    @render.data_frame
    def minimos_table():
        if minimos_data() is not None and not minimos_data().empty:
            # Formatar a tabela para exibição (ordenada por data - mais recente primeiro)
            display_df = minimos_data().copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df = display_df[['Date', 'Price']].rename(columns={
                'Date': 'Data',
                'Price': 'Preço (USD)'
            })
            
            return render.DataGrid(
                display_df,
                width="100%",
                height="400px",
                summary=f"Pilha de Mínimos (LIFO) - {len(display_df)} registros"
            )
        else:
            empty_df = pd.DataFrame({
                'Data': [],
                'Preço (USD)': []
            })
            return render.DataGrid(
                empty_df,
                width="100%",
                height="200px",
                summary="Pilha de Mínimos vazia"
            )

app = App(app_ui, server)

