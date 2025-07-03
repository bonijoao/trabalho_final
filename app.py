import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Análise Bitcoin com Estruturas de Dados")

# Classe da Fila de Tamanho Fixo (usada em várias seções)
class FixedQueue:
    """Uma fila que mantém um número máximo de elementos."""
    def __init__(self, max_size):
        self.max_size = max_size
        self.elements = deque(maxlen=max_size)
    
    def enqueue(self, item):
        """Adiciona um item. Se a fila estiver cheia, o item mais antigo é removido."""
        self.elements.append(item)
    
    def is_full(self):
        """Verifica se a fila atingiu sua capacidade máxima."""
        return len(self.elements) == self.max_size
    
    def get_elements(self):
        """Retorna os elementos atuais como uma lista."""
        return list(self.elements)

def main():
    """Função Principal da Aplicação Streamlit"""
    st.title("Bitcoin: Análise de Preços com Estruturas de Dados e ML 🪙")

    st.sidebar.title("Navegação")
    pagina = st.sidebar.radio("Selecione uma seção", [
        "O que são Pilhas e Filas?",
        "Preparando os Dados com uma Fila",
        "Treinando o Modelo de Machine Learning",
    ])
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Resetar Estados", help="Limpa cache e reseta todas as estruturas de dados"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if pagina == "O que são Pilhas e Filas?":
        mostrar_pagina_estruturas_dados()
    elif pagina == "Preparando os Dados com uma Fila":
        mostrar_pagina_preparacao_dados()
    elif pagina == "Treinando o Modelo de Machine Learning":
        mostrar_pagina_treinamento_modelo()

def mostrar_pagina_estruturas_dados():
    st.header("O que são Pilhas (Stacks) e Filas (Queues)?")

    st.markdown("""
    Antes de mergulhar na análise de dados, é crucial entender duas das estruturas de dados fundamentais que formam a base do nosso pipeline: a **Pilha (Stack)** e a **Fila (Queue)**.
    Elas organizam dados de maneiras diferentes, com regras específicas para adição e remoção de elementos.
    """)

    # Inicializar o estado da sessão para as visualizações
    if 'pilha' not in st.session_state:
        st.session_state.pilha = [20, 15, 30]
    if 'fila' not in st.session_state:
        st.session_state.fila = deque([20, 15, 30])
    if 'contador' not in st.session_state:
        st.session_state.contador = 31 # Próximo item a ser adicionado

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📚 Pilha (Stack)")
        st.markdown("""
        Uma Pilha funciona no princípio **LIFO (Last-In, First-Out)**, ou "o último a entrar é o primeiro a sair".
        - **`push`**: Adiciona um item no topo.
        - **`pop`**: Remove o item do topo.

        Pense em uma pilha de pratos: você adiciona um prato no topo e, quando precisa de um, remove o que está no topo.
        """)

        # Controles da Pilha
        c1, c2 = st.columns(2)
        if c1.button("Adicionar (push) 🥞", use_container_width=True, key="push_pilha"):
            st.session_state.pilha.append(st.session_state.contador)
            st.session_state.contador += 1
            st.rerun()

        if c2.button("Remover (pop) 🥞", use_container_width=True, disabled=len(st.session_state.pilha) == 0, key="pop_pilha"):
            if st.session_state.pilha:
                st.session_state.pilha.pop()
            st.rerun()

        # Visualização da Pilha
        st.write("**Visualização da Pilha:**")
        if not st.session_state.pilha:
            st.warning("A pilha está vazia!")
        else:
            container_pilha = st.container(border=True)
            for i, item in enumerate(reversed(st.session_state.pilha)):
                label = "<- TOPO" if i == 0 else ""
                container_pilha.info(f"**{item}** {label}")


    with col2:
        st.subheader("🚂 Fila (Queue)")
        st.markdown("""
        Uma Fila funciona no princípio **FIFO (First-In, First-Out)**, ou "o primeiro a entrar é o primeiro a sair".
        - **`enqueue`**: Adiciona um item no final.
        - **`dequeue`**: Remove o item do início.

        Pense em uma fila de supermercado: a primeira pessoa a chegar é a primeira a ser atendida.
        """)

        # Controles da Fila
        c1, c2 = st.columns(2)
        if c1.button("Adicionar (enqueue) 🚶", use_container_width=True, key="enqueue_fila"):
            st.session_state.fila.append(st.session_state.contador)
            st.session_state.contador += 1
            st.rerun()

        if c2.button("Remover (dequeue) 🚶", use_container_width=True, disabled=len(st.session_state.fila) == 0, key="dequeue_fila"):
            if st.session_state.fila:
                st.session_state.fila.popleft()
            st.rerun()

        # Visualização da Fila
        st.write("**Visualização da Fila:**")
        if not st.session_state.fila:
            st.warning("A fila está vazia!")
        else:
            container_fila = st.container(border=True)
            if len(st.session_state.fila) <= 8:  # Se não tem muitos elementos, mostra em colunas
                cols = container_fila.columns(len(st.session_state.fila))
                for i, item in enumerate(st.session_state.fila):
                    label = "INÍCIO" if i == 0 else ("FIM" if i == len(st.session_state.fila)-1 else "")
                    cols[i].info(f"**{item}**\n{label}")
                container_fila.markdown("`[ INÍCIO ] ←------------- FIFO ------------- [ FIM ]`")
            else:  # Se tem muitos elementos, mostra de forma linear
                elementos_str = " → ".join([str(x) for x in st.session_state.fila])
                container_fila.markdown(f"**INÍCIO:** {elementos_str} **:FIM**")
                container_fila.markdown(f"**Tamanho atual:** {len(st.session_state.fila)} elementos")


def mostrar_pagina_preparacao_dados():
    st.header("Preparando os Dados com uma Fila")
    st.markdown("""
    Nosso objetivo é prever o preço futuro do Bitcoin. Modelos de Machine Learning, no entanto, não entendem o tempo de forma nativa.
    Precisamos transformar a série temporal (uma sequência de preços ao longo do tempo) em um formato tabular, onde cada linha contém:
    - Um conjunto de preços passados (as **features**).
    - O preço futuro que queremos prever (o **alvo** ou **target**).

    É aqui que a **Fila de Tamanho Fixo** se torna extremamente útil. Ela funciona como uma "janela deslizante" que passa sobre nossos dados.
    """)

    # 1. Definição da classe
    st.subheader("1. A Estrutura: Fila de Tamanho Fixo (FixedQueue)")
    with st.expander("Clique para ver o código da classe `FixedQueue`"):
        st.code("""
class FixedQueue:
    \"\"\"Uma fila que mantém um número máximo de elementos.\"\"\"
    def __init__(self, max_size):
        self.max_size = max_size
        self.elements = deque(maxlen=max_size)

    def enqueue(self, item):
        \"\"\"Adiciona um item. Se a fila estiver cheia, o item mais antigo é removido.\"\"\"
        self.elements.append(item)

    def is_full(self):
        \"\"\"Verifica se a fila atingiu sua capacidade máxima.\"\"\"
        return len(self.elements) == self.max_size

    def get_elements(self):
        \"\"\"Retorna os elementos atuais como uma lista.\"\"\"
        return list(self.elements)
        """, language="python")

    # 2. Carregando os dados
    st.subheader("2. Carregando os Dados Originais")
    st.markdown("Carregamos um arquivo CSV com as cotações do Bitcoin minuto a minuto.")
    
    @st.cache_data
    def carregar_dados():
        df = pd.read_csv('dados/db_bitcoin_1dia.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df[['Date', 'Close']]

    df_original = carregar_dados()
    st.line_chart(df_original.set_index('Date')['Close'][:500], use_container_width=True)
    st.write(f"Total de cotações de 1 minuto carregadas: `{len(df_original)}`")
    
    # Mostrar uma amostra dos dados originais com timestamps
    st.write("**Amostra dos dados originais:**")
    sample_original = df_original.head(10).copy()
    sample_original['Minuto'] = range(0, len(sample_original))
    st.dataframe(sample_original[['Minuto', 'Date', 'Close']], use_container_width=True)

    # 3. Transformação com a Fila
    st.subheader("3. A Transformação: De Série para Tabela")
    st.markdown("""
    Agora, iteramos sobre cada cotação e a inserimos na nossa Fila. Quando a fila está cheia, temos uma janela completa de dados passados.
    Nesse momento, podemos criar uma linha no nosso novo dataset:
    - As `N-1` cotações na fila se tornam nossas features (ex: `t-10`, `t-9`, ..., `t-1`).
    - A cotação **seguinte** (a que estamos vendo agora) se torna nosso alvo (`t-0`).
    
    **🕐 Importante:** Agora você pode ver exatamente qual minuto cada `t-X` representa no tempo real!
    """)

    window_size = st.slider(
        "Selecione o tamanho da janela (número de minutos passados para prever o próximo):",
        min_value=5, max_value=60, value=15, step=5
    )
    
    # Visualização da janela temporal
    st.markdown("### 🕐 Como funciona a Janela Temporal:")
    if len(df_original) >= window_size:
        exemplo_inicio = 50  # Começar do minuto 50 para ter contexto
        exemplo_dados = df_original.iloc[exemplo_inicio:exemplo_inicio + window_size + 5]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Exemplo com janela de {window_size} minutos:**
            - 📅 **Minutos {exemplo_inicio} a {exemplo_inicio + window_size - 1}**: Features (entradas do modelo)
            - 🎯 **Minuto {exemplo_inicio + window_size}**: Alvo (o que queremos prever)
            """)
            
            # Criar uma tabela visual
            visual_data = []
            for i in range(window_size + 1):
                idx = exemplo_inicio + i
                if i < window_size:
                    tipo = f"Feature (t-{window_size - i})"
                    cor = "🟦"
                else:
                    tipo = "Alvo (t-0)"
                    cor = "🟨"
                
                if idx < len(df_original):
                    visual_data.append({
                        'Tipo': f"{cor} {tipo}",
                        'Minuto': idx,
                        'Timestamp': df_original.iloc[idx]['Date'].strftime('%H:%M:%S'),
                        'Preço': f"${df_original.iloc[idx]['Close']:.2f}"
                    })
            
            st.dataframe(pd.DataFrame(visual_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.info(f"""
            **Legenda:**
            🟦 = Features (dados passados)
            🟨 = Alvo (que queremos prever)
            
            **Interpretação:**
            Usar os preços dos minutos {exemplo_inicio} a {exemplo_inicio + window_size - 1} para prever o preço do minuto {exemplo_inicio + window_size}
            """)

    @st.cache_data(show_spinner=False)
    def transformar_dados_com_tempo(df_dados, window_size):
        queue = FixedQueue(window_size)
        formatted_data = []
        timestamps_info = []
        
        for idx, row in df_dados.iterrows():
            cotacao = row['Close']
            timestamp = row['Date']
            
            if queue.is_full():
                features = queue.get_elements()
                target = cotacao
                
                # Criar informação temporal
                tempo_info = {
                    'minuto_ref': idx,  # Minuto de referência (t-0)
                    'timestamp_t0': timestamp,
                    'minuto_inicio': idx - window_size,  # Onde começou a janela
                }
                
                row_data = features + [target]
                formatted_data.append(row_data)
                timestamps_info.append(tempo_info)
                
            queue.enqueue(cotacao)
        
        # Criar colunas
        feature_cols = [f't-{window_size-i}' for i in range(window_size)]
        target_col = ['t-0']
        all_cols = feature_cols + target_col
        
        df_transformed = pd.DataFrame(formatted_data, columns=all_cols)
        
        # Adicionar informações temporais
        df_timestamps = pd.DataFrame(timestamps_info)
        df_final = pd.concat([df_timestamps, df_transformed], axis=1)
        
        return df_final

    df_final = transformar_dados_com_tempo(df_original, window_size)

    st.write(f"**Dataset Transformado (com janela de {window_size} minutos):**")
    
    # Explicação temporal
    if not df_final.empty:
        st.info(f"""
        **📅 Como ler esta tabela:**
        - **minuto_ref**: O minuto absoluto que representa t-0 (alvo)
        - **timestamp_t0**: Data/hora exata do t-0
        - **minuto_inicio**: Minuto onde a janela começou
        - **t-{window_size} até t-1**: Preços dos {window_size} minutos anteriores ao t-0
        - **t-0**: O preço que queremos prever (alvo)
        """)
        
        # Mostrar as primeiras linhas com explicação
        sample_df = df_final.head(5).copy()
        
        # Adicionar explicação para cada linha
        explicacoes = []
        for idx, row in sample_df.iterrows():
            minuto_ref = int(row['minuto_ref'])
            minuto_inicio = int(row['minuto_inicio'])
            explicacao = f"Janela: min {minuto_inicio} → {minuto_ref} (prediz min {minuto_ref})"
            explicacoes.append(explicacao)
        
        sample_df['📍 Explicação'] = explicacoes
        
        # Reordenar colunas para ficar mais claro
        colunas_tempo = ['minuto_ref', 'timestamp_t0', 'minuto_inicio', '📍 Explicação']
        colunas_dados = [col for col in sample_df.columns if col.startswith('t-')]
        colunas_ordenadas = colunas_tempo + colunas_dados
        
        st.dataframe(sample_df[colunas_ordenadas], use_container_width=True)
        
        st.write(f"**Dimensões do dataset completo:** `{df_final.shape}` (linhas x colunas)")
        
        # Exemplo específico
        if len(df_final) > 0:
            exemplo = df_final.iloc[0]
            st.markdown(f"""
            **🔍 Exemplo da primeira linha:**
            - No minuto **{int(exemplo['minuto_ref'])}** (timestamp: `{exemplo['timestamp_t0']}`)
            - Usamos os preços dos minutos **{int(exemplo['minuto_inicio'])}** até **{int(exemplo['minuto_ref'])-1}**
            - Para prever o preço do minuto **{int(exemplo['minuto_ref'])}** (t-0 = ${exemplo['t-0']:.2f})
            """)
    else:
        st.warning("Não foi possível gerar dados transformados. Verifique o tamanho da janela.")

    # 4. BONUS: Usando uma Pilha para Detectar Extremos
    st.markdown("---")
    st.subheader("4. 🏔️ BONUS: Pilha para Detectar Extremos (Picos e Vales)")
    st.markdown("""
    **💡 Uso Real da Pilha:** Vamos usar uma pilha para detectar e armazenar **picos** (máximos locais) e **vales** (mínimos locais) 
    conforme processamos os dados. Isso adiciona features valiosas para o modelo sem complexidade excessiva!
    
    **🎯 Por que é útil:**
    - Identifica níveis de resistência e suporte
    - Detecta padrões de reversão de tendência
    - Adiciona contexto técnico aos dados
    """)
    
    if not df_final.empty and len(df_original) > 50:
        # Implementação simples da pilha de extremos
        class PilhaExtremos:
            def __init__(self):
                self.picos = []  # Pilha de máximos locais
                self.vales = []  # Pilha de mínimos locais
        
        def detectar_extremos(dados, janela=5):
            """Detecta picos e vales usando uma janela móvel simples"""
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
        
        # Aplicar detecção nos primeiros 200 pontos para visualização
        dados_sample = df_original['Close'].iloc[:200].values
        pilha_extremos, extremos = detectar_extremos(dados_sample)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualizar os extremos detectados
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(dados_sample)), dados_sample, 'b-', alpha=0.7, label='Preço Bitcoin')
            
            # Marcar picos e vales
            for idx, valor, tipo in extremos:
                if tipo == "pico":
                    ax.scatter(idx, valor, color='red', s=50, marker='^', label='Pico' if 'Pico' not in [l.get_label() for l in ax.get_legend().get_texts()] else '')
                else:
                    ax.scatter(idx, valor, color='green', s=50, marker='v', label='Vale' if 'Vale' not in [l.get_label() for l in ax.get_legend().get_texts()] else '')
            
            ax.set_title('Detecção de Extremos com Pilha', fontweight='bold')
            ax.set_xlabel('Minutos')
            ax.set_ylabel('Preço (USD)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📊 Estatísticas da Pilha")
            st.metric("Picos Detectados", len(pilha_extremos.picos))
            st.metric("Vales Detectados", len(pilha_extremos.vales))
            st.metric("Total de Extremos", len(extremos))
            
            if pilha_extremos.picos:
                ultimo_pico = pilha_extremos.picos[-1]
                st.metric("Último Pico", f"${ultimo_pico[1]:.2f}")
            
            if pilha_extremos.vales:
                ultimo_vale = pilha_extremos.vales[-1]
                st.metric("Último Vale", f"${ultimo_vale[1]:.2f}")
        
        # Mostrar como a pilha funciona
        st.markdown("### 🔍 Como a Pilha Funciona:")
        st.markdown("""
        1. **Push (Empilhar):** Quando detectamos um pico/vale, empilhamos na pilha correspondente
        2. **LIFO:** O último extremo detectado fica no topo (mais fácil de acessar)
        3. **Contexto:** Mantemos histórico dos extremos para análise de padrões
        """)
        
        # Mostrar últimos extremos detectados
        if extremos:
            ultimos_extremos = extremos[-5:]  # Últimos 5 extremos
            st.write("**Últimos 5 extremos detectados:**")
            extremos_df = pd.DataFrame(ultimos_extremos, columns=['Minuto', 'Preço', 'Tipo'])
            extremos_df['Preço'] = extremos_df['Preço'].apply(lambda x: f"${x:.2f}")
            st.dataframe(extremos_df, use_container_width=True, hide_index=True)


def mostrar_pagina_treinamento_modelo():
    st.header("Treinando o Modelo de Machine Learning")
    st.markdown("""
    Com nossos dados agora em formato tabular, podemos finalmente treinar um modelo de Machine Learning.
    O processo seguirá uma abordagem simplificada, mas robusta para séries temporais:
    1.  **Divisão Temporal**: Usaremos os primeiros 80% dos dados para treinar o modelo e os 20% mais recentes para testá-lo. Isso simula um cenário real, onde usamos o passado para prever o futuro.
    2.  **Treinamento**: Usaremos um modelo `RandomForestRegressor`, um algoritmo poderoso e versátil.
    3.  **Avaliação**: Faremos previsões no conjunto de teste e as compararemos com os valores reais para medir a performance do modelo.
    """)

    # Reutilizar a lógica de carregamento e transformação
    @st.cache_data
    def carregar_dados():
        df = pd.read_csv('dados/db_bitcoin_1dia.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df[['Date', 'Close']]

    df_dados = carregar_dados()

    st.subheader("1. Configuração do Treinamento")
    window_size = st.slider(
        "Selecione o tamanho da janela (deve ser o mesmo da página anterior):",
        min_value=5, max_value=60, value=15, step=5, key="training_window"
    )

    @st.cache_data(show_spinner=False)
    def transformar_dados_para_modelo(df_dados, window_size):
        queue = FixedQueue(window_size)
        formatted_data = []
        timestamps_info = []
        
        for idx, row in df_dados.iterrows():
            cotacao = row['Close']
            timestamp = row['Date']
            
            if queue.is_full():
                features = queue.get_elements()
                target = cotacao
                
                tempo_info = {
                    'minuto_ref': idx,
                    'timestamp_t0': timestamp,
                }
                
                row_data = features + [target]
                formatted_data.append(row_data)
                timestamps_info.append(tempo_info)
                
            queue.enqueue(cotacao)
        
        feature_cols = [f't-{window_size-i}' for i in range(window_size)]
        target_col = ['t-0']
        all_cols = feature_cols + target_col
        
        df_transformed = pd.DataFrame(formatted_data, columns=all_cols)
        df_timestamps = pd.DataFrame(timestamps_info)
        
        return pd.concat([df_timestamps, df_transformed], axis=1)

    df_model = transformar_dados_para_modelo(df_dados, window_size)
    
    # Separar features e target (apenas colunas t-X)
    feature_columns = [col for col in df_model.columns if col.startswith('t-') and col != 't-0']
    X = df_model[feature_columns].values
    y = df_model['t-0'].values

    # Divisão 80/20
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    st.write(f"Tamanho do dataset completo: `{len(X)}` amostras")
    st.write(f"Amostras de Treino: `{len(X_train)}` (80%)")
    st.write(f"Amostras de Teste: `{len(X_test)}` (20%)")

    st.subheader("2. Treinamento e Avaliação")

    if st.button("▶️ Iniciar Treinamento e Avaliação", use_container_width=True):
        with st.spinner("Treinando o modelo... Isso pode levar alguns segundos."):
            modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

        st.success("Modelo treinado com sucesso!")

        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        y_true_direction = np.sign(np.diff(y_test))
        y_pred_direction = np.sign(np.diff(y_pred))
        acuracia_direcao = np.mean(y_true_direction == y_pred_direction) * 100

        st.subheader("3. Resultados da Avaliação")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE (Erro Absoluto Médio)", f"${mae:.2f}")
        m2.metric("RMSE (Raiz do Erro Quadrático Médio)", f"${rmse:.2f}")
        m3.metric("R² Score", f"{r2:.2%}")
        m4.metric("Acurácia Direcional", f"{acuracia_direcao:.1f}%")

        # Gráfico melhorado
        st.subheader("4. Visualização: Real vs. Previsto")
        
        # Escolher um subset menor para visualização mais clara
        n_pontos = min(200, len(y_test))
        indices = range(n_pontos)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Gráfico principal - comparação direta
            ax1.plot(indices, y_test[:n_pontos], label='Valor Real', color='#1f77b4', linewidth=2)
            ax1.plot(indices, y_pred[:n_pontos], label='Previsão', color='#ff7f0e', linewidth=2, alpha=0.8)
            ax1.set_title('Comparação: Preço Real vs. Previsões do Modelo', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Passo de Tempo (minutos)')
            ax1.set_ylabel('Preço Bitcoin (USD)')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Formatar eixo Y para mostrar valores em formato monetário
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Gráfico de erro/resíduo
            residuos = y_test[:n_pontos] - y_pred[:n_pontos]
            ax2.plot(indices, residuos, color='red', alpha=0.7, linewidth=1)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_title('Resíduos (Erro = Real - Previsto)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Passo de Tempo (minutos)')
            ax2.set_ylabel('Erro (USD)')
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📊 Estatísticas do Gráfico")
            st.metric("Pontos Mostrados", f"{n_pontos}")
            st.metric("Preço Médio Real", f"${np.mean(y_test[:n_pontos]):,.2f}")
            st.metric("Preço Médio Previsto", f"${np.mean(y_pred[:n_pontos]):,.2f}")
            st.metric("Erro Médio", f"${np.mean(np.abs(residuos)):,.2f}")
            
            # Correlação
            correlacao = np.corrcoef(y_test[:n_pontos], y_pred[:n_pontos])[0,1]
            st.metric("Correlação", f"{correlacao:.3f}")
        
        # Scatter plot para mostrar correlação
        st.subheader("5. Gráfico de Dispersão: Real vs. Previsto")
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        
        # Sample para não sobrecarregar o gráfico
        sample_size = min(1000, len(y_test))
        idx_sample = np.random.choice(len(y_test), sample_size, replace=False)
        
        ax_scatter.scatter(y_test[idx_sample], y_pred[idx_sample], alpha=0.6, s=20)
        
        # Linha de perfeita predição
        min_val = min(np.min(y_test[idx_sample]), np.min(y_pred[idx_sample]))
        max_val = max(np.max(y_test[idx_sample]), np.max(y_pred[idx_sample]))
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predição Perfeita')
        
        ax_scatter.set_xlabel('Preço Real (USD)')
        ax_scatter.set_ylabel('Preço Previsto (USD)')
        ax_scatter.set_title('Correlação entre Valores Reais e Previstos')
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        
        # Formatar eixos
        ax_scatter.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax_scatter.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        st.pyplot(fig_scatter)


if __name__ == "__main__":
    main() 