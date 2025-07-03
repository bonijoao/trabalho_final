### Plano de Desenvolvimento: Análise de Séries Temporais de Bitcoin com Filas

**Objetivo:** Utilizar a estrutura de dados de Fila para transformar uma série temporal de cotações do Bitcoin em um formato tabular (features e alvo), adequado para treinamento de modelos de Machine Learning (como Random Forest ou XGBoost). O objetivo final é avaliar qual o melhor tamanho de janela (histórico de cotações) para prever a cotação seguinte.

---

#### **Fase 1: A Estrutura de Dados - A Fila Deslizante**

1.  **Arquivo:** `scripts/estrutura_fila_bitcoin.py`
2.  **Objetivo:** Criar uma classe Python para a Fila.
3.  **Ações:**
    - Definir uma classe `FilaDeslizante`.
    - O construtor (`__init__`) deverá receber `tamanho_maximo` como argumento, que definirá o tamanho da nossa janela de observação.
    - Implementar os métodos essenciais de uma fila:
      - `enfileirar(item)`: Adiciona um item ao final da fila. Se a fila já estiver cheia, ele deve primeiro `desenfileirar()` o item mais antigo para dar espaço ao novo, mantendo assim o tamanho da janela.
      - `desenfileirar()`: Remove e retorna o item do início da fila.
      - `esta_cheia()`: Retorna `True` se a fila atingiu seu `tamanho_maximo`.
      - `get_elementos()`: Retorna uma lista com os elementos atuais da fila.

---

#### **Fase 2: Processamento e Transformação dos Dados**

1.  **Arquivo:** `scripts/1_carregar_dados_iniciais.py`
2.  **Objetivo:** Ler os dados brutos e, usando a `FilaDeslizante`, criar o dataset tabular.
3.  **Ações:**
    - Importar a classe `FilaDeslizante`.
    - Ler o arquivo `dados/db_bitcoin_1dia.csv`.
    - Definir uma lista de tamanhos de janela que queremos testar (ex: `[15, 30, 60]` minutos).
    - Para cada tamanho de janela:
      - Instanciar uma `FilaDeslizante` com o tamanho escolhido.
      - Criar uma lista vazia para armazenar os dados tabulares (ex: `dados_formatados`).
      - Iterar sobre as cotações do Bitcoin. Para cada cotação:
        - Primeiro, preencher a fila até que ela esteja cheia.
        - Após a fila estar cheia, para cada nova cotação:
          1.  Os elementos na fila se tornam a nossa linha de features (X).
          2.  A cotação atual (que ainda não entrou na fila) se torna nosso alvo (y).
          3.  Adicionar `(X, y)` à lista `dados_formatados`.
          4.  Enfileirar a cotação atual (o que fará o item mais antigo ser removido).
    - Salvar os `dados_formatados` em um novo arquivo CSV (ex: `dados/bitcoin_janela_60min.csv`), com colunas nomeadas como `t-59`, `t-58`, ..., `t-0`, `alvo`.

---

#### **Fase 3: Treinamento, Previsão e Avaliação do Modelo**

1.  **Arquivo:** (Novo) `scripts/3_treinar_e_avaliar_modelo.py`
2.  **Objetivo:** Usar os dados transformados para treinar um modelo e avaliar sua performance.
3.  **Ações:**
    - Para cada arquivo de janela gerado (`bitcoin_janela_15min.csv`, etc.):
      - Carregar o dataset.
      - Dividir os dados em treino e teste (ex: 80% para treino, 20% para teste). É importante manter a ordem temporal.
      - Escolher um modelo (ex: `RandomForestRegressor`).
      - Treinar o modelo com os dados de treino.
      - Fazer previsões nos dados de teste.
      - Calcular métricas de erro, como Erro Absoluto Médio (MAE) ou Raiz do Erro Quadrático Médio (RMSE).
      - Guardar o resultado da avaliação associado ao tamanho da janela.

---

#### **Fase 4: Conclusão e Relatório**

1.  **Arquivo:** `Relatorio.qmd`
2.  **Objetivo:** Analisar os resultados e documentar o projeto.
3.  **Ações:**
    - Comparar as métricas de erro para cada tamanho de janela testado.
    - Identificar qual tamanho de janela produziu o melhor resultado preditivo.
    - Descrever todo o processo no relatório, destacando como a estrutura de Fila foi fundamental para a modelagem do problema.
