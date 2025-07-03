# 📊 Análise de Bitcoin com Filas e Médias Móveis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Shiny](https://img.shields.io/badge/Shiny-Python-orange.svg)
![Data Structures](https://img.shields.io/badge/Data%20Structures-Queues-green.svg)

**Projeto Final de Estrutura de Dados (GES-115) - Curso de Estatística**

Este projeto demonstra a aplicação prática de **filas** como estruturas de dados sequenciais para otimizar o cálculo de **médias móveis** em dados financeiros de Bitcoin (BTCUSD), implementando um painel interativo em Shiny para Python.

---

## 🎯 **Objetivo**

Demonstrar como estruturas de dados adequadas (especificamente **filas**) podem otimizar significativamente algoritmos de processamento sequencial, comparando a eficiência de diferentes abordagens para cálculo de médias móveis em séries temporais financeiras.

---

## 📚 **Conceitos Fundamentais**

### 🔷 1. **Estrutura de Dados**

* **Definição**: Forma de organizar e armazenar dados para facilitar o acesso e a modificação.
* **Tipos**: Sequenciais (vetores/listas), hierárquicas (árvores), ligadas (listas encadeadas), e assim por diante.

---

### 🔷 2. **Pilhas e Filas**

São **estruturas de dados lineares** que seguem regras específicas de inserção/remoção:

* **Pilha (Stack)**: LIFO (Last In, First Out). Ex: Desfazer ações (Ctrl + Z).
* **Fila (Queue)**: FIFO (First In, First Out). Ex: Impressoras ou filas de atendimento.

---

### 🔷 3. **Filas em Estruturas de Dados Sequenciais**

* Filas podem ser implementadas **em vetores ou listas sequenciais**.
* Isso implica definir dois índices:
  * `inicio`: onde os dados são retirados.
  * `fim`: onde os dados são inseridos.
* Importante gerenciar o deslocamento (fila circular) para eficiência de espaço.

---

### 🔷 4. **Estruturas de Dados Sequenciais**

* Ex: Vetores, Arrays, Listas.
* Acesso rápido por índice.
* Ideais para operações com janelas deslizantes, pois permitem percorrer os elementos de forma eficiente.

---

### 🔷 5. **Janela Deslizante (Sliding Window) em Estrutura de Dados Sequenciais**

* Técnica usada para **percorrer um vetor/lista mantendo um subconjunto (janela)** de elementos visível por vez.
* A janela se move **uma posição por vez**, descartando o primeiro elemento e incluindo o próximo.
* Exemplo:

  ```txt
  Vetor:       [1, 3, 5, 7, 9]
  Tamanho da janela: 3
  Janelas:     [1,3,5] → [3,5,7] → [5,7,9]
  ```

---

### 🔷 6. **Filas e Janelas Deslizantes**

* Uma **fila** (especialmente fila de tamanho fixo) é ideal para **representar uma janela deslizante**:
  * Ao inserir um novo elemento, remove-se o mais antigo.
  * Isso pode ser feito com uma **fila circular** ou **deque (fila dupla)** para maior eficiência.

---

### 🔷 7. **Médias Móveis com Janela Deslizante**

* Aplicação prática da janela deslizante.
* **Média móvel**: média dos últimos `k` elementos de uma série.
* Pode ser calculada eficientemente com estrutura tipo fila:
  * Soma atual da janela → Subtrai o valor que sai, adiciona o novo → Recalcula a média.
* Usado em análise de séries temporais, como em dados financeiros, previsão de temperatura, etc.

---

## 🔗 **Conexão dos Conceitos**

```txt
[Estrutura de Dados]
    → [Pilhas e Filas]
        → [Filas]
            → [Filas em Estruturas de Dados Sequenciais]
                → [Janela Deslizante em Estrutura Sequencial]
                    → [Médias Móveis com Janela Deslizante]
```

---

## 🚀 **Aplicação Prática**

### **Dados**: Bitcoin (BTCUSD)
- Séries históricas de preços
- Análise de tendências com médias móveis
- Visualização interativa em tempo real

### **Otimização Algorítmica**
- **Método Tradicional**: O(n×k) - recalcula toda média a cada novo ponto
- **Método com Fila**: O(1) - atualiza incrementalmente usando janela deslizante

### **Interface Interativa**
- Dashboard em Shiny para Python
- Controles para diferentes períodos de média móvel
- Comparação visual de performance
- Métricas de eficiência em tempo real

---

## 🛠️ **Tecnologias Utilizadas**

- **Python 3.8+**
- **Shiny for Python** - Interface web interativa
- **Collections.deque** - Implementação otimizada de fila
- **Plotly** - Visualizações interativas
- **Pandas** - Manipulação de dados
- **NumPy** - Operações numéricas
- **APIs de Criptomoedas** - Dados em tempo real

---

## 📁 **Estrutura do Projeto**

```
trabalho_final/
├── dados/                    # Datasets de Bitcoin
├── implementacao.py          # Implementação das estruturas de dados
├── ibovespa.py              # Scripts de análise
├── app/                     # Aplicação Shiny
│   ├── ui.py               # Interface do usuário
│   ├── server.py           # Lógica do servidor
│   └── utils.py            # Funções auxiliares
├── Relatorio.qmd           # Relatório em Quarto
├── README.md               # Este arquivo
└── requirements.txt        # Dependências
```

---

## 🔬 **Resultados Esperados**

1. **Demonstração quantitativa** da eficiência de filas vs métodos tradicionais
2. **Análise de complexidade** computacional com benchmarks
3. **Interface funcional** para exploração interativa dos dados
4. **Insights estatísticos** sobre padrões em dados de Bitcoin
5. **Aplicabilidade** em outras áreas da análise quantitativa

---

## 📊 **Métricas de Performance**

- **Tempo de execução**: Comparação entre implementações
- **Uso de memória**: Eficiência das estruturas de dados
- **Escalabilidade**: Performance com diferentes volumes de dados
- **Responsividade**: Interação em tempo real no dashboard

---

## 🎓 **Contexto Acadêmico**

**Disciplina**: Estrutura de Dados (GES-115)  
**Curso**: Estatística - UFLA  
**Período**: 3º Período  
**Foco**: Aplicação prática de estruturas de dados em problemas estatísticos reais

---

## 📈 **Aplicações Futuras**

- Análise de outras criptomoedas
- Indicadores técnicos avançados (MACD, RSI)
- Processamento de dados em streaming
- Algoritmos de trading automatizado
- Análise de séries temporais em outras áreas

---

## 🤝 **Contribuição**

Este projeto demonstra como escolhas adequadas de estruturas de dados podem transformar a eficiência de algoritmos estatísticos, servindo como base para análises quantitativas mais sofisticadas em finanças e outras áreas da estatística. 