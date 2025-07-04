# Scripts Online Machine Learning (OML)

Este diretório contém scripts Python que demonstram aplicações práticas de **filas** e **pilhas** em **Online Machine Learning**, usando dados reais de Bitcoin.

## 📁 Estrutura dos Arquivos

### 🔄 `janelas_deslizantes_filas.py`
**Implementa janelas deslizantes usando filas (FIFO) para médias móveis.**

- **Classe Principal**: `JanelaDeslizanteFila`
- **Estrutura**: `collections.deque(maxlen=n)` - Fila com tamanho fixo
- **Operações**: O(1) para inserir/remover elementos
- **Aplicação**: Calcular médias móveis de forma eficiente
- **Visualização**: Gráfico com série histórica + médias móveis

### 📚 `detecao_extremos_pilhas.py`
**Implementa detecção de extremos usando pilhas (LIFO) para máximos e mínimos.**

- **Classe Principal**: `PilhaExtremos`
- **Estrutura**: `list` com operações `append()`/`pop()` - Pilha LIFO
- **Algoritmo**: Janela móvel para detectar picos e vales
- **Aplicação**: Identificar máximos e mínimos locais
- **Visualização**: Gráfico com série completa + extremos marcados

### 🎯 `analise_completa.py`
**Combina análises de filas e pilhas para visão integrada.**

- **Funcionalidade**: Executa ambas as análises e gera comparação
- **Visualização**: Subplots separados para cada tipo de análise
- **Relatório**: Estatísticas consolidadas de ambas as estruturas

## 🚀 Como Usar

### Pré-requisitos
```bash
pip install pandas numpy matplotlib
```

### Executar Scripts Individuais

#### 1. Análise com Filas (Médias Móveis)
```bash
python janelas_deslizantes_filas.py
```
- Processa dados de Bitcoin usando 3 janelas deslizantes
- Calcula médias móveis de 15min, 60min e 240min
- Visualiza séries temporais com médias móveis

#### 2. Análise com Pilhas (Extremos)
```bash
python detecao_extremos_pilhas.py
```
- Detecta máximos e mínimos locais nos preços
- Usa janela móvel de 7 períodos para análise
- Visualiza série completa com extremos marcados

#### 3. Análise Completa
```bash
python analise_completa.py
```
- Executa ambas as análises em sequência
- Gera visualização comparativa
- Produz relatório final consolidado

## 📊 Dados Utilizados

- **Arquivo**: `dados/db_bitcoin_1dia.csv`
- **Formato**: Preços de Bitcoin por minuto
- **Período**: 2 dias completos (2880 registros)
- **Colunas**: Date, Close

## 🔧 Configurações Principais

### Filas (Médias Móveis)
```python
configuracoes_janelas = {
    'Curta (15min)': 15,     # Janela de 15 minutos
    'Média (60min)': 60,     # Janela de 1 hora  
    'Longa (240min)': 240    # Janela de 4 horas
}
```

### Pilhas (Extremos)
```python
# Tamanho da janela para detecção de extremos
janela_detecao = 7  # Análise em janela de 7 períodos
```

## 🎓 Conceitos Demonstrados

### 🔄 Filas (FIFO)
- **Estrutura**: First In, First Out
- **Implementação**: `collections.deque` com `maxlen`
- **Vantagem**: Memória constante O(1)
- **Aplicação**: Janelas deslizantes para médias móveis
- **Uso em OML**: Processamento contínuo de dados

### 📚 Pilhas (LIFO)
- **Estrutura**: Last In, First Out
- **Implementação**: `list` com `append`/`pop`
- **Vantagem**: Acesso rápido ao topo O(1)
- **Aplicação**: Detecção de extremos locais
- **Uso em OML**: Identificação de padrões

## 📈 Resultados Esperados

### Filas - Médias Móveis
- Gráfico com preço original e 3 médias móveis
- Suavização de ruído nos dados
- Identificação de tendências

### Pilhas - Extremos
- Gráfico com preço original e extremos marcados
- Pontos de máximo (triângulos vermelhos ▲)
- Pontos de mínimo (triângulos verdes ▼)

### Análise Completa
- Visualização comparativa dos dois métodos
- Relatório com estatísticas consolidadas
- Demonstração da complementaridade das estruturas

## 🎯 Aspectos Educacionais

1. **Estruturas de Dados**: Aplicação prática de filas e pilhas
2. **Complexidade**: Operações O(1) para eficiência
3. **Online ML**: Processamento incremental de dados
4. **Visualização**: Gráficos explicativos dos resultados
5. **Integração**: Combinação de diferentes estruturas

## 💡 Melhorias Futuras

- Adição de mais tipos de médias móveis
- Implementação de outros algoritmos de detecção
- Integração com modelos de machine learning
- Processamento de múltiplas séries temporais
- Interface web interativa

---

**Disciplina**: Estrutura de Dados (GES-115)  
**Autor**: João Paulo Assis Bonifácio  
**Objetivo**: Demonstrar aplicações práticas de estruturas de dados em Online Machine Learning 