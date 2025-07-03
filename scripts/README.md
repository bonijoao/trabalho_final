# Scripts de Análise Bitcoin com Estrutura de Fila

## Arquivos Implementados

1. **`estrutura_fila_bitcoin.py`** - Classe FilaDeslizante
2. **`1_carregar_dados_iniciais.py`** - Processamento dos dados
3. **`3_treinar_e_avaliar_modelo.py`** - Treinamento e avaliação
4. **`executar_pipeline_completo.py`** - Execução completa do pipeline

## Execução

### Opção 1: Pipeline Completo

```bash
cd scripts
python executar_pipeline_completo.py
```

### Opção 2: Execução Manual

```bash
cd scripts
python 1_carregar_dados_iniciais.py
python 3_treinar_e_avaliar_modelo.py
```

## Saídas Geradas

- `dados/bitcoin_janela_15min.csv`
- `dados/bitcoin_janela_30min.csv`
- `dados/bitcoin_janela_60min.csv`
- `dados/resultados_avaliacao.csv`
- `dados/comparacao_modelos.png`
- `dados/comparacao_modelos.pdf`

## Estrutura da Análise

O pipeline utiliza uma **Fila Deslizante** para transformar séries temporais em dados tabulares, testando diferentes tamanhos de janela para prever cotações do Bitcoin usando Random Forest.

## Métricas Avaliadas

- **MAE** (Mean Absolute Error): Erro absoluto médio
- **RMSE** (Root Mean Square Error): Raiz do erro quadrático médio
- **R²** (Coefficient of Determination): Coeficiente de determinação

O script gera automaticamente gráficos comparativos das métricas por tamanho de janela.
