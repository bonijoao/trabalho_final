import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def walk_forward_validation_longo_prazo(X, y, n_splits=3, test_size=0.15):
    """
    Validação walk forward para previsões de longo prazo
    Reduzido o número de splits para ter dados suficientes para 60 previsões sequenciais
    """
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    min_train_size = int(n_samples * 0.4)  # Mais dados para treino
    
    results = []
    
    for i in range(n_splits):
        end_test = n_samples - (n_splits - 1 - i) * (test_samples // n_splits)
        start_test = end_test - test_samples // n_splits
        
        # Garantir que temos pelo menos 60 pontos para previsão sequencial
        if end_test - start_test < 60:
            continue
            
        end_train = start_test
        start_train = max(0, end_train - min_train_size - i * (test_samples // n_splits))
        
        if start_train >= end_train or start_test >= end_test:
            continue
            
        X_train = X[start_train:end_train]
        y_train = y[start_train:end_train]
        X_test = X[start_test:end_test]
        y_test = y[start_test:end_test]
        
        results.append({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'fold': i + 1,
            'train_start': start_train,
            'train_end': end_train,
            'test_start': start_test,
            'test_end': end_test
        })
    
    return results

def fazer_previsoes_sequenciais(modelo, X_inicial, n_passos=60):
    """
    Faz previsões sequenciais: usa previsão anterior como input para próxima
    """
    previsoes = []
    X_atual = X_inicial.copy()
    
    for passo in range(n_passos):
        # Prever próximo valor
        pred = modelo.predict(X_atual.reshape(1, -1))[0]
        previsoes.append(pred)
        
        # Atualizar X_atual: shift features e adicionar previsão
        # Assumindo que a última feature é o valor anterior
        X_atual[:-1] = X_atual[1:]  # Shift features
        X_atual[-1] = pred  # Nova previsão como última feature
    
    return np.array(previsoes)

def calcular_metricas_por_horizonte(y_true, y_pred_sequencial):
    """
    Calcula métricas para cada horizonte de previsão (1 min, 2 min, ..., 60 min)
    """
    horizontes = min(len(y_true), len(y_pred_sequencial))
    metricas_horizonte = []
    
    for h in range(1, horizontes + 1):
        if h <= len(y_true) and h <= len(y_pred_sequencial):
            y_t = y_true[:h]
            y_p = y_pred_sequencial[:h]
            
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            
            # Erro percentual médio
            mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100
            
            # Acurácia direcional
            if h > 1:
                y_true_dir = np.sign(np.diff(y_t))
                y_pred_dir = np.sign(np.diff(y_p))
                acuracia_dir = np.mean(y_true_dir == y_pred_dir) * 100
            else:
                acuracia_dir = np.nan
            
            metricas_horizonte.append({
                'horizonte': h,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'acuracia_direcional': acuracia_dir
            })
    
    return metricas_horizonte

def avaliar_previsoes_longo_prazo():
    """
    Avalia previsões de longo prazo para todas as janelas
    """
    arquivos_dados = glob.glob('dados/bitcoin_janela_*min.csv')
    
    if not arquivos_dados:
        print("Nenhum arquivo de dados encontrado. Execute primeiro o script 1_carregar_dados_iniciais.py")
        return
    
    resultados_completos = {}
    
    for arquivo in arquivos_dados:
        tamanho_janela = int(arquivo.split('_')[-1].replace('min.csv', ''))
        
        print(f"\n{'='*80}")
        print(f"AVALIANDO PREVISÕES DE LONGO PRAZO - JANELA {tamanho_janela} MINUTOS")
        print(f"{'='*80}")
        
        df = pd.read_csv(arquivo)
        print(f"Total de amostras: {len(df)}")
        
        X = df.drop('alvo', axis=1).values
        y = df['alvo'].values
        
        # Walk Forward Validation para longo prazo
        folds = walk_forward_validation_longo_prazo(X, y, n_splits=3, test_size=0.2)
        
        resultados_fold = []
        
        for fold_data in folds:
            X_train = fold_data['X_train']
            y_train = fold_data['y_train']
            X_test = fold_data['X_test']
            y_test = fold_data['y_test']
            fold_num = fold_data['fold']
            
            print(f"\nFold {fold_num}: Treino[{fold_data['train_start']}:{fold_data['train_end']}] -> Teste[{fold_data['test_start']}:{fold_data['test_end']}]")
            
            # Treinar modelo
            modelo = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=12,
                min_samples_split=3
            )
            
            modelo.fit(X_train, y_train)
            
            # Fazer múltiplas previsões sequenciais dentro do fold
            metricas_todas_sequencias = []
            
            # Fazer previsões sequenciais a cada 10 pontos para ter várias amostras
            for inicio in range(0, len(X_test) - 60, 10):
                X_inicial = X_test[inicio]
                y_real_sequencia = y_test[inicio:inicio + 60]
                
                # Previsões sequenciais
                y_pred_sequencia = fazer_previsoes_sequenciais(modelo, X_inicial, n_passos=60)
                
                # Calcular métricas por horizonte
                metricas_seq = calcular_metricas_por_horizonte(y_real_sequencia, y_pred_sequencia)
                metricas_todas_sequencias.append(metricas_seq)
            
            # Agregar métricas por horizonte
            if metricas_todas_sequencias:
                metricas_agregadas = []
                for h in range(1, 61):  # 1 a 60 minutos
                    mae_h = []
                    rmse_h = []
                    mape_h = []
                    acur_h = []
                    
                    for seq in metricas_todas_sequencias:
                        if h <= len(seq):
                            mae_h.append(seq[h-1]['mae'])
                            rmse_h.append(seq[h-1]['rmse'])
                            mape_h.append(seq[h-1]['mape'])
                            if not np.isnan(seq[h-1]['acuracia_direcional']):
                                acur_h.append(seq[h-1]['acuracia_direcional'])
                    
                    if mae_h:  # Se temos dados para este horizonte
                        metricas_agregadas.append({
                            'horizonte': h,
                            'mae_mean': np.mean(mae_h),
                            'mae_std': np.std(mae_h),
                            'rmse_mean': np.mean(rmse_h),
                            'rmse_std': np.std(rmse_h),
                            'mape_mean': np.mean(mape_h),
                            'mape_std': np.std(mape_h),
                            'acuracia_mean': np.mean(acur_h) if acur_h else np.nan,
                            'acuracia_std': np.std(acur_h) if acur_h else np.nan,
                            'n_sequencias': len(mae_h)
                        })
                
                resultados_fold.append({
                    'fold': fold_num,
                    'metricas_por_horizonte': metricas_agregadas,
                    'n_sequencias_testadas': len(metricas_todas_sequencias)
                })
                
                print(f"  Sequências testadas: {len(metricas_todas_sequencias)}")
                print(f"  MAE 1 min: {metricas_agregadas[0]['mae_mean']:.2f}")
                print(f"  MAE 30 min: {metricas_agregadas[29]['mae_mean']:.2f}")
                print(f"  MAE 60 min: {metricas_agregadas[59]['mae_mean']:.2f}")
        
        resultados_completos[tamanho_janela] = resultados_fold
        print(f"\nConcluído: Janela {tamanho_janela} min")
    
    # Agregar resultados finais por janela
    print(f"\n{'='*100}")
    print("RESUMO FINAL - PREVISÕES DE LONGO PRAZO")
    print(f"{'='*100}")
    
    resumo_final = {}
    
    for janela, folds in resultados_completos.items():
        print(f"\nJANELA {janela} MINUTOS:")
        
        # Agregar métricas de todos os folds por horizonte
        mae_por_horizonte = []
        mape_por_horizonte = []
        
        for h in range(1, 61):
            mae_h_todos_folds = []
            mape_h_todos_folds = []
            
            for fold in folds:
                metricas_h = fold['metricas_por_horizonte']
                if h <= len(metricas_h):
                    mae_h_todos_folds.append(metricas_h[h-1]['mae_mean'])
                    mape_h_todos_folds.append(metricas_h[h-1]['mape_mean'])
            
            if mae_h_todos_folds:
                mae_por_horizonte.append(np.mean(mae_h_todos_folds))
                mape_por_horizonte.append(np.mean(mape_h_todos_folds))
            else:
                mae_por_horizonte.append(np.nan)
                mape_por_horizonte.append(np.nan)
        
        resumo_final[janela] = {
            'mae_por_horizonte': mae_por_horizonte,
            'mape_por_horizonte': mape_por_horizonte
        }
        
        # Mostrar algumas métricas chave
        print(f"  MAE 1 min:  {mae_por_horizonte[0]:.2f}")
        print(f"  MAE 15 min: {mae_por_horizonte[14]:.2f}")
        print(f"  MAE 30 min: {mae_por_horizonte[29]:.2f}")
        print(f"  MAE 60 min: {mae_por_horizonte[59]:.2f}")
        print(f"  Degradação (60min/1min): {mae_por_horizonte[59]/mae_por_horizonte[0]:.2f}x")
    
    # Encontrar melhor modelo por horizonte
    print(f"\n{'='*80}")
    print("RANKING POR HORIZONTE DE PREVISÃO:")
    print(f"{'='*80}")
    
    horizontes_chave = [1, 5, 15, 30, 45, 60]
    for h in horizontes_chave:
        print(f"\nHORIZONTE {h} MINUTOS:")
        ranking_h = []
        for janela, dados in resumo_final.items():
            mae_h = dados['mae_por_horizonte'][h-1]
            if not np.isnan(mae_h):
                ranking_h.append((janela, mae_h))
        
        ranking_h.sort(key=lambda x: x[1])
        for i, (janela, mae) in enumerate(ranking_h):
            print(f"  {i+1}º. Janela {janela} min: MAE = {mae:.2f}")
    
    # Criar visualizações
    criar_visualizacoes_longo_prazo(resumo_final)
    
    # Salvar resultados
    salvar_resultados_longo_prazo(resumo_final)
    
    return resumo_final

def criar_visualizacoes_longo_prazo(resumo_final):
    """
    Cria visualizações do decaimento da performance ao longo do tempo
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise de Decaimento - Previsões de Longo Prazo (1-60 min)', 
                 fontsize=16, fontweight='bold')
    
    horizontes = list(range(1, 61))
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c']
    janelas = sorted(resumo_final.keys())
    
    # 1. Evolução do MAE ao longo do tempo
    ax1 = axes[0, 0]
    for i, janela in enumerate(janelas):
        mae_valores = resumo_final[janela]['mae_por_horizonte']
        ax1.plot(horizontes, mae_valores, 'o-', label=f'Janela {janela} min', 
                color=cores[i], linewidth=2, markersize=4)
    
    ax1.set_title('Evolução do MAE por Horizonte', fontweight='bold')
    ax1.set_xlabel('Horizonte de Previsão (minutos)')
    ax1.set_ylabel('MAE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Evolução do MAPE ao longo do tempo
    ax2 = axes[0, 1]
    for i, janela in enumerate(janelas):
        mape_valores = resumo_final[janela]['mape_por_horizonte']
        ax2.plot(horizontes, mape_valores, 's-', label=f'Janela {janela} min', 
                color=cores[i], linewidth=2, markersize=4)
    
    ax2.set_title('Evolução do MAPE por Horizonte', fontweight='bold')
    ax2.set_xlabel('Horizonte de Previsão (minutos)')
    ax2.set_ylabel('MAPE (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Degradação relativa (normalizado pelo valor em t=1)
    ax3 = axes[1, 0]
    for i, janela in enumerate(janelas):
        mae_valores = np.array(resumo_final[janela]['mae_por_horizonte'])
        degradacao = mae_valores / mae_valores[0]  # Normalizar pelo primeiro valor
        ax3.plot(horizontes, degradacao, '^-', label=f'Janela {janela} min', 
                color=cores[i], linewidth=2, markersize=4)
    
    ax3.set_title('Degradação Relativa do MAE', fontweight='bold')
    ax3.set_xlabel('Horizonte de Previsão (minutos)')
    ax3.set_ylabel('MAE Relativo (vs. t=1)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # 4. Comparação em horizontes chave
    ax4 = axes[1, 1]
    horizontes_chave = [1, 15, 30, 45, 60]
    x_pos = np.arange(len(horizontes_chave))
    width = 0.25
    
    for i, janela in enumerate(janelas):
        mae_valores = resumo_final[janela]['mae_por_horizonte']
        mae_chave = [mae_valores[h-1] for h in horizontes_chave]
        ax4.bar(x_pos + i*width, mae_chave, width, label=f'Janela {janela} min', 
               color=cores[i], alpha=0.8)
    
    ax4.set_title('MAE em Horizontes Chave', fontweight='bold')
    ax4.set_xlabel('Horizonte de Previsão')
    ax4.set_ylabel('MAE')
    ax4.set_xticks(x_pos + width)
    ax4.set_xticklabels([f'{h} min' for h in horizontes_chave])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for i, janela in enumerate(janelas):
        mae_valores = resumo_final[janela]['mae_por_horizonte']
        mae_chave = [mae_valores[h-1] for h in horizontes_chave]
        for j, v in enumerate(mae_chave):
            ax4.text(j + i*width, v + max(mae_chave)*0.01, f'{v:.1f}', 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('dados/analise_longo_prazo.png', dpi=300, bbox_inches='tight')
    plt.savefig('dados/analise_longo_prazo.pdf', bbox_inches='tight')
    
    print(f"\n📈 Visualizações de longo prazo salvas:")
    print(f"  - dados/analise_longo_prazo.png")
    print(f"  - dados/analise_longo_prazo.pdf")

def salvar_resultados_longo_prazo(resumo_final):
    """
    Salva resultados em CSV para análise posterior
    """
    dados_para_csv = []
    
    for janela, dados in resumo_final.items():
        for h, (mae, mape) in enumerate(zip(dados['mae_por_horizonte'], dados['mape_por_horizonte'])):
            dados_para_csv.append({
                'janela': janela,
                'horizonte': h + 1,
                'mae': mae,
                'mape': mape
            })
    
    df_resultados = pd.DataFrame(dados_para_csv)
    df_resultados.to_csv('dados/resultados_longo_prazo.csv', index=False)
    
    print(f"📊 Resultados detalhados salvos em: dados/resultados_longo_prazo.csv")

if __name__ == "__main__":
    avaliar_previsoes_longo_prazo() 