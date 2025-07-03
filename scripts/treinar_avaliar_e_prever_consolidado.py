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

def walk_forward_validation(X, y, n_splits=5, test_size=0.15):
    """Validação walk forward para séries temporais"""
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    min_train_size = int(n_samples * 0.3)
    
    results = []
    for i in range(n_splits):
        end_test = n_samples - (n_splits - 1 - i) * (test_samples // n_splits)
        start_test = end_test - test_samples // n_splits
        end_train = start_test
        start_train = max(0, end_train - min_train_size - i * (test_samples // n_splits))
        
        if start_train >= end_train or start_test >= end_test:
            continue
            
        results.append({
            'X_train': X[start_train:end_train], 'y_train': y[start_train:end_train],
            'X_test': X[start_test:end_test], 'y_test': y[start_test:end_test],
            'fold': i + 1
        })
    return results

def calcular_metricas(y_true, y_pred):
    """Calcula métricas de performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Acurácia direcional
    if len(y_true) > 1:
        y_true_direction = np.sign(y_true[1:] - y_true[:-1])
        y_pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
        acuracia_direcao = np.mean(y_true_direction == y_pred_direction) * 100
    else:
        acuracia_direcao = 50.0
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'acuracia_direcao': acuracia_direcao}

def fazer_previsoes_sequenciais(modelo, X_inicial, n_passos):
    """Faz previsões sequenciais"""
    previsoes = []
    X_atual = X_inicial.copy()
    
    for passo in range(n_passos):
        pred = modelo.predict(X_atual.reshape(1, -1))[0]
        previsoes.append(pred)
        X_atual = np.roll(X_atual, -1)
        X_atual[-1] = pred
    
    return np.array(previsoes)

def avaliar_janela_temporal(arquivo_dados):
    """Avalia uma janela temporal específica"""
    nome_base = os.path.basename(arquivo_dados)
    if 'janela_15min' in nome_base:
        janela_min = 15
    elif 'janela_30min' in nome_base:
        janela_min = 30
    elif 'janela_60min' in nome_base:
        janela_min = 60
    else:
        return None
    
    print(f"\n{'='*80}")
    print(f"AVALIANDO JANELA TEMPORAL: {janela_min} MINUTOS")
    print(f"{'='*80}")
    
    df = pd.read_csv(arquivo_dados)
    print(f"Total de amostras: {len(df)}")
    
    X = df.drop('t-0', axis=1).values
    y = df['t-0'].values
    
    # Avaliação básica
    print(f"\n🔄 AVALIAÇÃO BÁSICA")
    folds = walk_forward_validation(X, y, n_splits=5, test_size=0.2)
    
    metricas_folds = []
    for fold_data in folds:
        modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5)
        modelo.fit(fold_data['X_train'], fold_data['y_train'])
        y_pred = modelo.predict(fold_data['X_test'])
        
        metricas = calcular_metricas(fold_data['y_test'], y_pred)
        metricas['fold'] = fold_data['fold']
        metricas_folds.append(metricas)
        
        print(f"  Fold {fold_data['fold']}: MAE={metricas['mae']:.2f}, Acur.Dir={metricas['acuracia_direcao']:.1f}%")
    
    df_metricas = pd.DataFrame(metricas_folds)
    resumo_basico = {
        'janela_min': janela_min,
        'mae_mean': df_metricas['mae'].mean(),
        'mae_std': df_metricas['mae'].std(),
        'mape_mean': df_metricas['mape'].mean(),
        'acuracia_direcao_mean': df_metricas['acuracia_direcao'].mean(),
        'r2_mean': df_metricas['r2'].mean(),
        'n_folds': len(folds)
    }
    
    print(f"\n📊 RESUMO BÁSICO:")
    print(f"  MAE: {resumo_basico['mae_mean']:.2f} ± {resumo_basico['mae_std']:.2f}")
    print(f"  MAPE: {resumo_basico['mape_mean']:.1f}%")
    print(f"  Acurácia Direcional: {resumo_basico['acuracia_direcao_mean']:.1f}%")
    
    # Previsões de longo prazo
    print(f"\n🚀 PREVISÕES DE LONGO PRAZO")
    horizontes = [60, 120, 180]
    resultados_longo_prazo = {}
    
    split_idx = int(len(X) * 0.7)
    X_train_lp = X[:split_idx]
    y_train_lp = y[:split_idx]
    X_test_lp = X[split_idx:]
    y_test_lp = y[split_idx:]
    
    modelo_final = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, max_depth=12, min_samples_split=3)
    modelo_final.fit(X_train_lp, y_train_lp)
    
    for horizonte in horizontes:
        print(f"\n  🎯 Horizonte: {horizonte} minutos")
        
        mae_horizonte = []
        step = max(1, len(X_test_lp) // 20)
        
        for i in range(0, len(X_test_lp) - horizonte, step):
            X_inicial = X_test_lp[i]
            y_real_seq = y_test_lp[i:i + horizonte]
            
            if len(y_real_seq) < horizonte:
                continue
                
            y_pred_seq = fazer_previsoes_sequenciais(modelo_final, X_inicial, horizonte)
            mae_seq = mean_absolute_error(y_real_seq, y_pred_seq)
            mae_horizonte.append(mae_seq)
        
        if mae_horizonte:
            mae_mean = np.mean(mae_horizonte)
            mae_std = np.std(mae_horizonte)
            degradacao = mae_mean / resumo_basico['mae_mean']
            
            resultados_longo_prazo[horizonte] = {
                'mae_mean': mae_mean,
                'mae_std': mae_std,
                'degradacao': degradacao,
                'n_testes': len(mae_horizonte)
            }
            
            print(f"    MAE: {mae_mean:.2f} ± {mae_std:.2f}")
            print(f"    Degradação: {degradacao:.1f}x vs. básico")
            print(f"    Testes realizados: {len(mae_horizonte)}")
    
    return {
        'resumo_basico': resumo_basico,
        'longo_prazo': resultados_longo_prazo,
        'metricas_folds': metricas_folds
    }

def criar_visualizacoes_consolidadas(resultados_todas_janelas):
    """Cria visualizações consolidadas"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análise Consolidada - Avaliação e Previsões de Longo Prazo', fontsize=16, fontweight='bold')
    
    janelas = sorted(resultados_todas_janelas.keys())
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. MAE Básico
    ax1 = axes[0, 0]
    mae_basico = [resultados_todas_janelas[j]['resumo_basico']['mae_mean'] for j in janelas]
    mae_std = [resultados_todas_janelas[j]['resumo_basico']['mae_std'] for j in janelas]
    bars = ax1.bar(janelas, mae_basico, yerr=mae_std, capsize=5, color=cores, alpha=0.8)
    ax1.set_title('MAE - Avaliação Básica', fontweight='bold')
    ax1.set_xlabel('Janela (min)')
    ax1.set_ylabel('MAE')
    ax1.grid(True, alpha=0.3)
    
    # 2. Acurácia Direcional
    ax2 = axes[0, 1]
    acur_dir = [resultados_todas_janelas[j]['resumo_basico']['acuracia_direcao_mean'] for j in janelas]
    bars2 = ax2.bar(janelas, acur_dir, color=cores, alpha=0.8)
    ax2.set_title('Acurácia Direcional', fontweight='bold')
    ax2.set_xlabel('Janela (min)')
    ax2.set_ylabel('Acurácia (%)')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Acaso (50%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. MAPE
    ax3 = axes[0, 2]
    mape_values = [resultados_todas_janelas[j]['resumo_basico']['mape_mean'] for j in janelas]
    bars3 = ax3.bar(janelas, mape_values, color=cores, alpha=0.8)
    ax3.set_title('MAPE (%)', fontweight='bold')
    ax3.set_xlabel('Janela (min)')
    ax3.set_ylabel('MAPE (%)')
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Previsões de Longo Prazo
    horizontes = [60, 120, 180]
    for idx, horizonte in enumerate(horizontes):
        ax = axes[1, idx]
        
        mae_horizonte = []
        degradacao_horizonte = []
        
        for janela in janelas:
            if horizonte in resultados_todas_janelas[janela]['longo_prazo']:
                mae_h = resultados_todas_janelas[janela]['longo_prazo'][horizonte]['mae_mean']
                deg_h = resultados_todas_janelas[janela]['longo_prazo'][horizonte]['degradacao']
                mae_horizonte.append(mae_h)
                degradacao_horizonte.append(deg_h)
            else:
                mae_horizonte.append(0)
                degradacao_horizonte.append(0)
        
        bars_h = ax.bar(janelas, mae_horizonte, color=cores, alpha=0.8)
        ax.set_title(f'MAE - Horizonte {horizonte} min', fontweight='bold')
        ax.set_xlabel('Janela (min)')
        ax.set_ylabel('MAE')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dados/analise_consolidada.png', dpi=300, bbox_inches='tight')
    plt.savefig('dados/analise_consolidada.pdf', bbox_inches='tight')

def main():
    """Função principal"""
    print("🚀 INICIANDO ANÁLISE CONSOLIDADA")
    print("="*80)
    
    arquivos_dados = glob.glob('dados/bitcoin_janela_*min.csv')
    
    if not arquivos_dados:
        print("❌ Nenhum arquivo de dados encontrado!")
        return
    
    print(f"📁 Encontrados {len(arquivos_dados)} arquivos de dados")
    
    resultados_todas_janelas = {}
    for arquivo in arquivos_dados:
        resultado = avaliar_janela_temporal(arquivo)
        if resultado:
            janela = resultado['resumo_basico']['janela_min']
            resultados_todas_janelas[janela] = resultado
    
    # Ranking final
    print(f"\n{'='*100}")
    print("🏆 RANKING FINAL CONSOLIDADO")
    print(f"{'='*100}")
    
    dados_ranking = []
    for janela, resultados in resultados_todas_janelas.items():
        basico = resultados['resumo_basico']
        longo_prazo = resultados['longo_prazo']
        
        linha = {
            'janela': janela,
            'mae_basico': basico['mae_mean'],
            'acur_dir': basico['acuracia_direcao_mean'],
            'mape': basico['mape_mean']
        }
        
        for horizonte in [60, 120, 180]:
            if horizonte in longo_prazo:
                linha[f'mae_{horizonte}min'] = longo_prazo[horizonte]['mae_mean']
                linha[f'degradacao_{horizonte}min'] = longo_prazo[horizonte]['degradacao']
            else:
                linha[f'mae_{horizonte}min'] = np.nan
                linha[f'degradacao_{horizonte}min'] = np.nan
        
        dados_ranking.append(linha)
    
    df_ranking = pd.DataFrame(dados_ranking)
    df_ranking = df_ranking.sort_values('mae_basico')
    
    print(f"{'Janela':<8} {'MAE Básico':<12} {'Acur.Dir':<10} {'MAPE':<8} {'MAE 60min':<12} {'MAE 120min':<12} {'MAE 180min':<12}")
    print("-" * 100)
    
    for _, row in df_ranking.iterrows():
        janela = f"{int(row['janela'])} min"
        mae_basico = f"{row['mae_basico']:.2f}"
        acur_dir = f"{row['acur_dir']:.1f}%"
        mape = f"{row['mape']:.1f}%"
        
        mae_60 = f"{row['mae_60min']:.1f}" if not pd.isna(row['mae_60min']) else "N/A"
        mae_120 = f"{row['mae_120min']:.1f}" if not pd.isna(row['mae_120min']) else "N/A"
        mae_180 = f"{row['mae_180min']:.1f}" if not pd.isna(row['mae_180min']) else "N/A"
        
        print(f"{janela:<8} {mae_basico:<12} {acur_dir:<10} {mape:<8} {mae_60:<12} {mae_120:<12} {mae_180:<12}")
    
    melhor = df_ranking.iloc[0]
    print(f"\n🏆 MELHOR MODELO GERAL:")
    print(f"   Janela: {int(melhor['janela'])} minutos")
    print(f"   MAE Básico: {melhor['mae_basico']:.2f}")
    print(f"   Acurácia Direcional: {melhor['acur_dir']:.1f}%")
    
    df_ranking.to_csv('../dados/ranking_consolidado.csv', index=False)
    
    print(f"\n📊 Gerando visualizações...")
    criar_visualizacoes_consolidadas(resultados_todas_janelas)
    
    print(f"\n✅ ANÁLISE CONCLUÍDA!")
    print(f"📄 Resultados salvos:")
    print(f"  - dados/ranking_consolidado.csv")
    print(f"  - dados/analise_consolidada.png")
    print(f"  - dados/analise_consolidada.pdf")

if __name__ == "__main__":
    main() 