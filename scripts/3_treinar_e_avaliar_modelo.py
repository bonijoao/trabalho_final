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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def walk_forward_validation(X, y, n_splits=5, test_size=0.1):
    """
    Implementa validação walk forward para séries temporais
    """
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    
    # Garante que temos dados suficientes para pelo menos n_splits
    min_train_size = int(n_samples * 0.3)  # Mínimo 30% para treino inicial
    
    results = []
    
    for i in range(n_splits):
        # Calcula índices para esta divisão
        end_test = n_samples - (n_splits - 1 - i) * (test_samples // n_splits)
        start_test = end_test - test_samples // n_splits
        
        # Garante que temos dados suficientes para treino
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

def calcular_metricas_financeiras(y_true, y_pred):
    """
    Calcula métricas específicas para trading/finanças
    """
    # Métricas básicas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Métricas financeiras
    erro_relativo = np.abs((y_true - y_pred) / y_true) * 100
    mape = np.mean(erro_relativo)  # Mean Absolute Percentage Error
    
    # Direção correta (importante para trading)
    y_true_direction = np.sign(y_true)
    y_pred_direction = np.sign(y_pred)
    acuracia_direcao = np.mean(y_true_direction == y_pred_direction) * 100
    
    # Sharpe-like ratio (retorno/risco)
    mean_return = np.mean(y_pred)
    std_return = np.std(y_pred)
    sharpe_like = mean_return / std_return if std_return > 0 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'acuracia_direcao': acuracia_direcao,
        'sharpe_like': sharpe_like,
        'erro_relativo_median': np.median(erro_relativo),
        'erro_relativo_q95': np.percentile(erro_relativo, 95)
    }

def treinar_e_avaliar_modelos():
    arquivos_dados = glob.glob('../dados/bitcoin_janela_*min_buffer_*min.csv')
    
    if not arquivos_dados:
        print("Nenhum arquivo de dados encontrado. Execute primeiro o script 1_carregar_dados_iniciais.py")
        return
    
    resultados_completos = []
    
    for arquivo in arquivos_dados:
        # Extrair parâmetros do nome do arquivo: bitcoin_janela_15min_buffer_5min.csv
        nome_base = os.path.basename(arquivo)
        partes = nome_base.replace('.csv', '').split('_')
        janela_temporal = int(partes[2].replace('min', ''))
        buffer_size = int(partes[4].replace('min', ''))
        num_amostras = janela_temporal // buffer_size
        
        print(f"\n{'='*70}")
        print(f"AVALIANDO JANELA TEMPORAL: {janela_temporal} min")
        print(f"BUFFER: {buffer_size} min → {num_amostras} amostras")
        sequencia = [f't-{janela_temporal-i*buffer_size}' for i in range(num_amostras)]
        print(f"SEQUÊNCIA: {sequencia} → t-0")
        print(f"{'='*70}")
        
        df = pd.read_csv(arquivo)
        print(f"Total de amostras: {len(df)}")
        
        X = df.drop('t-0', axis=1).values  # Mudança: coluna alvo agora é 't-0'
        y = df['t-0'].values
        
        # Walk Forward Validation
        folds = walk_forward_validation(X, y, n_splits=5, test_size=0.2)
        
        metricas_fold = []
        predicoes_fold = []
        
        for fold_data in folds:
            X_train = fold_data['X_train']
            y_train = fold_data['y_train']
            X_test = fold_data['X_test']
            y_test = fold_data['y_test']
            fold_num = fold_data['fold']
            
            print(f"\nFold {fold_num}: Treino[{fold_data['train_start']}:{fold_data['train_end']}] -> Teste[{fold_data['test_start']}:{fold_data['test_end']}]")
            print(f"  Amostras treino: {len(X_train)} | Amostras teste: {len(X_test)}")
            
            # Treinar modelo
            modelo = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=10,
                min_samples_split=5
            )
            
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            
            # Calcular métricas
            metricas = calcular_metricas_financeiras(y_test, y_pred)
            metricas['fold'] = fold_num
            metricas['janela_temporal'] = janela_temporal
            metricas['buffer_size'] = buffer_size
            metricas['num_amostras'] = num_amostras
            metricas['n_train'] = len(X_train)
            metricas['n_test'] = len(X_test)
            
            metricas_fold.append(metricas)
            
            # Guardar predições para análise
            predicoes_fold.append({
                'fold': fold_num,
                'y_true': y_test,
                'y_pred': y_pred,
                'test_start': fold_data['test_start'],
                'test_end': fold_data['test_end']
            })
            
            print(f"  MAE: {metricas['mae']:.2f} | RMSE: {metricas['rmse']:.2f} | R²: {metricas['r2']:.4f}")
            print(f"  MAPE: {metricas['mape']:.2f}% | Acurácia Direção: {metricas['acuracia_direcao']:.1f}%")
        
        # Calcular métricas médias
        df_metricas = pd.DataFrame(metricas_fold)
        
        resultado_medio = {
            'janela_temporal': janela_temporal,
            'buffer_size': buffer_size,
            'num_amostras': num_amostras,
            'mae_mean': df_metricas['mae'].mean(),
            'mae_std': df_metricas['mae'].std(),
            'rmse_mean': df_metricas['rmse'].mean(),
            'rmse_std': df_metricas['rmse'].std(),
            'r2_mean': df_metricas['r2'].mean(),
            'r2_std': df_metricas['r2'].std(),
            'mape_mean': df_metricas['mape'].mean(),
            'mape_std': df_metricas['mape'].std(),
            'acuracia_direcao_mean': df_metricas['acuracia_direcao'].mean(),
            'acuracia_direcao_std': df_metricas['acuracia_direcao'].std(),
            'sharpe_like_mean': df_metricas['sharpe_like'].mean(),
            'n_folds': len(folds),
            'total_samples': len(df)
        }
        
        resultados_completos.append({
            'resultado_medio': resultado_medio,
            'metricas_fold': metricas_fold,
            'predicoes': predicoes_fold
        })
        
        print(f"\nRESUMO JANELA {janela_temporal}min (buffer {buffer_size}min, {num_amostras} amostras):")
        print(f"  MAE: {resultado_medio['mae_mean']:.2f} ± {resultado_medio['mae_std']:.2f}")
        print(f"  MAPE: {resultado_medio['mape_mean']:.2f}% ± {resultado_medio['mape_std']:.2f}%")
        print(f"  Acurácia Direção: {resultado_medio['acuracia_direcao_mean']:.1f}% ± {resultado_medio['acuracia_direcao_std']:.1f}%")
        print(f"  R²: {resultado_medio['r2_mean']:.4f} ± {resultado_medio['r2_std']:.4f}")
    
    # Criar DataFrame com resultados médios para ranking
    df_ranking = pd.DataFrame([r['resultado_medio'] for r in resultados_completos])
    df_ranking = df_ranking.sort_values('mae_mean')
    
    print(f"\n{'='*120}")
    print("RANKING FINAL DOS MODELOS (Walk Forward Validation)")
    print(f"{'='*120}")
    print(f"{'Rank':<4} {'Janela':<8} {'Buffer':<8} {'Amostras':<8} {'MAE':<15} {'MAPE (%)':<12} {'Acur.Dir (%)':<13} {'R²':<12} {'Folds':<6}")
    print("-" * 120)
    
    for i, (_, row) in enumerate(df_ranking.iterrows()):
        mae_str = f"{row['mae_mean']:.2f}±{row['mae_std']:.2f}"
        mape_str = f"{row['mape_mean']:.1f}±{row['mape_std']:.1f}"
        acur_str = f"{row['acuracia_direcao_mean']:.1f}±{row['acuracia_direcao_std']:.1f}"
        r2_str = f"{row['r2_mean']:.3f}±{row['r2_std']:.3f}"
        
        print(f"{i+1:<4} {int(row['janela_temporal']):<8} {int(row['buffer_size']):<8} {int(row['num_amostras']):<8} {mae_str:<15} {mape_str:<12} {acur_str:<13} {r2_str:<12} {int(row['n_folds']):<6}")
    
    melhor = df_ranking.iloc[0]
    print(f"\n🏆 MELHOR MODELO:")
    print(f"   Janela Temporal: {int(melhor['janela_temporal'])} min")
    print(f"   Buffer: {int(melhor['buffer_size'])} min")
    print(f"   Amostras: {int(melhor['num_amostras'])}")
    sequencia_melhor = [f't-{int(melhor["janela_temporal"])-i*int(melhor["buffer_size"])}' for i in range(int(melhor['num_amostras']))]
    print(f"   Sequência: {sequencia_melhor} → t-0")
    print(f"   MAE: {melhor['mae_mean']:.2f} ± {melhor['mae_std']:.2f}")
    print(f"   Acurácia Direcional: {melhor['acuracia_direcao_mean']:.1f}% ± {melhor['acuracia_direcao_std']:.1f}%")
    
    # Salvar resultados
    df_ranking.to_csv('../dados/resultados_avaliacao.csv', index=False)
    print(f"\n📊 Resultados salvos em: ../dados/resultados_avaliacao.csv")
    
    # Criar visualizações avançadas
    criar_visualizacoes_avancadas(resultados_completos, df_ranking)
    
    return resultados_completos

def criar_visualizacoes_avancadas(resultados_completos, df_ranking):
    """
    Cria visualizações muito mais informativas e relevantes
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Configuração geral
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Comparação de Métricas Principais (com barras de erro)
    ax1 = fig.add_subplot(gs[0, :2])
    janelas = df_ranking['janela_temporal'].values
    mae_means = df_ranking['mae_mean'].values
    mae_stds = df_ranking['mae_std'].values
    
    bars = ax1.bar(janelas, mae_means, yerr=mae_stds, capsize=5, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax1.set_title('MAE por Janela (com Desvio Padrão)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tamanho da Janela (min)')
    ax1.set_ylabel('MAE')
    ax1.grid(True, alpha=0.3)
    
    # Adicionar valores
    for i, (bar, mean, std) in enumerate(zip(bars, mae_means, mae_stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + max(mae_means)*0.01,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Acurácia Direcional
    ax2 = fig.add_subplot(gs[0, 2:])
    acur_means = df_ranking['acuracia_direcao_mean'].values
    acur_stds = df_ranking['acuracia_direcao_std'].values
    
    bars2 = ax2.bar(janelas, acur_means, yerr=acur_stds, capsize=5,
                    color=['#d62728', '#9467bd', '#8c564b'], alpha=0.8)
    ax2.set_title('Acurácia Direcional por Janela', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tamanho da Janela (min)')
    ax2.set_ylabel('Acurácia (%)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Acaso (50%)')
    ax2.legend()
    
    for i, (bar, mean, std) in enumerate(zip(bars2, acur_means, acur_stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Heatmap de Performance por Fold
    ax3 = fig.add_subplot(gs[1, :2])
    
    # Criar matriz para heatmap
    heatmap_data = []
    janelas_ordenadas = sorted([r['resultado_medio']['janela_temporal'] for r in resultados_completos])
    
    for janela in janelas_ordenadas:
        resultado = next(r for r in resultados_completos if r['resultado_medio']['janela_temporal'] == janela)
        fold_maes = [m['mae'] for m in resultado['metricas_fold']]
        # Pad com NaN se necessário
        while len(fold_maes) < 5:
            fold_maes.append(np.nan)
        heatmap_data.append(fold_maes)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax3.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    ax3.set_title('MAE por Fold e Janela', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Janela (min)')
    ax3.set_xticks(range(5))
    ax3.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax3.set_yticks(range(len(janelas_ordenadas)))
    ax3.set_yticklabels([f'{int(j)} min' for j in janelas_ordenadas])
    
    # Adicionar valores no heatmap
    for i in range(len(janelas_ordenadas)):
        for j in range(5):
            if not np.isnan(heatmap_data[i, j]):
                ax3.text(j, i, f'{heatmap_data[i, j]:.1f}', 
                        ha='center', va='center', fontweight='bold')
    
    plt.colorbar(im, ax=ax3, label='MAE')
    
    # 4. Distribuição de Erros
    ax4 = fig.add_subplot(gs[1, 2:])
    
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, resultado in enumerate(resultados_completos):
        janela = resultado['resultado_medio']['janela_temporal']
        todos_erros = []
        
        for pred_data in resultado['predicoes']:
            erros_relativos = np.abs((pred_data['y_true'] - pred_data['y_pred']) / pred_data['y_true']) * 100
            todos_erros.extend(erros_relativos)
        
        ax4.hist(todos_erros, bins=30, alpha=0.6, label=f'{int(janela)} min', 
                color=cores[i], density=True)
    
    ax4.set_title('Distribuição do Erro Relativo (%)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Erro Relativo (%)')
    ax4.set_ylabel('Densidade')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance Temporal (exemplo com melhor modelo)
    ax5 = fig.add_subplot(gs[2, :])
    
    melhor_janela = df_ranking.iloc[0]['janela_temporal']
    melhor_resultado = next(r for r in resultados_completos if r['resultado_medio']['janela_temporal'] == melhor_janela)
    
    todos_true = []
    todos_pred = []
    indices_temporais = []
    
    for pred_data in melhor_resultado['predicoes']:
        todos_true.extend(pred_data['y_true'])
        todos_pred.extend(pred_data['y_pred'])
        indices_temporais.extend(range(pred_data['test_start'], pred_data['test_end']))
    
    # Subsampling para visualização (pegar 1 a cada 10 pontos)
    step = max(1, len(todos_true) // 200)
    
    ax5.plot(indices_temporais[::step], todos_true[::step], 'b-', alpha=0.7, label='Real', linewidth=1)
    ax5.plot(indices_temporais[::step], todos_pred[::step], 'r-', alpha=0.7, label='Predito', linewidth=1)
    
    ax5.set_title(f'Série Temporal - Melhor Modelo (Janela {int(melhor_janela)} min)', 
                 fontsize=14, fontweight='bold')
    ax5.set_xlabel('Índice Temporal')
    ax5.set_ylabel('Valor Target')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Scatter Plot Real vs Predito
    ax6 = fig.add_subplot(gs[3, :2])
    
    for i, resultado in enumerate(resultados_completos):
        janela = resultado['resultado_medio']['janela_temporal']
        todos_true = []
        todos_pred = []
        
        for pred_data in resultado['predicoes']:
            todos_true.extend(pred_data['y_true'])
            todos_pred.extend(pred_data['y_pred'])
        
        # Subsampling
        step = max(1, len(todos_true) // 100)
        ax6.scatter(todos_true[::step], todos_pred[::step], alpha=0.6, 
                   label=f'{int(janela)} min', s=20)
    
    # Linha perfeita
    min_val = min([min(todos_true) for resultado in resultados_completos for pred_data in resultado['predicoes'] for todos_true in [pred_data['y_true']]])
    max_val = max([max(todos_true) for resultado in resultados_completos for pred_data in resultado['predicoes'] for todos_true in [pred_data['y_true']]])
    
    ax6.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Predição Perfeita')
    ax6.set_title('Real vs Predito', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Valor Real')
    ax6.set_ylabel('Valor Predito')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Resumo Estatístico
    ax7 = fig.add_subplot(gs[3, 2:])
    ax7.axis('off')
    
    # Criar tabela de resumo
    texto_resumo = "RESUMO ESTATÍSTICO\n" + "="*50 + "\n\n"
    
    for i, (_, row) in enumerate(df_ranking.iterrows()):
        janela = int(row['janela_temporal'])
        texto_resumo += f"JANELA {janela} MINUTOS:\n"
        texto_resumo += f"  • MAE: {row['mae_mean']:.2f} ± {row['mae_std']:.2f}\n"
        texto_resumo += f"  • MAPE: {row['mape_mean']:.1f}% ± {row['mape_std']:.1f}%\n"
        texto_resumo += f"  • Acurácia Dir.: {row['acuracia_direcao_mean']:.1f}% ± {row['acuracia_direcao_std']:.1f}%\n"
        texto_resumo += f"  • R²: {row['r2_mean']:.3f} ± {row['r2_std']:.3f}\n"
        texto_resumo += f"  • Folds: {int(row['n_folds'])}\n\n"
    
    melhor = df_ranking.iloc[0]
    texto_resumo += f"🏆 MELHOR: {int(melhor['janela_temporal'])} min\n"
    texto_resumo += f"Supera acaso em {melhor['acuracia_direcao_mean']-50:.1f}pp"
    
    ax7.text(0.05, 0.95, texto_resumo, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Análise Completa de Performance - Walk Forward Validation', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Salvar
    plt.savefig('dados/analise_completa_modelos.png', dpi=300, bbox_inches='tight')
    plt.savefig('dados/analise_completa_modelos.pdf', bbox_inches='tight')
    
    print(f"\n📈 Visualizações avançadas salvas em:")
    print(f"  - dados/analise_completa_modelos.png")
    print(f"  - dados/analise_completa_modelos.pdf")

if __name__ == "__main__":
    treinar_e_avaliar_modelos()