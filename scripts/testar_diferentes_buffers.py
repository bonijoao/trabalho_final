#!/usr/bin/env python3
"""
Teste de diferentes configurações de buffer da pilha
Mostra como o tamanho do buffer afeta o espaçamento da amostragem
"""

from estrutura_fila_bitcoin import FilaDeslizante, PilhaBuffer

def demonstrar_buffer(dados, tamanho_fila, tamanho_buffer):
    """Demonstra como funciona um buffer específico"""
    print(f"\n{'='*60}")
    print(f"BUFFER DE PILHA: TAMANHO {tamanho_buffer}")
    print(f"FILA DESLIZANTE: TAMANHO {tamanho_fila}")
    print(f"{'='*60}")
    
    fila = FilaDeslizante(tamanho_fila)
    pilha_buffer = PilhaBuffer(tamanho_buffer)
    dados_processados = []
    
    print(f"Dados originais: {dados}")
    print("\nProcessamento passo a passo:")
    
    for i, valor in enumerate(dados):
        # Estado antes do processamento
        pilha_antes = pilha_buffer.pilha.copy()
        fila_antes = fila.get_elementos()
        
        # Processar
        valor_liberado = pilha_buffer.empilhar(valor)
        
        # Estado da pilha após empilhar
        pilha_depois = pilha_buffer.pilha.copy()
        
        print(f"\n  Passo {i+1}: Valor {valor}")
        print(f"    Pilha antes: {pilha_antes} -> Pilha depois: {pilha_depois}")
        
        if valor_liberado is not None:
            print(f"    ✅ PILHA CHEIA! Libera valor: {valor_liberado}")
            
            # Processar na fila
            if not fila.esta_cheia():
                fila.enfileirar(valor_liberado)
            else:
                elementos_atuais = fila.get_elementos()
                alvo = valor_liberado
                
                # Criar janela
                janela = elementos_atuais + [alvo]
                dados_processados.append(janela)
                print(f"    🎯 JANELA CRIADA: {janela}")
                
                fila.enfileirar(valor_liberado)
            
            print(f"    Fila agora: {fila.get_elementos()}")
        else:
            print(f"    ⏳ Pilha ainda não está cheia")
    
    print(f"\n📊 RESULTADO FINAL:")
    print(f"   Total de dados originais: {len(dados)}")
    print(f"   Total de janelas geradas: {len(dados_processados)}")
    print(f"   Taxa de amostragem: {len(dados_processados)}/{len(dados)} = {len(dados_processados)/len(dados)*100:.1f}%")
    print(f"   Espaçamento efetivo: 1 janela a cada {tamanho_buffer} dados originais")
    
    if dados_processados:
        print(f"\n   Primeiras janelas: {dados_processados[:3]}")
        if len(dados_processados) > 3:
            print(f"   Últimas janelas: {dados_processados[-2:]}")
    
    return dados_processados

def main():
    """Função principal"""
    print("🧪 TESTE DE DIFERENTES CONFIGURAÇÕES DE BUFFER")
    
    # Dados de exemplo simples
    dados_exemplo = list(range(1, 21))  # [1, 2, 3, ..., 20]
    tamanho_fila = 4
    
    print(f"\nDados de teste: {dados_exemplo}")
    print(f"Tamanho da fila deslizante: {tamanho_fila}")
    
    # Testar diferentes tamanhos de buffer
    configuracoes = [1, 2, 3, 5]
    
    resultados = {}
    
    for buffer_size in configuracoes:
        resultado = demonstrar_buffer(dados_exemplo, tamanho_fila, buffer_size)
        resultados[buffer_size] = resultado
    
    # Comparação final
    print(f"\n{'='*80}")
    print("COMPARAÇÃO FINAL DOS RESULTADOS")
    print(f"{'='*80}")
    
    print(f"{'Buffer':<8} {'Janelas':<8} {'Taxa':<8} {'Espaçamento':<12} {'Primeiras Janelas'}")
    print("-" * 80)
    
    for buffer_size in configuracoes:
        janelas = resultados[buffer_size]
        taxa = len(janelas) / len(dados_exemplo) * 100
        primeiras = str(janelas[:2]) if janelas else "[]"
        
        print(f"{buffer_size:<8} {len(janelas):<8} {taxa:.1f}%{'':<4} {f'1/{buffer_size}':<12} {primeiras}")
    
    print(f"\n🎯 CONCLUSÕES:")
    print(f"   • Buffer maior = menos janelas (mais espaçamento)")
    print(f"   • Buffer menor = mais janelas (menos espaçamento)")
    print(f"   • Buffer de 1 = praticamente todos os dados")
    print(f"   • Buffer de N = pega 1 a cada N dados originais")

if __name__ == "__main__":
    main() 