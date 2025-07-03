class FilaDeslizante:
    def __init__(self, tamanho_maximo):
        self.tamanho_maximo = tamanho_maximo
        self.elementos = []
    
    def enfileirar(self, item):
        if self.esta_cheia():
            self.desenfileirar()
        self.elementos.append(item)
    
    def desenfileirar(self):
        if len(self.elementos) == 0:
            return None
        return self.elementos.pop(0)
    
    def esta_cheia(self):
        return len(self.elementos) >= self.tamanho_maximo
    
    def get_elementos(self):
        return self.elementos.copy()

class PilhaBuffer:
    def __init__(self, tamanho_buffer):
        self.tamanho_buffer = tamanho_buffer
        self.pilha = []
    
    def empilhar(self, item):
        self.pilha.append(item)
        
        if len(self.pilha) >= self.tamanho_buffer:
            topo = self.pilha[-1]  # pega o último (topo)
            self.pilha.clear()     # limpa a pilha
            return topo
        return None
    
    def esta_cheia(self):
        return len(self.pilha) >= self.tamanho_buffer 