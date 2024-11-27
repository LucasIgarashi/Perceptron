import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

gerador_aleatorio = np.random.default_rng()
bias = 0.5
taxa_aprendizado = 0.01

def gerar_dados(instancias, n_atributos, separavel, balanceada):
    """
    Função para gerar dados sintéticos.
    
    Parâmetros:
    - instancias: Número de amostras.
    - n_atributos: Número de características.
    - separavel: Booleano indicando se os dados são linearmente separáveis.
    - balanceada: Booleano indicando se os dados são balanceados.
    
    Retorna:
    - DataFrame contendo as características e os alvos.
    """
    if separavel:
        #Base de dados linearmente separável
        if balanceada:
            atributos, alvos = make_blobs(n_samples=instancias, n_features=n_atributos, centers=2, cluster_std=1.4, random_state=42)
        else:
            #Criar uma base de dados linearmente separável e não balanceada
            n_amostras_0 = int(0.9 * instancias)
            n_amostras_1 = instancias - n_amostras_0
            atributos_0, alvos_0 = make_blobs(n_samples=n_amostras_0, n_features=n_atributos, centers=1, cluster_std=1.2, random_state=42)
            atributos_1, alvos_1 = make_blobs(n_samples=n_amostras_1, n_features=n_atributos, centers=1, cluster_std=1.2, random_state=43)
            alvos_0[:] = 0
            alvos_1[:] = 1
            atributos = np.vstack((atributos_0, atributos_1))
            alvos = np.hstack((alvos_0, alvos_1))
    else:
        #Base de dados não linearmente separável
        if balanceada:
            atributos, alvos = make_blobs(n_samples=instancias, n_features=n_atributos, centers=2, cluster_std=8.0, random_state=42)
        else:
            #Criar uma base de dados não linearmente separável e não balanceada
            n_amostras_0 = int(0.9 * instancias)
            n_amostras_1 = instancias - n_amostras_0
            atributos_0, alvos_0 = make_blobs(n_samples=n_amostras_0, n_features=n_atributos, centers=1, cluster_std=8.0, random_state=42)
            atributos_1, alvos_1 = make_blobs(n_samples=n_amostras_1, n_features=n_atributos, centers=1, cluster_std=8.0, random_state=43)
            alvos_0[:] = 0
            alvos_1[:] = 1
            atributos = np.vstack((atributos_0, atributos_1))
            alvos = np.hstack((alvos_0, alvos_1))
    
    colunas = [f"x{i}" for i in range(n_atributos)]
    dados = pd.DataFrame(atributos, columns=colunas)
    dados["alvos"] = alvos
    return dados

def inicializar_pesos(n_atributos):
    """
    Função para inicializar os pesos aleatoriamente.
    
    Parâmetros:
    - n_atributos: Número de características.
    
    Retorna:
    - Lista de pesos inicializados.
    """
    return gerador_aleatorio.random(n_atributos)

def salvar_dados_csv(dados, nome_arquivo):
    """
    Função para salvar os dados em um arquivo CSV.
    
    Parâmetros:
    - dados: DataFrame contendo os dados.
    - nome_arquivo: Nome do arquivo CSV.
    """
    dados.to_csv(nome_arquivo, index=False)

def calcular_soma_ponderada(atributos, pesos, bias):
    """
    Função para calcular a soma ponderada das características.
    
    Parâmetros:
    - atributos: Características da amostra.
    - pesos: Lista de pesos.
    - bias: Valor do bias.
    
    Retorna:
    - Soma ponderada.
    """
    return np.dot(atributos, pesos) + bias

def funcao_sigmoide(soma_ponderada):
    """
    Função para calcular a função sigmoide.
    
    Parâmetros:
    - soma_ponderada: Soma ponderada.
    
    Retorna:
    - Valor da função sigmoide.
    """
    return 1 / (1 + np.exp(-soma_ponderada))

def entropia_cruzada(alvo_real, alvo_previsto):
    """
    Função para calcular a perda usando a entropia cruzada.
    
    Parâmetros:
    - alvo_real: Valor alvo.
    - alvo_previsto: Valor previsto.
    
    Retorna:
    - Valor da perda.
    """
    epsilon = 1e-15 # Pequeno valor para evitar log(0)
    alvo_previsto = np.clip(alvo_previsto, epsilon, 1 - epsilon)
    return -((alvo_real * np.log(alvo_previsto)) + ((1 - alvo_real) * np.log(1 - alvo_previsto)))

def atualizar_pesos(pesos, taxa_aprendizado, alvo_real, alvo_previsto, atributos):
    """
    Função para atualizar os pesos.
    
    Parâmetros:
    - pesos: Lista de pesos.
    - taxa_aprendizado: Taxa de aprendizado.
    - alvo_real: Valor alvo.
    - alvo_previsto: Valor previsto.
    - atributos: Características da amostra.
    
    Retorna:
    - Lista de pesos atualizados.
    """
    novos_pesos = []
    for x, w in zip(atributos, pesos):
        novo_peso = w + (taxa_aprendizado * (alvo_real - alvo_previsto) * x)
        novos_pesos.append(novo_peso)
    return novos_pesos

def atualizar_bias(bias, taxa_aprendizado, alvo_real, alvo_previsto):
    """
    Função para atualizar o bias.
    
    Parâmetros:
    - bias: Valor do bias.
    - taxa_aprendizado: Taxa de aprendizado.
    - alvo_real: Valor alvo.
    - alvo_previsto: Valor previsto.
    
    Retorna:
    - Valor do bias atualizado.
    """
    return bias + (taxa_aprendizado * (alvo_real - alvo_previsto))

def prever(X, pesos, bias):
    """
    Faz previsões para um conjunto de dados.
    
    Parâmetros:
    - X: Conjunto de dados.
    - pesos: Lista de pesos do modelo.
    - bias: Bias do modelo.
    
    Retorna:
    - Array de previsões.
    """
    previsoes = []
    for i in range(len(X)):
        soma_ponderada = calcular_soma_ponderada(X.iloc[i], pesos, bias)
        alvo_previsto = funcao_sigmoide(soma_ponderada)
        previsoes.append(round(alvo_previsto))
    return np.array(previsoes)
