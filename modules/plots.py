import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .globals import perda_por_epoca, acuracia_por_epoca

def plotar_perda_por_epoca():
    """
    Função para plotar a perda média por época.
    """
    df = pd.DataFrame(perda_por_epoca, columns=["Perda"])  # Criando um DataFrame com a perda média por época
    df.plot(kind="line", grid=True, title="Perda X Épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Perda")
    plt.show()

def plotar_acuracia_por_epoca():
    """
    Função para plotar a acurácia por época.
    """
    df = pd.DataFrame(acuracia_por_epoca, columns=["Acurácia"])  # Criando um DataFrame com a acurácia por época
    df.plot(kind="line", grid=True, title="Acurácia X Épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.show()

def plotar_2D(dados, pesos, bias):
    """
    Função para plotar os dados em 2D com a linha de decisão.

    Parâmetros:
    - dados: DataFrame com os dados.
    - pesos: Lista de pesos do modelo.
    - bias: Bias do modelo.
    """
    # Separando os dados por classe
    classe_0 = dados[dados['alvos'] == 0]
    classe_1 = dados[dados['alvos'] == 1]

    # Plotando os dados
    plt.scatter(classe_0['x0'], classe_0['x1'], label='Classe 0')
    plt.scatter(classe_1['x0'], classe_1['x1'], label='Classe 1')

    # Calculando a linha de decisão (y = -(w1*x1 + b) / w0)
    x1_min, x1_max = dados['x0'].min() - 1, dados['x0'].max() + 1

    x1 = np.linspace(x1_min, x1_max, len(dados))
    x2 = -(pesos[0] * x1 + bias) / pesos[1]

    # Plotando a linha de decisão
    plt.plot(x1, x2, 'r', label='Linha de Decisão')

    plt.xlabel('Atributo 1')
    plt.ylabel('Atributo 2')
    plt.legend()
    plt.grid(True)
    plt.show()
