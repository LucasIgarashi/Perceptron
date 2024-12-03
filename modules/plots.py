from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

def plotar_perda_por_epoca(perda_por_epoca: list) -> None:
    """
    Plota a perda por época durante o treinamento do modelo.

    Parâmetros de entrada:
        perda_por_epoca: Lista contendo a perda por época.
    """
    df = DataFrame(perda_por_epoca, columns=["Perda"])
    df.plot(kind="line", grid=True, title="Perda X Épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Perda")
    plt.show()

def plotar_metricas(acuracia_por_epoca: list, precisao_por_epoca: list, revocacao_por_epoca: list, f1_por_epoca: list, auc_roc_por_epoca: list) -> None:
    """
    Plota as métricas por época durante o treinamento do modelo em gráficos separados.

    Parâmetros de entrada:
        acuracia_por_epoca: Lista contendo a acurácia por época.
        precisao_por_epoca: Lista contendo a precisão por época.
        revocacao_por_epoca: Lista contendo a revocação por época.
        f1_por_epoca: Lista contendo o F1-Score por época.
        auc_roc_por_epoca: Lista contendo o AUC-ROC por época.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 6))

    #Plotando Acurácia
    axs[0, 0].plot(acuracia_por_epoca, label="Acurácia", color="blue")
    axs[0, 0].set_title("Acurácia X Épocas")
    axs[0, 0].set_xlabel("Épocas")
    axs[0, 0].set_ylabel("Acurácia")
    axs[0, 0].set_ylim(0, 1.05)
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    #Plotando Precisão
    axs[0, 1].plot(precisao_por_epoca, label="Precisão", color="green")
    axs[0, 1].set_title("Precisão X Épocas")
    axs[0, 1].set_xlabel("Épocas")
    axs[0, 1].set_ylabel("Precisão")
    axs[0, 1].set_ylim(0, 1.05)
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    #Plotando Revocação
    axs[0, 2].plot(revocacao_por_epoca, label="Revocação", color="red")
    axs[0, 2].set_title("Revocação X Épocas")
    axs[0, 2].set_xlabel("Épocas")
    axs[0, 2].set_ylabel("Revocação")
    axs[0, 2].set_ylim(0, 1.05)
    axs[0, 2].grid(True)
    axs[0, 2].legend()

    #Plotando F1-Score
    axs[1, 0].plot(f1_por_epoca, label="F1-Score", color="purple")
    axs[1, 0].set_title("F1-Score X Épocas")
    axs[1, 0].set_xlabel("Épocas")
    axs[1, 0].set_ylabel("F1-Score")
    axs[1, 0].set_ylim(0, 1.05)
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    #Plotando AUC-ROC
    axs[1, 1].plot(auc_roc_por_epoca, label="AUC-ROC", color="orange")
    axs[1, 1].set_title("AUC-ROC X Épocas")
    axs[1, 1].set_xlabel("Épocas")
    axs[1, 1].set_ylabel("AUC-ROC")
    axs[1, 1].set_ylim(0, 1.05)
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    fig.delaxes(axs[1, 2])
    plt.tight_layout()
    plt.show()

def plotar_2D(dados: DataFrame, pesos: list, bias: float) -> None:
    """
    Plota os dados em 2D junto com a linha de decisão do modelo.

    Parâmetros de entrada:
        dados: Conjunto de dados contendo as características e os alvos.
        pesos: Lista de pesos do modelo.
        bias: Valor do bias do modelo.
    """
    # Separando as classes
    classe_0 = dados[dados['alvos'] == 0]
    classe_1 = dados[dados['alvos'] == 1]

    # Plotando os pontos de dados
    plt.scatter(classe_0['x0'], classe_0['x1'], label='Classe 0')
    plt.scatter(classe_1['x0'], classe_1['x1'], label='Classe 1')
    
    # Definindo os limites do gráfico
    x1_min, x1_max = dados['x0'].min() - 1, dados['x0'].max() + 1
    x1 = np.linspace(x1_min, x1_max, len(dados))
    x2 = -(pesos[0] * x1 + bias) / pesos[1]

    # Plotando a linha de decisão
    plt.plot(x1, x2, 'r', label='Linha de Decisão')

    plt.xlabel('Atributo 1')
    plt.ylabel('Atributo 2')
    plt.legend()
    plt.grid(True)
    plt.title('Linha de Decisão')
    plt.show()