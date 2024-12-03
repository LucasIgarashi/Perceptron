import numpy as np
from pandas import DataFrame, read_csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.datasets import make_blobs

gerador_aleatorio = np.random.default_rng()

# Função utilizada para gerar as bases de dados

# def gerar_dados(instancias: int, n_atributos: int, separavel: bool, balanceada: bool) -> DataFrame:
#     """
#     Gera dados sintéticos para treinamento e teste.

#     Parâmetros de entrada:
#         instancias: Número de amostras.
#         n_atributos: Número de características.
#         separavel: Indica se os dados são linearmente separáveis.
#         balanceada: Indica se os dados são balanceados.

#     Retorna:
#         dados (DataFrame): DataFrame contendo as características e os alvos.
#     """
#     if separavel:
#         if balanceada:
#             atributos, alvos = make_blobs(n_samples=instancias, n_features=n_atributos, centers=2, cluster_std=1, random_state=40)
#         else:
#             n_amostras_0 = int(0.9 * instancias)
#             n_amostras_1 = instancias - n_amostras_0
#             atributos_0, alvos_0 = make_blobs(n_samples=n_amostras_0, n_features=n_atributos, centers=1, cluster_std=1.2, random_state=42)
#             atributos_1, alvos_1 = make_blobs(n_samples=n_amostras_1, n_features=n_atributos, centers=1, cluster_std=1.2, random_state=43)
#             alvos_0[:] = 0
#             alvos_1[:] = 1
#             atributos = np.vstack((atributos_0, atributos_1))
#             alvos = np.hstack((alvos_0, alvos_1))
#     else:
#         if balanceada:
#             atributos, alvos = make_blobs(n_samples=instancias, n_features=n_atributos, centers=2, cluster_std=8.0, random_state=42)
#         else:
#             n_amostras_0 = int(0.9 * instancias)
#             n_amostras_1 = instancias - n_amostras_0
#             atributos_0, alvos_0 = make_blobs(n_samples=n_amostras_0, n_features=n_atributos, centers=1, cluster_std=8.0, random_state=42)
#             atributos_1, alvos_1 = make_blobs(n_samples=n_amostras_1, n_features=n_atributos, centers=1, cluster_std=8.0, random_state=44)
#             alvos_0[:] = 0
#             alvos_1[:] = 1
#             atributos = np.vstack((atributos_0, atributos_1))
#             alvos = np.hstack((alvos_0, alvos_1))
    
#     colunas = [f"x{i}" for i in range(n_atributos)]
#     dados = DataFrame(atributos, columns=colunas)
#     dados["alvos"] = alvos
#     return dados


def carregar_dados(caminho: str) -> DataFrame:
    """
    Carrega os dados de um arquivo CSV.

    Parâmetros de entrada:
        caminho: Caminho para o arquivo CSV.

    Retorna:
        DataFrame: DataFrame contendo os dados carregados.
    """
    return read_csv(caminho)

def get_caminho_dados(escolha: int) -> str:
    """
    Mapeia a escolha do usuário para o caminho do arquivo CSV correspondente.

    Parâmetros de entrada:
        escolha: Escolha do usuário.

    Retorna:
        str: Caminho para o arquivo CSV correspondente.
    """
    caminhos = {
        1: './Data/2D/dados_L_&_B.csv',
        2: './Data/2D/dados_L_&_NB.csv',
        3: './Data/2D/dados_NL_&_B.csv',
        4: './Data/2D/dados_NL_&_NB.csv',
        5: './Data/MD/dados_L_&_B.csv',
        6: './Data/MD/dados_L_&_NB.csv',
        7: './Data/MD/dados_NL_&_B.csv',
        8: './Data/MD/dados_NL_&_NB.csv',
#        9: './Data/dados_rand.csv'
    }
    return caminhos.get(escolha, '')

def inicializar_pesos(n_atributos: int) -> np.ndarray:
    """
    Inicializa os pesos aleatoriamente.

    Parâmetros de entrada:
        n_atributos: Número de características.

    Retorna:
        np.ndarray: Array de pesos inicializados.
    """
    return gerador_aleatorio.random(n_atributos)

# Função utilizada para salvar os dados que haviam sido gerados pela função gerar_dados

# def salvar_dados_csv(dados: DataFrame, nome_arquivo: str) -> None:
#     """
#     Salva os dados em um arquivo CSV.

#     Parâmetros de entrada:
#         dados: DataFrame contendo os dados.
#         nome_arquivo: Nome do arquivo CSV.
#     """
#     dados.to_csv(nome_arquivo, index=False)

def calcular_soma_ponderada(atributos: np.ndarray, pesos: np.ndarray, bias: float) -> float:
    """
    Calcula a soma ponderada das características.

    Parâmetros de entrada:
        atributos: Características da amostra.
        pesos: Array de pesos.
        bias: Valor do bias.

    Retorna:
        float: Soma ponderada.
    """
    return np.dot(atributos, pesos) + bias

def funcao_sigmoide(soma_ponderada: float) -> float:
    """
    Calcula a função sigmoide.

    Parâmetros de entrada:
        soma_ponderada: Soma ponderada.

    Retorna:
        float: Valor da função sigmoide.
    """
    soma_ponderada = np.clip(soma_ponderada, -500, 500) # Limitar a soma ponderada para evitar overfit da sigmoid
    return 1 / (1 + np.exp(-soma_ponderada))

def entropia_cruzada(alvo_real: float, alvo_previsto: float) -> float:
    """
    Calcula a perda usando a entropia cruzada.

    Parâmetros de entrada:
        alvo_real: Valor alvo.
        alvo_previsto: Valor previsto.

    Retorna:
        float: Valor da perda.
    """
    alvo_previsto = np.clip(alvo_previsto, 1e-15, 1 - 1e-15) # Limitar os valores previstos para evitar problemas numéricos na entropia cruzada
    return -((alvo_real * np.log(alvo_previsto)) + ((1 - alvo_real) * np.log(1 - alvo_previsto)))

def atualizar_pesos(pesos: np.ndarray, taxa_aprendizado: float, alvo_real: float, alvo_previsto: float, atributos: np.ndarray) -> np.ndarray:
    """
    Atualiza os pesos.

    Parâmetros de entrada:
        pesos: Array de pesos.
        taxa_aprendizado: Taxa de aprendizado.
        alvo_real: Valor alvo.
        alvo_previsto: Valor previsto.
        atributos: Características da amostra.

    Retorna:
        np.ndarray: Array de pesos atualizados.
    """
    novos_pesos = []
    for x, w in zip(atributos, pesos):
        novo_peso = w + (taxa_aprendizado * (alvo_real - alvo_previsto) * x)
        novos_pesos.append(novo_peso)
    return np.array(novos_pesos)

def atualizar_bias(bias: float, taxa_aprendizado: float, alvo_real: float, alvo_previsto: float) -> float:
    """
    Atualiza o bias.

    Parâmetros de entrada:
        bias: Valor do bias.
        taxa_aprendizado: Taxa de aprendizado.
        alvo_real: Valor alvo.
        alvo_previsto: Valor previsto.

    Retorna:
        float: Valor do bias atualizado.
    """
    return bias + (taxa_aprendizado * (alvo_real - alvo_previsto))

def prever(X: DataFrame, pesos: np.ndarray, bias: float) -> np.ndarray:
    """
    Faz previsões para um conjunto de dados.

    Parâmetros de entrada:
        X: Conjunto de dados.
        pesos: Array de pesos do modelo.
        bias: Bias do modelo.

    Retorna:
        np.ndarray: Array de previsões.
    """
    previsoes = []
    for i in range(len(X)):
        soma_ponderada = calcular_soma_ponderada(X.iloc[i], pesos, bias)
        alvo_previsto = funcao_sigmoide(soma_ponderada)
        previsoes.append(round(alvo_previsto))
    return np.array(previsoes)

def calcular_metricas(y_real: np.array, y_pred: np.array, y_pred_prob: np.array):
    """
    Calcula as métricas de avaliação.

    Parâmetros de entrada:
        y_real: Valores reais dos alvos.
        y_pred: Valores previstos dos alvos.
        y_pred_prob: Probabilidades previstas dos alvos.

    Retorna:
        tuple: Uma tupla contendo:
            - acuracia (float): Acurácia do modelo.
            - precisao (float): Precisão do modelo.
            - revocacao (float): Revocação do modelo.
            - f1 (float): F1-Score do modelo.
            - auc_roc (float): AUC-ROC do modelo.
    """
    acuracia = 0
    precisao = 0
    revocacao = 0
    f1 = 0
    auc_roc = 0
    
    acuracia = accuracy_score(y_real, y_pred)
    precisao = precision_score(y_real, y_pred)
    revocacao = recall_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)
    auc_roc = roc_auc_score(y_real, y_pred_prob)
    return acuracia, precisao, revocacao, f1, auc_roc