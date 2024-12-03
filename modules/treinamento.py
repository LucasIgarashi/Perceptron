import numpy as np
from pandas import DataFrame, Series
from .funcoes import calcular_soma_ponderada, funcao_sigmoide, entropia_cruzada, atualizar_bias, atualizar_pesos, calcular_metricas
from sklearn.model_selection import KFold
from .plots import plotar_2D

def treinar_modelo(X_treino: DataFrame, y_treino: Series, pesos: list, bias: float, taxa_aprendizado: float, epocas: int):
    """
    Treina o modelo Perceptron de uma camada.

    Parâmetros de entrada:
        X_treino: Conjunto de dados de treino, onde cada linha representa uma amostra e cada coluna representa uma característica.
        y_treino: Rótulos verdadeiros do conjunto de treino.
        pesos: Lista de pesos iniciais do modelo.
        bias: Valor inicial do bias do modelo.
        taxa_aprendizado: Taxa de aprendizado usada para ajustar os pesos e o bias.
        epocas: Número de épocas para treinar o modelo.

    Retorna:
        tuple: Uma tupla contendo:
            - pesos (list): Lista de pesos atualizados após o treinamento.
            - bias (float): Valor do bias atualizado após o treinamento.
            - metricas_por_epoca (dict): Dicionário das métricas calculadas por época
    """
    perda_por_epoca = []
    acuracia_por_epoca = []
    precisao_por_epoca = []
    revocacao_por_epoca = []
    f1_por_epoca = []
    auc_roc_por_epoca = []

    for e in range(epocas):
        perda_individual = []
        y_pred = []
        y_pred_prob = []
        for i in range(len(X_treino)):
            atributos = X_treino.iloc[i, :]
            alvo_real = y_treino.iloc[i]
            soma_ponderada = calcular_soma_ponderada(atributos, pesos, bias)
            alvo_previsto = funcao_sigmoide(soma_ponderada)
            perda = entropia_cruzada(alvo_real, alvo_previsto)
            perda_individual.append(perda)
            y_pred_prob.append(alvo_previsto)
            y_pred.append(round(alvo_previsto))
            pesos = atualizar_pesos(pesos, taxa_aprendizado, alvo_real, alvo_previsto, atributos)
            bias = atualizar_bias(bias, taxa_aprendizado, alvo_real, alvo_previsto)

        perda_media = sum(perda_individual) / len(perda_individual)
        perda_por_epoca.append(perda_media)

        print(f"Perda na Época {e + 1}: {perda_media}")

        acuracia, precisao, revocacao, f1, auc_roc = calcular_metricas(y_treino, y_pred, y_pred_prob)

        acuracia_por_epoca.append(acuracia)
        precisao_por_epoca.append(precisao)
        revocacao_por_epoca.append(revocacao)
        f1_por_epoca.append(f1)
        auc_roc_por_epoca.append(auc_roc)

    metricas_por_epoca = {
        "perda": perda_por_epoca,
        "acuracia": acuracia_por_epoca,
        "precisao": precisao_por_epoca,
        "revocacao": revocacao_por_epoca,
        "f1": f1_por_epoca,
        "auc_roc": auc_roc_por_epoca
    }

    return pesos, bias, metricas_por_epoca

def avaliar_modelo(X_teste: DataFrame, y_teste: Series, pesos: list, bias: float) -> dict:
    """
    Avalia o modelo em um conjunto de teste usando várias métricas.

    Parâmetros de entrada:
        X_teste (DataFrame): Conjunto de dados de teste.
        y_teste (Series): Rótulos verdadeiros do conjunto de teste.
        pesos (list): Pesos do modelo.
        bias (float): Bias do modelo.

    Retorna:
        dict: Dicionário contendo:
            - 'Acuracia' (float): Valor da acurácia.
            - 'Precisao' (float): Valor da precisão.
            - 'Revocacao' (float): Valor da revocação.
            - 'F1-Score' (float): Valor do F1-Score.
            - 'AUC-ROC' (float): Valor do AUC-ROC.
    """

    y_pred_prob = [funcao_sigmoide(calcular_soma_ponderada(X_teste.iloc[i, :], pesos, bias)) for i in range(len(X_teste))]
    y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_prob]

    acuracia, precisao, revocacao, f1, auc_roc = calcular_metricas(y_teste, y_pred, y_pred_prob)

    return {
        "Acuracia": acuracia,
        "Precisao": precisao,
        "Revocacao": revocacao,
        "F1-Score": f1,
        "AUC-ROC": auc_roc
    }

def executar_validacao_cruzada(dados: DataFrame, pesos_iniciais: list, bias_inicial: float, taxa_aprendizado: float, epocas: int) -> dict:
    """
    Treina em K-1 folds e avalia o modelo no fold restante. O processo é repetido K vezes.

    Parâmetros de entrada:
        dados: Conjunto de dados.
        pesos_iniciais: Pesos iniciais.
        bias_inicial: Bias inicial.
        taxa_aprendizado: Taxa de aprendizado.
        epocas: Número de épocas.
        k: Número de folds para a validação cruzada. Padrão é 5.

    Retorna:
        tuple: Uma tupla contendo:
            - metricas_medias (dict): Dicionário contendo as métricas de avaliação médias na validação cruzada.
            - lista_metricas_por_epoca (dict): Dicionário contendo as métricas por época durante o treinamento.
    """


    kf = KFold(n_splits=5, shuffle=True, random_state=42) # Inicializa a validação cruzada com cinco folds
    lista_metricas = {
        "Acuracia": [],
        "Precisao": [],
        "Revocacao": [],
        "F1-Score": [],
        "AUC-ROC": []
    }
    lista_metricas_por_epoca = {
        "perda": [],
        "acuracia": [],
        "precisao": [],
        "revocacao": [],
        "f1": [],
        "auc_roc": []
    }

    for fold, (train_index, test_index) in enumerate(kf.split(dados), start=1):
        # Inicializa os pesos e bias para cada fold
        pesos = pesos_iniciais.copy()
        bias = bias_inicial

        # Divide a base de dados em conjuntos de treino e teste para o fold atual
        X_treino, X_teste = dados.iloc[train_index, :-1], dados.iloc[test_index, :-1]
        y_treino, y_teste = dados.iloc[train_index, -1], dados.iloc[test_index, -1]

        # Treina o modelo no conjunto de treino
        print(f"Pesos não atualizados: {pesos}\nBias não atualizado: {bias}")
        print("--------------------------------------------")
        pesos, bias, metricas_fold = treinar_modelo(X_treino, y_treino, pesos, bias, taxa_aprendizado, epocas)
        print("--------------------------------------------")
        print(f"Pesos atualizados: {pesos}\nBias atualizado: {bias}")

        # Armazena as métricas por época
        for chave, valores in metricas_fold.items():
            lista_metricas_por_epoca[chave].extend(valores)

        # Avalia o modelo no conjunto de teste
        metricas_fold = avaliar_modelo(X_teste, y_teste, pesos, bias)
        for chave, valor in metricas_fold.items():
            lista_metricas[chave].append(valor)

        # Imprime as métricas de avaliação por fold
        print("============================================")
        print(f"Métricas de Avaliação por Fold | k = {fold}")
        for metrica, valor in metricas_fold.items():
            print(f"{metrica}: {valor}")
        print("============================================")

        # Plota os dados em 2D se o número de atributos for 2
        if dados.shape[1] - 1 == 2:
            plotar_2D(dados, pesos, bias)

    # Calcula as métricas médias na validação cruzada
    metricas_medias = {chave: np.mean(valores) for chave, valores in lista_metricas.items()}

    return metricas_medias, lista_metricas_por_epoca