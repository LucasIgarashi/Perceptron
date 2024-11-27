import numpy as np
from .funcoes import calcular_soma_ponderada, funcao_sigmoide, entropia_cruzada, atualizar_bias, atualizar_pesos
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from .globals import perda_por_epoca, contador_epocas, acuracia_por_epoca
from .plots import plotar_2D

def treinar_modelo(X_treino, y_treino, pesos, bias, taxa_aprendizado, epocas):
    global contador_epocas
    for e in range(1, epocas + 1): 
        perda_individual = []  #Lista para armazenar a perda individual de cada amostra
        y_pred = []  #Lista para armazenar as previsões
        for i in range(len(X_treino)):
            atributos = X_treino.iloc[i, :]  #Extraindo as características da amostra
            alvo_real = y_treino.iloc[i]  #Extraindo o alvo da amostra
            soma_ponderada = calcular_soma_ponderada(atributos, pesos, bias)  #Calculando a soma ponderada
            alvo_previsto = funcao_sigmoide(soma_ponderada)  #Calculando a previsão usando a função sigmoide
            perda = entropia_cruzada(alvo_real, alvo_previsto)  #Calculando a perda usando a entropia cruzada
            perda_individual.append(perda)  #Adicionando a perda individual à lista
            y_pred.append(round(alvo_previsto))  #Armazenando a previsão
            pesos = atualizar_pesos(pesos, taxa_aprendizado, alvo_real, alvo_previsto, atributos)  #Atualizando os pesos
            bias = atualizar_bias(bias, taxa_aprendizado, alvo_real, alvo_previsto)  #Atualizando o bias
        perda_media = sum(perda_individual) / len(perda_individual)  #Calculando a perda média da época
        perda_por_epoca.append(perda_media)  #Adicionando a perda média à lista de perdas por época
        acuracia = accuracy_score(y_treino, y_pred)  #Calculando a acurácia
        acuracia_por_epoca.append(acuracia)  #Adicionando a acurácia à lista de acurácias por época
        contador_epocas += 1  #Incrementando o contador global de épocas
        acuracia_media_total = sum(acuracia_por_epoca) / len(acuracia_por_epoca)
        print("============================================")
        print(f"Época {e} | Época Global {contador_epocas} ")
        print(f"Erro médio = {perda_media}")
        print(f"Acurácia = {acuracia} | Acurácia média total = {acuracia_media_total}")
        print("============================================")
    return pesos, bias


def avaliar_modelo(X_teste, y_teste, pesos, bias):
    """
    Avalia o modelo em um conjunto de teste.

    Parâmetros:
        X_teste (pd.DataFrame): Conjunto de dados de teste.
        y_teste (pd.Series): Rótulos verdadeiros do conjunto de teste.
        pesos (list): Pesos do modelo.
        bias (float): Bias do modelo.

    Retorna:
        float: Acurácia do modelo no conjunto de teste.
    """
    #Faz as previsões para o conjunto de teste
    y_pred = [funcao_sigmoide(calcular_soma_ponderada(X_teste.iloc[i, :], pesos, bias)) for i in range(len(X_teste))]
    y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]

    #Calcula a acurácia
    return accuracy_score(y_teste, y_pred)

def treinar_e_validar(dados, pesos, bias, taxa_aprendizado, epocas, k=5):
    """
    Treina e avalia o modelo de perceptron usando validação cruzada.

    Parâmetros:
        dados (pd.DataFrame): Conjunto de dados.
        pesos (list): Pesos iniciais.
        bias (float): Bias inicial.
        taxa_aprendizado (float): Taxa de aprendizado.
        epocas (int): Número de épocas.
        k (int, opcional): Número de folds para a validação cruzada. Padrão é 5.

    Retorna:
        float: Acurácia média na validação cruzada.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    global contador_epocas
    for train_index, test_index in kf.split(dados):
        X_treino, X_teste = dados.iloc[train_index, :-1], dados.iloc[test_index, :-1]
        y_treino, y_teste = dados.iloc[train_index, -1], dados.iloc[test_index, -1]

        #Treina o modelo
        pesos, bias = treinar_modelo(X_treino, y_treino, pesos, bias, taxa_aprendizado, epocas)

        #Avalia o modelo
        acuracia = avaliar_modelo(X_teste, y_teste, pesos, bias)
        scores.append(acuracia)

    #Chamar plotar_2D apenas se o número de atributos for igual a 2
    if dados.shape[1] - 1 == 2:
        plotar_2D(dados, pesos, bias)

    return np.mean(scores)
