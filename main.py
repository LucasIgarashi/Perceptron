from modules.funcoes import inicializar_pesos, carregar_dados, get_caminho_dados
from modules.treinamento import executar_validacao_cruzada
from modules.plots import plotar_perda_por_epoca, plotar_metricas
# from modules.funcoes import gerar_dados, salvar_ dados_csv

bias_inicial = 0.5
taxa_aprendizado = 0.01

def main():
    # if escolha == 9:
    #     n_epocas = int(input("Insira a quantidade de épocas desejadas: "))
    #     instancias = int(input("Insira a quantidade de instâncias desejadas: "))
    #     atributos = int(input("Insira a dimensão desejada: "))
    #     separavel = str(input("Insira\n1: Para base de dados linearmente separável\n0: Para base de dados não linearmente separável\nR: "))
    #     balanceada = str(input("Insira\n1: Para balanceada\n0: Para não balanceada\nR: "))
    #     dados = gerar_dados(instancias, atributos, separavel, balanceada)
    #     salvar_dados_csv(dados, caminho)
    # else:
    n_epocas = 20
    instancias = 100
    atributos = 2 if escolha in [1, 2, 3, 4] else 5
    separavel = True if escolha in [1, 2, 6, 7] else False
    balanceada = True if escolha in [1, 3, 5, 7] else False
    dados = carregar_dados(caminho)
    print(f"============================================\nÉpocas: {n_epocas}\nInstâncias: {instancias}\nAtributos: {atributos}\nSeparavel: {separavel}\nBalanceada: {balanceada}\n============================================")

    pesos_iniciais = inicializar_pesos(atributos)
    metricas_medias, metricas_por_epoca = executar_validacao_cruzada(dados, pesos_iniciais, bias_inicial, taxa_aprendizado, n_epocas)
    print("Métricas de Avaliação Média:")
    for metrica, valor in metricas_medias.items():
        print(f"{metrica}: {valor}")
    print(f"============================================\n")

    # Passando as métricas para as funções de plotagem
    plotar_perda_por_epoca(metricas_por_epoca["perda"])
    plotar_metricas(metricas_por_epoca["acuracia"], metricas_por_epoca["precisao"], metricas_por_epoca["revocacao"], metricas_por_epoca["f1"], metricas_por_epoca["auc_roc"])

print("\n============================================")
print("Escolha entre as opções abaixo:")
print("1: Dados 2D - Linearmente Separáveis e Balanceados")
print("2: Dados 2D - Linearmente Separáveis e Não Balanceados")
print("3: Dados 2D - Não Linearmente Separáveis e Balanceados")
print("4: Dados 2D - Não Linearmente Separáveis e Não Balanceados")
print("5: Dados 5D - Linearmente Separáveis e Balanceados")
print("6: Dados 5D - Linearmente Separáveis e Não Balanceados")
print("7: Dados 5D - Não Linearmente Separáveis e Balanceados")
print("8: Dados 5D - Não Linearmente Separáveis e Não Balanceados")
# print("9: Gerar nova base de dados")
escolha = int(input("R: "))
print("============================================")

caminho = get_caminho_dados(escolha)

if __name__ == "__main__":
    main()