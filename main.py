from modules.funcoes import gerar_dados as gd, inicializar_pesos as ip, salvar_dados_csv as sdc
from modules.treinamento import treinar_e_validar as tv
from modules.plots import plotar_perda_por_epoca as ppe, plotar_acuracia_por_epoca as pae

bias = 0.5
taxa_aprendizado = 0.1

def main():
    dados = gd(instancias, atributos, separavel, balanceada)  #Gerando as instâncias
    pesos = ip(atributos)  #Inicializando os pesos
    acuracia = tv(dados, pesos, bias, taxa_aprendizado, n_epocas)  #Treina e avalia o modelo
    print(f"Acurácia final: {acuracia:.2f}")
    ppe()  #Plotando o gráfico de erro
    pae()  #Plotando o gráfico de acurácia
    sdc(dados, './Data/dados.csv')  #Salvando os instâncias
    
print("\n============================================")
x = int(input(f"Escolha\n1: Dados pré setados\n2: Dados personalizados\nR: "))
if x == 1:
    n_epocas = 20
    instancias = 100
    atributos = 2
    separavel = False
    balanceada = False
elif x == 2:
    n_epocas = int(input("Insira a quantidade de épocas desejadas: "))
    instancias = int(input("Insira a quantidade de instâncias desejadas: "))
    atributos = int(input("Insira a dimensão desejada: "))
    separavel = str(input("Insira\n1: Para base de dados linearmente separável\n0: Para base de dados não linearmente separável\nR: ")) == '1'
    balanceada = str(input("Insira\n1: Para balanceada\n0: Para não balanceada\nR: ")) == '1'

print("============================================\n")

if __name__ == "__main__":
    main()
