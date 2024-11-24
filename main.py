from modules.funcoes import generate_data as gd
from modules.funcoes import initialize_weights as iw
from modules.funcoes import data_csv as dc
from modules.treinamento import train_model as tm
from modules.plots import plot_epoch_loss as pel, plot_2D as p2d

bias = 0.5
l_rate = 0.1

def main():
    data = gd(dados, dimensao, separable)
    weights = iw(dimensao)
    tm(data, weights, bias, l_rate, epochs)
    pel()
    dc(data, './Data/data.csv')
    if dimensao == 2:
        p2d(data, weights, bias)

print("\n============================================")
x = int(input(f"Escolha\n1: Dados pré setados\n2: Dados personalizados\nR: "))
if x == 1:
    epochs = 50
    dados = 100
    dimensao = 2
    separable = True
else:
    epochs = int(input("Insira a quantidade de épocas desejadas: "))
    dados = int(input("Insira a quantidade de dados desejados: "))
    dimensao = int(input("Insira a dimensão desejada: "))
    separable = str(input("Insira 'True' para separável linearmente ou 'False' para não separável linearmente: "))

print("============================================\n")

if __name__ == "__main__":
    main()
