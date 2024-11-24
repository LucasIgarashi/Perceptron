import pandas as pd
import matplotlib.pyplot as plt
from .treinamento import epoch_loss
from .funcoes import sigmoid
import numpy as np


def plot_epoch_loss():
    df = pd.DataFrame(epoch_loss, columns=["Loss"])
    df.plot(kind="line", grid=True, title="Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_2D(data, weights, bias):
    #Separar as características e os alvos
    features = data.iloc[:, :-1].values
    targets = data.iloc[:, -1].values

    #Plotar os dados
    plt.scatter(features[:, 0], features[:, 1], c=targets, cmap='viridis', marker='o')

    #Calcular os limites da linha de decisão
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    #Calcular a função de decisão
    Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)
    Z = Z.reshape(xx.shape)

    #Plotar a linha de decisão
    plt.contour(xx, yy, Z, levels=[0.5], colors='red')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Data Classification')
    plt.grid()
    plt.show()