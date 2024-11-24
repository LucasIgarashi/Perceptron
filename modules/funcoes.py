import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

bias = 0.5
l_rate = 0.1
rg = np.random.default_rng()

def generate_data(n_feature, n_values, separable):
    if separable:
        # Linearly separable data
        features, targets = make_blobs(n_samples=n_feature, n_features=n_values, centers=2, cluster_std=0.5, random_state=40)
    else:
        # Non-linearly separable data
        features, targets = make_blobs(n_samples=n_feature, n_features=n_values, centers=2, cluster_std=5, random_state=42)
    
    columns = [f"x{i}" for i in range(n_values)]
    data = pd.DataFrame(features, columns=columns)
    data["targets"] = targets
    return data

def initialize_weights(n_values):
    return rg.random((1, n_values))[0]

def data_csv(data, filename):
    data.to_csv(filename, index=False)

def get_wsum(feature, weights, bias):
    return np.dot(feature, weights) + bias

def sigmoid(w_sum):
    return 1 / (1 + np.exp(-w_sum))

def cross_entropy(target, prediction):
    return -(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

def update_w(weights, l_rate, target, prediction, feature):
    new_ws = []
    for x, w in zip(feature, weights):
        new_w = w + (l_rate * (target - prediction) * x)
        new_ws.append(new_w)
    return new_ws

def update_b(bias, l_rate, target, prediction):
    return bias + (l_rate * (target - prediction))
