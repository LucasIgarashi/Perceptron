from .funcoes import get_wsum, sigmoid, cross_entropy, update_b, update_w

epoch_loss = []

def train_model(data, weights, bias, l_rate, epochs):
    for e in range(epochs + 1): 
        individual_loss = []
        for i in range(len(data)):
            feature = data.iloc[i, :-1]
            target = data.iloc[i, -1]
            w_sum = get_wsum(feature, weights, bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy(target, prediction)
            individual_loss.append(loss)
            weights = update_w(weights, l_rate, target, prediction, feature)
            bias = update_b(bias, l_rate, target, prediction)
        average_loss = sum(individual_loss) / len(individual_loss)
        epoch_loss.append(average_loss)
        print("============================================")
        print(f"epoch {e}")
        print(f"Erro médio = {average_loss}")
        print("============================================")
