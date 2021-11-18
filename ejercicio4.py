import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.Perceptron import Perceptron


def prepare_dataset(train_size, test_size):
    df1 = pd.read_excel(r"./resources/acath.xls", engine="xlrd")
    df1 = df1.fillna(df1.mean())
    df1 = df1.drop(['tvdlm', 'sex'], axis=1)
    df1 = df1.sample(frac=1).reset_index(drop=True)
    df1 = (df1 - df1.min()) / (df1.max() - df1.min())
    tr_df = df1.loc[0:train_size-1, :]
    te_df = df1.loc[train_size:test_size+train_size-1, :]
    return tr_df, te_df


def get_v_array(i, j, k, r):
    v_array = []

    for t in range(k):
        for u in range(k):
            distance = math.sqrt(math.pow(t-i, 2) + math.pow(u-j, 2))
            if distance <= r:
                v_array.append([t, u, math.exp(-2*distance/r)])

    return v_array


def get_r(total_epoch, epoch, k):
    return (total_epoch - epoch) * k / total_epoch


def get_learning_rate(total_epoch, epoch):
    return 0.1 * (1 - epoch / total_epoch)


def collect_hits(network, k):
    arr = []

    for i in range(k):
        arr.append([])
        for j in range(k):
            arr[i].append(network[i][j].hits)

    return arr


def kohonen(data, k, total_epoch):
    # Me guardo el original
    original = data.copy()

    # Remuevo la variable de enfermo
    data = data.drop(['sigdz'], axis=1)

    # Creo la red de k x k
    network = []
    for i in range(k):
        network.append([])
        for j in range(k):
            network[i].append(Perceptron(i, j, len(data.columns), data.sample(n=1)))

    # Inicializo la primera epoca
    epoch = 0

    # Comienzo el algoritmo
    while epoch < total_epoch:
        r = get_r(total_epoch, epoch, k)
        learning_rate = get_learning_rate(total_epoch, epoch)

        for idx in range(len(data)):
            # Con la red, tengo que iterar por los elementos y ver la minima distancia
            min_distance = 99999
            selected_i = -1
            selected_j = -1

            # Levanto el ejemplo y calculo learning rate y vecinos en la red
            example = data.iloc[idx]

            for i in range(k):
                for j in range(k):
                    distance = network[i][j].get_distance(example)
                    if distance < min_distance:
                        min_distance = distance
                        selected_i = i
                        selected_j = j

            # Ahora que tengo la neurona ganadora, necesito actualizar los pesos de todas
            v_array = get_v_array(selected_i, selected_j, k, r)

            # Con el valor de V, puedo actualizar los pesos de todas las neuronas que corresponda
            for t in range(len(v_array)):
                i = v_array[t][0]
                j = v_array[t][1]
                v = v_array[t][2]
                network[i][j].train(example, learning_rate, v)

        # Termina la epoca, incremento en 1
        print("Succesfuly finished epoch ", epoch)
        epoch += 1

    return network


def test_kohonen(data, network, k):
    # Me guardo el original
    original = data.copy()

    # Remuevo la variable de enfermo
    data = data.drop(['sigdz'], axis=1)

    # Calculo la ganadora
    for idx in range(len(data)):
        # Con la red, tengo que iterar por los elementos y ver la minima distancia
        min_distance = 99999
        selected_i = -1
        selected_j = -1

        # Levanto el ejemplo
        example = data.iloc[idx]

        for i in range(k):
            for j in range(k):
                distance = network[i][j].get_distance(example)
                if distance < min_distance:
                    min_distance = distance
                    selected_i = i
                    selected_j = j

        network[selected_i][selected_j].hit()


# Inicializacion y algoritmo
my_k = 3
my_epoch = 1000
train_df, test_df = prepare_dataset(1000, 500)
net = kohonen(train_df, my_k, my_epoch)
test_kohonen(test_df, net, my_k)

# Debug
hits = collect_hits(net, my_k)
total_sick = test_df[test_df['sigdz'] == 1]
total_not_sick = test_df[test_df['sigdz'] == 0]
print("Sick:", len(total_sick), ", Not sick:", len(total_not_sick))
print(hits)

# Gráfico
sns.set_theme()
ax = sns.heatmap(hits, annot=True, fmt="d", cmap="Blues")
plt.show()
