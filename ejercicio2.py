import math

import pandas as pd
import numpy as np


# Prepara el dataset para ser utilizado
def prepare_dataset():
    df1 = pd.read_excel(r"./resources/acath.xls", engine="xlrd")
    df1 = df1.fillna(df1.mean())
    df1 = df1.drop(['tvdlm', 'sex'], axis=1)
    df1['cluster'] = 0
    return df1


def calculate_center(data):
    center = {}

    for col in data.columns:
        total = 0
        count = 0
        for j in range(len(data)):
            total += data.iloc[j][col]
            count += 1

        center[col] = total/count

    return pd.DataFrame(center, index=[0])


def calculate_distance(row, center_map):
    min_distance = 999999
    selected_class = -1

    for i in range(len(center_map)):
        distance = 0

        for col in center_map[i].columns:
            distance += math.pow(row[col] - center_map[i][col], 2)
        distance = math.sqrt(distance)

        if distance < min_distance:
            min_distance = distance
            selected_class = i

    return selected_class


# Algoritmo de las k-medias
def k_mean(data, k, iterations):
    # Shuffleo el dataset y guardo copia del original para luego comparar
    data = data.sample(frac=1).reset_index(drop=True)
    original = data.copy()

    # Remuevo la variable de enfermo
    data = data.drop(['sigdz'], axis=1)

    # Seteo el cluster de cada observación en el dataset
    data_array = np.array_split(data, k)
    data = pd.DataFrame()
    index = 0
    for arr in data_array:
        arr.loc[:, ['cluster']] = index
        data = data.append(arr)
        index += 1

    # Numero de iteraciones requeridas
    iters = 0

    while iters < iterations:
        # Inicializo lo que necesito
        center_map = {}
        cluster_number = 0

        # Metemos el centro al mapa de centros, ahora tenemos todos los centroides guardados ahi
        for i in range(k):
            center_map[cluster_number] = calculate_center(data[data['cluster'] == i])
            cluster_number += 1

        # Cuando este flag sea negativo al final, corto la ejecucion
        flag = False

        print("Iteración:", iters, ", Tamaños iniciales: ", len(data[data['cluster'] == 0]), len(data[data['cluster'] == 1]))

        # Por row en cada cluster quiero ver a que centroide se acerca más
        for index, row in data.iterrows():
            selected_class = calculate_distance(row, center_map)
            if selected_class != row['cluster']:
                data.loc[index, ['cluster']] = selected_class
                flag = True

        # Si no hay cambio de cluster, retorno
        if not flag:
            return original, data

        iters += 1


df = prepare_dataset()
original_df, predicted_df = k_mean(df, 2, 20)
results = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
for i in range(len(original_df)):
    sick = original_df.iloc[i]['sigdz']
    cluster = predicted_df.iloc[i]['cluster']
    results[sick][cluster] += 1

result_max = max(results[0], key=results[0].get)
print("Precisión no enfermos:", results[0][result_max]/(results[0][0] + results[0][1]))

result_max = max(results[1], key=results[1].get)
print("Precisión enfermos:", results[1][result_max]/(results[1][0] + results[1][1]))
