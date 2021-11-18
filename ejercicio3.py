import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy


def prepare_dataset(train_size, test_size):
    df1 = pd.read_excel(r"./resources/acath.xls", engine="xlrd")
    df1 = df1.fillna(df1.mean())
    df1 = df1.drop(['tvdlm', 'sex'], axis=1)
    df1 = df1.sample(frac=1).reset_index(drop=True)
    df1 = (df1 - df1.min()) / (df1.max() - df1.min())
    df1['cluster'] = 0
    tr_df = df1.loc[0:train_size-1, :]
    te_df = df1.loc[train_size:test_size+train_size-1, :]
    return tr_df, te_df


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


# Calcula la distancia entre dos grupos
def calculate_distance(data, g1, g2):
    center_g1 = calculate_center(data.loc[data['cluster'] == g1])
    center_g2 = calculate_center(data.loc[data['cluster'] == g2])
    distance = 0

    for col in center_g1.columns:
        if col != 'cluster':
            distance += math.pow(center_g1.iloc[0][col] - center_g2.iloc[0][col], 2)

    return math.sqrt(distance)


def algorithm(data):
    original = data.copy()

    # Remuevo la variable de enfermo
    data = data.drop(['sigdz'], axis=1)

    # Esta es la matriz para devolver con todos los cambios
    changes = []

    # Seteo el cluster de cada observación en el dataset
    for i in range(len(data)):
        data.loc[i, ['cluster']] = i

    while len(data['cluster'].unique()) > 1:
        # Quiero obtener la menor distancia entre dos clusters
        min_distance = 99999
        group_1 = -1
        group_2 = -1

        for g1 in data['cluster'].unique():
            for g2 in data['cluster'].unique():
                if g1 != g2:
                    distance = calculate_distance(data, g1, g2)
                    if distance < min_distance:
                        min_distance = distance
                        group_1 = g1
                        group_2 = g2

        # Ahora que tengo los grupos con distancia minima, los junto en un nuevo cluster
        # Calculo el numero de cluster
        next_cluster = max(data['cluster'].unique()) + 1

        # Agarro los indices de todos los grupos y los actualizo
        g1_indexes = data.loc[data['cluster'] == group_1].index
        g2_indexes = data.loc[data['cluster'] == group_2].index
        data.loc[g1_indexes, ['cluster']] = next_cluster
        data.loc[g2_indexes, ['cluster']] = next_cluster

        # Agrego el cambio de cluster al array de cambios
        changes.append([group_1, group_2, min_distance, len(g1_indexes) + len(g2_indexes)])
        print("These groups are together now:", group_1, group_2)

    return changes
    #return original, data, data['cluster'].unique()


def predict(example, centers):
    min_distance = 9999
    group = -1

    for key in centers.keys():
        distance = 0
        for col in centers[key].columns:
            distance += math.pow(example[col] - centers[key][col], 2)

        distance = math.sqrt(distance)

        if distance < min_distance:
            min_distance = distance
            group = key

    return group


train_df, test_df = prepare_dataset(500, 20)
arr = algorithm(train_df)
hierarchy.dendrogram(np.array(arr))
print(train_df)
print("---")
print(arr)
plt.show()
# original_df, predicted_df, arr = algorithm(train_df)
#
# idx = predicted_df.loc[predicted_df['cluster'] == arr[0]].index
# predicted_df.loc[idx, ['cluster']] = 0
# idx = predicted_df.loc[predicted_df['cluster'] == arr[1]].index
# predicted_df.loc[idx, ['cluster']] = 1
#
# center_map = {0: calculate_center(predicted_df.loc[predicted_df['cluster'] == 0]),
#               1: calculate_center(predicted_df.loc[predicted_df['cluster'] == 1])}
#
# for i in range(len(predicted_df)):
#     row = predicted_df.iloc[i]
#     selected_cluster = predict(row, center_map)
#     print("Predicted:", selected_cluster, ", Real:", row['cluster'])
#
# fig = ff.create_dendrogram(predicted_df)
# fig.update_layout(width=1024, height=768)
# fig.show()
