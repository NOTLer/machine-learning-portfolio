import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.cluster.hierarchy import linkage
from mpl_toolkits.mplot3d import Axes3D


# --- Минковский ---
def minkowski_distance(x, y, p=2):
    return sum(abs(a - b) ** p for a, b in zip(x, y)) ** (1 / p)


# --- Ward с формулой Ланса-Вильямса (словарь расстояний) ---
def ward_clustering(data, p=2):
    n = len(data)
    # Инициализация словаря расстояний
    D = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = minkowski_distance(data[i], data[j], p)
            D[i][j] = d
            D[j][i] = d

    clusters = {i: [i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    next_cluster_id = n
    Z = []

    while len(clusters) > 1:
        keys = list(clusters.keys())
        # Находим два ближайших кластера
        min_dist = float("inf")
        a_id = b_id = -1
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                id1, id2 = keys[i], keys[j]
                d = D[id1][id2]
                if d < min_dist:
                    min_dist = d
                    a_id, b_id = id1, id2

        # размеры кластеров
        k_u, k_v = sizes[a_id], sizes[b_id]
        k_w = k_u + k_v

        # Вычисляем новые расстояния с формулой Ланса-Вильямса
        new_distances = {}
        for s_id in clusters:
            if s_id not in (a_id, b_id):
                k_s = sizes[s_id]
                d_su = D[s_id][a_id]
                d_sv = D[s_id][b_id]
                d_uv = D[a_id][b_id]

                alpha_u = (k_s + k_u) / (k_s + k_w)
                alpha_v = (k_s + k_v) / (k_s + k_w)
                beta = -k_s / (k_s + k_w)
                gamma = 0

                d_sw = alpha_u * d_su + alpha_v * d_sv + beta * d_uv + gamma * abs(d_su - d_sv)
                new_distances[s_id] = d_sw

        # Создаём новый кластер
        new_cluster = clusters[a_id] + clusters[b_id]
        clusters[next_cluster_id] = new_cluster
        sizes[next_cluster_id] = k_w

        # Сохраняем слияние для scipy
        Z.append([a_id, b_id, min_dist, k_w])

        # Удаляем старые кластеры
        del clusters[a_id]
        del clusters[b_id]
        del sizes[a_id]
        del sizes[b_id]

        # Обновляем словарь расстояний
        D[next_cluster_id] = {}
        for s_id, d in new_distances.items():
            D[next_cluster_id][s_id] = d
            D[s_id][next_cluster_id] = d

        next_cluster_id += 1

    return np.array(Z)


# --- Визуализация кластеров 2D/3D ---
def plot_clusters_2d3d(data, clusters):
    data = np.array(data)
    clusters = np.array(clusters)
    unique_clusters = np.unique(clusters)
    colors = plt.cm.get_cmap("tab10", len(unique_clusters))

    if data.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        for i, c in enumerate(unique_clusters):
            plt.scatter(data[clusters == c, 0], data[clusters == c, 1], label=f'Cluster {c}', color=colors(i))
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Кластеры в 2D пространстве")
        plt.legend()
        plt.show()
    elif data.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i, c in enumerate(unique_clusters):
            ax.scatter(data[clusters == c, 0], data[clusters == c, 1], data[clusters == c, 2],
                       label=f'Cluster {c}', color=colors(i))
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
        ax.set_title("Кластеры в 3D пространстве")
        plt.legend()
        plt.show()
    else:
        print("Невозможно отобразить, размерность > 3")


# --- Пример данных ---
# data1 = [
#     [62.3, 1.5, 8.4],
#     [66.5, 0.8, 5.2],
#     [56.5, 1.1, -0.7],
#     [69, 2.5, 1.2],
#     [66.9, 1.3, -5],
#     [73.6, 1.5, -4],
#     [60.4, 0.3, 3],
#     [60.7, 1, 8.7],
#     [66.8, 1.8, 11.7],
#     [76.9, 5.6, -5.9],
#     [70.8, 2, 5],
#     [64.5, 1.4, 4.1],
#     [70.9, 1.4, 3.4],
#     [65.3, 1, 16.8],
#     [70.2, 1.3, 7],
#     [59, 0.5, 11.5],
#     [52.6, 0.5, 11.2],
#     [76.8, 3.6, -6]
# ]

# data1 = [
#     [11.5, 6.9],
#     [20.1, 11.1],
#     [34.2, 21.3],
#     [22.1, 20.5],
#     [13.4, 9.7],
#     [29.4, 18.2]
# ]

from sklearn.datasets import load_breast_cancer
data1 = load_breast_cancer()


# Z_scipy = linkage(data1, method='ward')
# clusters = fcluster(Z_scipy, 2, criterion='maxclust')
#
# # --- Дендрограмма ---
# plt.figure(figsize=(10, 6))
# dendrogram(Z_scipy)
# plt.title("Дендограмма")
# plt.xlabel("Объекты")
# plt.ylabel("Расстояние")
# plt.show()
#
# # --- Визуализация кластеров ---
# plot_clusters_2d3d(data1, clusters)


# --- Кластеризация ---
Z = ward_clustering(data1, p=2)
# --- Финальные кластеры (например, 2) ---
clusters = fcluster(Z, 2, criterion='maxclust')
print("Финальные кластеры:", clusters)
# --- Дендрограмма ---
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=[f"Obj{i}" for i in range(len(data1))], color_threshold=None)
plt.title("Дендограмма")
plt.xlabel("Объекты")
plt.ylabel("Расстояние")
plt.show()

# --- Визуализация кластеров ---
plot_clusters_2d3d(data1, clusters)
