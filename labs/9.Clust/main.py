import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from enum import Enum
from random import shuffle


class DatasetObject:
    def __init__(self, x, y):
        self.signs = x
        self.obj_class = y


class DBSCAN:
    class TypesOfObjects(Enum):
        CORNER = 1
        BORDER = 2
        NOISE = 3

    def __init__(self, rad, count):
        self.rad = rad
        self.count = count

    def cluster(self, dataset):
        non_t = dataset.copy()
        shuffle(non_t)

        clusters = dict()

        result = dict()
        for obj in dataset:
            result[obj] = -1

        cluster_number = 0
        while len(non_t) != 0:
            rand_object = non_t.pop()

            eps_sphere_new_obj = self.__eps_sphere(dataset, rand_object)
            if len(eps_sphere_new_obj) < self.count:
                clusters[rand_object] = self.TypesOfObjects.NOISE
            else:
                cluster_number += 1
                for obj in eps_sphere_new_obj:
                    if result[obj] == -1 or clusters.get(obj) == self.TypesOfObjects.NOISE:
                        eps_sphere_cur_obj = self.__eps_sphere(dataset, obj)
                        if len(eps_sphere_cur_obj) >= self.count:
                            eps_sphere_new_obj.union(eps_sphere_cur_obj)
                        else:
                            clusters[obj] = self.TypesOfObjects.BORDER
                for obj in eps_sphere_new_obj:
                    result[obj] = cluster_number
                    if obj in non_t:
                        non_t.remove(obj)

        return result

    def __eps_sphere(self, dataset, obj):
        objects = set()

        for cur_obj in dataset:
            if cur_obj != obj:
                if self.euclid_dist(obj.signs, cur_obj.signs) <= self.rad:
                    objects.add(cur_obj)

        return objects

    @staticmethod
    def euclid_dist(u, v):
        result_number = 0
        for k in range(0, len(u) - 1):
            result_number += (u[k] - v[k]) ** 2
        return np.sqrt(result_number)


def normalize(dataset):
    dataset.drop_duplicates()
    for label in dataset.columns:
        if label == "Class":
            dataset[label] = pd.factorize(dataset[label])[0]
        else:
            min_parameter = dataset[label].min()
            max_parameter = dataset[label].max()
            dataset[label] = (dataset[label] - min_parameter) / (max_parameter - min_parameter)

    x = dataset.values[:, :-1]
    y = dataset.values[:, -1]

    return x, y


def index_rand(dataset, clast_res):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for obj_1 in dataset:
        for obj_2 in dataset:
            if obj_1 == obj_2:
                continue
            if obj_1.obj_class == obj_2.obj_class:
                if clast_res[obj_1] == clast_res[obj_2]:
                    tp += 1
                else:
                    fp += 1
            else:
                if clast_res[obj_1] == clast_res[obj_2]:
                    tn += 1
                else:
                    fn += 1

    return (tp + fn) / (tp + fp + tn + fn)


def silhouette(dataset, clasters):
    sum = 0
    for key, value in clasters.items():
        for x in value:
            sum += (b_for_silhouette(x, clasters, key) - a_for_silhouette(x, value)) / \
                   max(b_for_silhouette(x, clasters, key), a_for_silhouette(x, value))

    return sum / len(dataset)


def a_for_silhouette(x, cluster):
    return sum(map(lambda cur_x: DBSCAN.euclid_dist(cur_x, x), cluster)) / len(cluster)


def b_for_silhouette(x, clusters, key):
    temp_clusters = clusters.copy()
    temp_clusters.pop(key)
    return min(map(lambda cur_cluster: a_for_silhouette(x, cur_cluster), temp_clusters.values()))


def print_index_rand_dependence(dataset, m):
    indices = []
    epsilons = []
    for eps in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1):
        object_to_cluster_map = DBSCAN(eps, m).cluster(dataset)
        indices.append(index_rand(dataset, object_to_cluster_map))
        epsilons.append(eps)

    plt.grid(linestyle='--')
    plt.plot(epsilons, indices, linestyle='-', marker='o', color='b')
    plt.title("Зависимость index_rand от радиуса эпсилон окрестности. m = " + str(m))
    plt.xlabel("Радиус эпсилон окрестности")
    plt.ylabel("Значение index_rand")
    plt.show()


def print_silhouette_dependence(dataset, m):
    measure = []
    epsilons = []

    for eps in (1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8):
        object_to_cluster_map = DBSCAN(eps, m).cluster(dataset)
        clusters = get_clusters_as_map_from_cluster_to_objects_x(object_to_cluster_map)
        measure.append(silhouette(dataset, clusters))
        epsilons.append(eps)

    plt.grid(linestyle='--')
    plt.plot(epsilons, measure, linestyle='-', marker='o', color='b')
    plt.title("Зависимость silhouette от радиуса эпсилон окрестности. m = " + str(m))
    plt.xlabel("Радиус эпсилон окрестности")
    plt.ylabel("Значение silhouette")
    plt.show()


def print_spaces(dataset, eps, m):
    object_to_cluster_map = DBSCAN(eps, m).cluster(dataset)
    X_res = []
    y_res = []
    y_real = []
    for obj, cluster in object_to_cluster_map.items():
        X_res.append(obj.signs)
        y_res.append(cluster)
        y_real.append(obj.obj_class)

    reduced_X = TSNE(n_components=2).fit_transform(X)
    black_color = "000000"
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#af4035',
               '#9467bd', '#ff8c69', '#ccccff', '#4682b4', '#993366',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#708090',
               '#17becf', '#2e8b57', '#77dd77', '#ffba00', '#cd00cd']

    for y in (y_real, y_res):
        for i in range(-1, 18):
            one_cluster = []
            for j in range(len(y)):
                if y[j] == i:
                    one_cluster.append(reduced_X[j])
            if len(one_cluster) == 0:
                continue
            one_cluster = np.array(one_cluster).transpose().tolist()
            color = black_color if i == -1 else colours[i]
            plt.scatter(one_cluster[0], one_cluster[1], color=color)
        plt.title(("Истинные" if y == y_real else "Кластеризованные") + " метки. " + "eps = " + str(eps) + ", m = " + str(m))
        plt.show()


def get_clusters_as_map_from_cluster_to_objects_x(object_to_cluster_map):
    clusts = dict()
    for key, value in object_to_cluster_map.items():
        if not (value in clusts):
            clusts[value] = []
        clusts[value].append(key.signs)

    return clusts


def X_and_y_to_list_of_DatasetObject(X, y):
    result = []
    for i in range(len(y)):
        result.append(DatasetObject(X[i], y[i]))

    return result


if __name__ == '__main__':
    dataset_dir = "datasets"
    dataset_name = "dataset_54_vehicle.csv"
    X, y = normalize(pd.read_csv(dataset_dir + "/" + dataset_name))
    dataset = X_and_y_to_list_of_DatasetObject(X, y)
    print_spaces(dataset, eps=1.5, m=6)
    print_index_rand_dependence(dataset, m=5)
    print_silhouette_dependence(dataset, m=10)

