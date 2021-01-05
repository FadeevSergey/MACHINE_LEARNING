import pandas
import numpy
import matplotlib.pyplot


def euclidean_function(first_vector, second_vector):
    # numpy.linalg.norm(first_vector - second_vector)
    result_number = 0
    for k in range(0, len(first_vector) - 1):
        result_number += (first_vector[k] - second_vector[k]) ** 2
    return numpy.sqrt(result_number)


def triweight_kernel_function(vector):
    return ((35 / 32) * ((1 - vector ** 2) ** 3)) if abs(vector) <= 1 else 0


def nadar_vatson(n_data_set, vector, distance, window, kernel_function):
    data = n_data_set.loc[:, n_data_set.columns != n_data_set.columns[-1]]

    num = 0
    denom = 0
    for ii, temp_vector in data.iterrows():
        weight = kernel_function(distance(vector, temp_vector) / window)
        num += n_data_set.iloc[ii, -1] * weight
        denom += weight
    return num / denom


def naive_regression(n_data_set, vector, window, distance_function, kernel_function):
    return round(nadar_vatson(n_data_set, vector, distance_function, window, kernel_function))


def one_hot(n_data_set, vector, window, distance_function, kernel_function):
    temp_data_set = n_data_set.copy()
    res = []
    for i in range(0, n_data_set["Class"].max() + 1):
        new_classes = []
        for j in range(len(n_data_set)):
            if n_data_set.iloc[j, -1] == i:
                new_classes.append(1)
            else:
                new_classes.append(0)
        temp_data_set["Class"] = new_classes
        res.append(nadar_vatson(temp_data_set, vector, distance_function, window, kernel_function))
    return numpy.argmax(res)


def get_normalize_data_set(data):
    new_data_set = data.copy()
    for label in data_set.columns:
        if label == "Class":
            new_data_set[label] = pandas.factorize(data_set[label])[0]
        else:
            min_parameter = data_set[label].min()
            max_parameter = data_set[label].max()
            new_data_set[label] = (data_set[label] - min_parameter) / (max_parameter - min_parameter)
    return new_data_set


def leave_one_out(normalize_data_set, window, model, distance, kernel_function):
    correct = 0  # tp and fn
    incorrect = 0  # tn and fp

    for j, vector in normalize_data_set.iterrows():
        nd = normalize_data_set.copy()
        nd.drop(index=j)
        result = model(nd, vector, window, distance, kernel_function)
        if result == normalize_data_set.iloc[j, -1]:
            correct += 1
        else:
            incorrect += 1

    # tp / (tp + fp)
    precision = correct / (correct + incorrect)
    # tp / (tp + fn)
    recall = correct / (correct + incorrect)
    # recall = 1 / 2
    return 2 * ((precision * recall) / (precision + recall))


def calc_naive_regression(n_data_set):
    h = 0.1
    delta_of_h = 0.1
    points_of_x = []
    points_of_y = []

    for i in range(0, 30):
        points_of_x.append(h + delta_of_h * i)
        points_of_y.append(leave_one_out(n_data_set, h + delta_of_h * i, naive_regression,
                                         euclidean_function, triweight_kernel_function))
    matplotlib.pyplot.text(1.5, 0.8, "naive_regression", fontsize=13)
    matplotlib.pyplot.xlabel("h")
    matplotlib.pyplot.ylabel("F")
    matplotlib.pyplot.plot(points_of_x, points_of_y)
    matplotlib.pyplot.show()


def calc_one_hot(n_data_set):
    h = 0.1
    delta_of_h = 0.1
    points_of_x = []
    points_of_y = []

    for i in range(0, 30):
        points_of_x.append(h + delta_of_h * i)
        points_of_y.append(leave_one_out(n_data_set, h + delta_of_h * i, one_hot,
                                         euclidean_function, triweight_kernel_function))

    matplotlib.pyplot.text(1.5, 0.8, "one_hot", fontsize=13)
    matplotlib.pyplot.xlabel("h")
    matplotlib.pyplot.ylabel("F")
    matplotlib.pyplot.plot(points_of_x, points_of_y)
    matplotlib.pyplot.show()


if __name__ == '__main__':
    data_set = pandas.read_csv('dataset_54_vehicleNew.csv')
    normalize_data_set = get_normalize_data_set(data_set)
    calc_naive_regression(normalize_data_set)
    calc_one_hot(normalize_data_set)
