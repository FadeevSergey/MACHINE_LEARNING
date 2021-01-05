import matplotlib.pyplot
import numpy as np
import random


class DataSet:
    def __init__(self, x, y, number_of_features):
        self.x = x
        self.y = y
        self.number_of_features = number_of_features


def read_dataset(path):
    file = open(path, "r")

    number_of_features = int(file.readline())

    training_sample = get_dataset_from_file(file, number_of_features)
    testing_sample = get_dataset_from_file(file, number_of_features)

    return training_sample, testing_sample


def get_dataset_from_file(dataset, number_of_features):
    number_of_objects = int(dataset.readline())

    x = []
    y = []

    for _ in range(number_of_objects):
        data_object = [int(x) for x in dataset.readline().split()]

        data_object_x = data_object[:number_of_features]
        data_object_x.append(1)

        x.append(data_object_x)
        y.append(data_object[number_of_features])

    return DataSet(x, y, number_of_features)


def init_weight(number_of_features):
    left_limit = -1 / (2 * number_of_features)
    right_limit = 1 / (2 * number_of_features)

    return np.array([random.uniform(left_limit, right_limit) for _ in range(number_of_features + 1)])


def find_gradient(cur_x, cur_y, cur_w):
    grad = []

    for temp_x in cur_x:
        grad.append(2 * (np.array(cur_x).dot(np.array(cur_w)) - cur_y) * temp_x)

    return grad


def update_weights(x, y, w, h):
    rand_object_number = random.randint(0, len(x) - 1)
    x = x[rand_object_number]
    y = y[rand_object_number]

    a = y - np.array(x).dot(w)
    b = np.array(x).dot(find_gradient(x, y, w))

    # print(w)
    # print("h = ", h)
    # print("a = ", a)
    # print("b = ", b)

    if b != 0:
        t = a / b
    else:
        t = 0
    # print("t = ", t)
    return np.array(w) * (1 - h * t) - h * (np.array(find_gradient(x, y, w)) + t * np.linalg.norm(w))
    # return np.array(w) - (h / 10) * np.array(find_gradient(x, y, w))


def stochastic_gradient_descent(dataset, number_of_steps):
    weight = init_weight(dataset.number_of_features)

    for step_number in range(1, number_of_steps + 1):
        h = 1 / step_number

        weight = update_weights(dataset.x, dataset.y, weight, h)

    return weight


def get_nrmse(dataset, w):
    X = np.array(dataset.x)
    Y = np.array(dataset.y)
    sum = np.sum((Y - (X @ w)) ** 2) / Y.size

    y_diff = np.max(Y) - np.min(Y)
    return np.math.sqrt(sum) / y_diff


if __name__ == '__main__':
    path_to_datasets = "datasets/"
    file_format = ".txt"

    for i in range(2, 8):
        train_sample, test_sample = read_dataset(path_to_datasets + str(i) + file_format)

        number_of_steps = 100

        train_result = []
        test_result = []

        steps = []

        while number_of_steps <= 2000:
            model = stochastic_gradient_descent(train_sample, number_of_steps)

            test_result.append(get_nrmse(test_sample, model))

            steps.append(number_of_steps)

            number_of_steps += 100

        matplotlib.pyplot.xlabel("steps")
        matplotlib.pyplot.ylabel("nrmse")
        matplotlib.pyplot.plot(steps, test_result)
        matplotlib.pyplot.title(path_to_datasets + str(i) + file_format)
        matplotlib.pyplot.show()


