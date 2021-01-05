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
    # training_sample = get_normalize_dataset(training_sample)
    testing_sample = get_dataset_from_file(file, number_of_features)
    # testing_sample = get_normalize_dataset(testing_sample)

    return training_sample, testing_sample


def get_dataset_from_file(dataset, number_of_features):
    number_of_objects = int(dataset.readline())

    x = []
    y = []

    for _ in range(number_of_objects):
        data_object = [float(x) for x in dataset.readline().split()]

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

    grad = find_gradient(x, y, w)
    a = np.array(x).dot(w) - y
    b = np.array(x).dot(grad)

    if b != 0:
        t = a / b
    else:
        t = 1

    return np.array(w) - np.array(find_gradient(x, y, w)) * t * h


def stochastic_gradient_descent(dataset, number_of_steps):
    weight = init_weight(dataset.number_of_features)
    for step_number in range(1, number_of_steps + 1):
        h = 1 / step_number

        weight = update_weights(dataset.x, dataset.y, weight, h)

    return weight


def get_nrmse(dataset, w):
    x = np.array(dataset.x)
    y = np.array(dataset.y)
    return np.math.sqrt(np.sum((y - (x @ w)) ** 2) / y.size) / (np.max(y) - np.min(y))


def get_normalize_dataset(dataset):
    dataset.x = np.transpose(dataset.x)
    for k in range(0, len(dataset.x) - 1):
        if (max(dataset.x[k]) - min(dataset.x[k])) == 0:
            for j in range(0, len(dataset.x[k])):
                dataset.x[k][j] = 0
        else:
            dataset.x[k] = np.array(np.array(dataset.x[k]) - min(dataset.x[k])) * (
                    1 / (max(dataset.x[k]) - min(dataset.x[k])))

    dataset.x = np.transpose(dataset.x)
    return dataset


def process_dataset(path):
    train_sample, test_sample = read_dataset(path)
    number_of_steps = 3

    test_result = []
    steps = []
    train_result = []

    while number_of_steps <= 2013:
        model = stochastic_gradient_descent(train_sample, number_of_steps)

        train_result.append(get_nrmse(train_sample, model))
        test_result.append(get_nrmse(test_sample, model))

        steps.append(number_of_steps)

        number_of_steps += 30

    model1 = generalized_inverse(train_sample)
    xxx = [get_nrmse(test_sample, model1)] * len(test_result)

    matplotlib.pyplot.xlabel("steps")
    matplotlib.pyplot.ylabel("nrmse")
    matplotlib.pyplot.plot(steps, xxx)
    matplotlib.pyplot.plot(steps, test_result, steps, xxx, steps, train_result)
    matplotlib.pyplot.title(path)
    matplotlib.pyplot.show()


def generalized_inverse(dataset, l=0.1):
    x = np.array(dataset.x)
    y = np.array(dataset.y)

    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x) + l * np.eye(len(x[0]))), x.transpose()), y)
    # return np.matmul(np.linalg.pinv(x), y)


if __name__ == '__main__':
    path_to_datasets = "./datasets/"
    file_format = ".txt"

    threads = []

    for i in range(2, 8):
        process_dataset(path_to_datasets + str(i) + file_format)
