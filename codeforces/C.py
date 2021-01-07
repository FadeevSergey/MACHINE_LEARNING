# score - 1000/1000

import numpy


def euclidean_function(first_vector, second_vector):
    assert(len(first_vector) == len(second_vector))
    return numpy.sqrt(sum([(first_vector[i] - second_vector[i]) ** 2 for i in range(len(first_vector))]))


def manhattan_function(first_vector, second_vector):
    assert(len(first_vector) == len(second_vector))
    return sum([abs(first_vector[i] - second_vector[i]) for i in range(len(first_vector))])


def chebyshev_function(first_vector, second_vector):
    assert(len(first_vector) == len(second_vector))
    return max([abs(first_vector[i] - second_vector[i]) for i in range(len(first_vector))])


def uniform_kernel_function(value):
    return (1/2) if abs(value) < 1 else 0


def triangular_kernel_function(value):
    return (1 - abs(value)) if abs(value) <= 1 else 0


def epanechnikov_kernel_function(value):
    return ((3 / 4) * (1 - value ** 2)) if abs(value) <= 1 else 0


def quartic_kernel_function(value):
    return ((15 / 16) * (1 - value ** 2) ** 2) if abs(value) <= 1 else 0


def triweight_kernel_function(value):
    return ((35 / 32) * ((1 - value ** 2) ** 3)) if abs(value) <= 1 else 0


def tricube_kernel_function(value):
    return ((70 / 81) * (1 - abs(value) ** 3) ** 3) if abs(value) <= 1 else 0


def gaussian_kernel_function(value):
    return 1 / numpy.sqrt(2 * numpy.pi) * (numpy.exp(-1 / 2 * value ** 2))


def cosine_kernel_function(value):
    return (numpy.pi / 4 * numpy.cos(numpy.pi / 2 * value)) if abs(value) <= 1 else 0


def logistic_kernel_function(value):
    return 1 / (numpy.exp(value) + 2 + numpy.exp(-value))


def sigmoid_kernel_function(value):
    return 2 / numpy.pi * (1 / (numpy.exp(value) + numpy.exp(-value)))


def get_dist_function(name):
    if name == "manhattan":
        dist_function = manhattan_function
    elif name == "euclidean":
        dist_function = euclidean_function
    else:
        dist_function = chebyshev_function
    return dist_function


def get_kernel_function(name):
    if name == "uniform":
        kernel_function = uniform_kernel_function
    elif name == "triangular":
        kernel_function = triangular_kernel_function
    elif name == "epanechnikov":
        kernel_function = epanechnikov_kernel_function
    elif name == "quartic":
        kernel_function = quartic_kernel_function
    elif name == "triweight":
        kernel_function = triweight_kernel_function
    elif name == "tricube":
        kernel_function = tricube_kernel_function
    elif name == "gaussian":
        kernel_function = gaussian_kernel_function
    elif name == "cosine":
        kernel_function = cosine_kernel_function
    elif name == "logistic":
        kernel_function = logistic_kernel_function
    else:
        kernel_function = sigmoid_kernel_function
    return kernel_function


if __name__ == '__main__':
    n, m = map(int, input().split())
    objects = []
    for _ in range(n):
        objects.append(list(map(int, input().split())))
    predict_object = list(map(int, input().split()))
    dist_function = get_dist_function(input().replace(" ", ""))
    kernel_function = get_kernel_function(input().replace(" ", ""))
    window_type = input().replace(" ", "")
    window_size = int(input().replace(" ", ""))
    objects.sort(key=lambda cur_object: dist_function(predict_object, cur_object[:-1]))

    distance = [dist_function(predict_object, objects[i][:-1]) for i in range(n)]

    if window_type == "variable":
        window_size = distance[window_size]
    num = 0
    denom = 0

    if window_size == 0:
        for i in range(n):
            if distance[i] == 0:
                num += objects[i][-1]
                denom += 1
    else:
        for i in range(n):
            num += objects[i][-1] * kernel_function(distance[i] / window_size)
            denom += kernel_function(distance[i] / window_size)

    if denom == 0 or distance[0] > window_size:
        num = sum(objects[i][-1] for i in range(n))
        denom = n

    print(num / denom)
 