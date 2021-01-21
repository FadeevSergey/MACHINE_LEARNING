import pandas as pd
import numpy as np
import matplotlib.pyplot
from sklearn.tree import DecisionTreeClassifier


def get_normalize_dataset(data):
    new_dataset = data.copy()

    x = new_dataset.values[:, :-1]
    y = new_dataset.values[:, -1]
    for i in range(len(y)):
        if y[i] == "P":
            y[i] = -1
        else:
            y[i] = 1

    return list(x), list(y)


def weighted_classification_error(classifier, X, y, w):
    result = 0
    size = len(w)
    predicts = classifier(X)
    for i in range(size):
        if y[i] != predicts[i]:
            result += w[i]
    return result


def find_best_classifier(classifiers, X, y, w):
    best_score = -1
    best_classifier = 0
    for classifier in classifiers:
        cur_score = weighted_classification_error(classifier, X, y, w)
        if cur_score < best_score or best_score == -1:
            best_score = cur_score
            best_classifier = classifier
    return best_classifier


def new_alpha(classifier, X, y, w):
    error = weighted_classification_error(classifier, X, y, w)
    if error == 0:
        return 0
    else:
        return 1 / 2 * np.log((1 - error) / error)


def random_DecisionTreeClassifier():
    criterion = "entropy" if np.random.randint(2) == 0 else "gini"
    splitter = "best" if np.random.randint(2) == 0 else "random"
    depth = np.random.randint(1, 10)

    return DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)


def update_weights(w, classifier, alpha, X, y):
    predicts = classifier(X)
    for i in range(len(w)):
        w[i] = w[i] * np.exp(-alpha * y[i] * predicts[i])

    weight_sum = sum(w)
    for i in range(len(w)):
        w[i] /= weight_sum

    return w


def ada_boost_predict(classifiers, weights, x):
    result = 0
    for i in range(len(weights)):
        result += weights[i] * classifiers[i]([x])

    return np.sign(result)


def ada_boost(X, y):
    size = len(y)
    classifiers = []
    w = [1/size] * size

    first = 1
    second = 1

    alphas = []

    accuracy = []

    result_classifiers = []
    for i in range(1, 100):
        new_tree = random_DecisionTreeClassifier()
        new_tree = new_tree.fit(X, y)

        classifiers.append(new_tree.predict)

        cur_best_classifier = find_best_classifier(classifiers, X, y, w)
        result_classifiers.append(cur_best_classifier)
        cur_alpha = new_alpha(cur_best_classifier, X, y, w)

        w = update_weights(w, cur_best_classifier, cur_alpha, X, y)

        alphas.append(cur_alpha)
        cur_accuracy = calc_accuracy(X, y, result_classifiers, alphas)
        if i == second:
            print_space(result_classifiers, alphas, X, y, i, cur_accuracy)
            first, second = second, first
            second = second + first

        accuracy.append(cur_accuracy)

    return accuracy


def calc_accuracy(X, y, classifiers, weights):
    P = 0
    N = 0
    for i in range(len(X)):
        predict = ada_boost_predict(classifiers, weights, X[i])
        if predict == y[i]:
            P += 1
        else:
            N += 1
    return P / (P + N)


def print_space(classifiers, weights, X, y, step, accuracy):
    x1_coord = []
    y1_coord = []
    x2_coord = []
    y2_coord = []

    for i in range(len(X)):
        if y[i] == -1:
            x1_coord.append(X[i][0])
            y1_coord.append(X[i][1])
        else:
            x2_coord.append(X[i][0])
            y2_coord.append(X[i][1])

    matplotlib.pyplot.plot(x1_coord, y1_coord, 'bo')
    matplotlib.pyplot.plot(x2_coord, y2_coord, 'ro')

    ax = matplotlib.pyplot.gca()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    y_pred = []

    for obj in xy:
        y_pred.append(ada_boost_predict(classifiers, weights, obj))
    Z = y_pred
    Z = np.array(Z).reshape(XX.shape)

    ax.contourf(XX, YY, Z, levels=[-100, 0, 100], alpha=0.5, colors=['#0000ff', '#ff0000'])

    ax.contour(XX, YY, Z, levels=[0], alpha=1, linestyles=['-'], colors='k')
    matplotlib.pyplot.title("Step number " + str(step) + ". Accuracy = " + str(accuracy))
    matplotlib.pyplot.show()


def print_graph(results, dataset_names):
    steps = [i for i in range(1, len(results[0]) + 1)]
    matplotlib.pyplot.title("results")
    matplotlib.pyplot.xlabel("steps")
    matplotlib.pyplot.ylabel("accuracy")
    print()
    matplotlib.pyplot.plot(steps, results[0], label=dataset_names[0])
    matplotlib.pyplot.plot(steps, results[1], label=dataset_names[1])
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


if __name__ == '__main__':
    paths_to_dataset = "./datasets/"
    csv_datasets = ["chips.csv", "geyser.csv"]

    results = []

    for dataset_name in csv_datasets:
        dataset = pd.read_csv(paths_to_dataset + dataset_name)
        X, y = get_normalize_dataset(dataset)
        results.append(ada_boost(X, y))

    print_graph(results, csv_datasets)
